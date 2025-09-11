import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import pydantic
import pytest

from semlib.filter import _DEFAULT_TEMPLATE
from semlib.find import Find, find, find_sync
from tests.conftest import LLMMocker


@pytest.mark.parametrize("negate", [True, False])
def test_find(llm_mocker: Callable[[dict[str, str]], LLMMocker], negate: bool) -> None:  # noqa: FBT001
    people = ["Barack Obama", "Miley Cyrus", "Jeff Dean", "Jeff Bezos", "Bill Clinton", "Elon Musk"]
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(item="Barack Obama", by="singer?"): '{"decision": false}',
            _DEFAULT_TEMPLATE.format(item="Miley Cyrus", by="singer?"): '{"decision": true}',
            _DEFAULT_TEMPLATE.format(item="Jeff Dean", by="singer?"): '{"decision": false}',
            _DEFAULT_TEMPLATE.format(item="Jeff Bezos", by="singer?"): '{"decision": false}',
            _DEFAULT_TEMPLATE.format(item="Bill Clinton", by="singer?"): '{"decision": false}',
            _DEFAULT_TEMPLATE.format(item="Elon Musk", by="singer?"): '{"decision": false}',
        }
    )
    with mocker.patch_llm():
        result = find_sync(people, by="singer?", negate=negate)

    if negate:
        assert result != "Miley Cyrus"
    else:
        assert result == "Miley Cyrus"


def test_find_not_found(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    people = ["Miley Cyrus", "Jeff Dean"]
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(item="Miley Cyrus", by="president of the United States"): '{"decision": false}',
            _DEFAULT_TEMPLATE.format(item="Jeff Dean", by="president of the United States"): '{"decision": false}',
        }
    )
    with mocker.patch_llm():
        result = find_sync(people, by="president of the United States")
    assert result is None


def test_find_template_str(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    people = ["Barack Obama", "Miley Cyrus", "Jeff Dean", "Jeff Bezos", "Bill Clinton", "Elon Musk"]
    mocker = llm_mocker(
        {
            "Is Barack Obama a famous computer scientist?": '{"decision": false}',
            "Is Miley Cyrus a famous computer scientist?": '{"decision": false}',
            "Is Jeff Dean a famous computer scientist?": '{"decision": true}',
            "Is Jeff Bezos a famous computer scientist?": '{"decision": false}',
            "Is Bill Clinton a famous computer scientist?": '{"decision": false}',
            "Is Elon Musk a famous computer scientist?": '{"decision": false}',
        }
    )
    with mocker.patch_llm():
        result = find_sync(people, template="Is {} a famous computer scientist?")
    assert result == "Jeff Dean"


@pytest.mark.asyncio
async def test_find_callable(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    pairs = [
        (123, 321),
        (1838, 1883),
        (33, 44),
        (12, 21),
    ]
    mocker = llm_mocker(
        {
            "Is 123 the reverse of 321?": '{"decision": true}',
            "Is 12 the reverse of 21?": '{"decision": true}',
            "Is 1838 the reverse of 1883?": '{"decision": false}',
            "Is 33 the reverse of 44?": '{"decision": false}',
        }
    )
    with mocker.patch_llm():
        result = await find(pairs, template=lambda pair: f"Is {pair[0]} the reverse of {pair[1]}?")
    assert result in {(123, 321), (12, 21)}


@pytest.mark.asyncio
async def test_find_cancellation() -> None:
    async def deadlock() -> None:
        q: asyncio.Queue[None] = asyncio.Queue()
        await q.get()

    async def mock_acompletion(*_args: Any, **kwargs: Any) -> Any:
        if kwargs.get("messages", [])[0].get("content", "") == "Is 1 odd?":
            response = AsyncMock()
            response.choices = [AsyncMock()]
            response.choices[0].message.content = '{"decision": true}'
            return response
        await deadlock()
        return None

    finder = Find(max_concurrency=2)
    with patch("litellm.acompletion", side_effect=mock_acompletion), patch("litellm.completion_cost", return_value=1.0):
        result = await finder.find([1, 2, 3, 4, 5], template="Is {} odd?")
    assert result == 1
    assert finder.total_cost() == 1.0


def test_find_bad_args() -> None:
    with pytest.raises(ValueError, match="must specify either"):
        find_sync([1, 2, 3])
    with pytest.raises(ValueError, match="cannot provide"):
        find_sync([1, 2, 3], template=lambda x: f"Is {x} odd?", to_str=str)
    with pytest.raises(ValueError, match="cannot provide"):
        find_sync([1, 2, 3], by="odd", template="Is {} odd?")


def test_find_raises(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    people = ["Barack Obama", "Miley Cyrus", "Jeff Dean"]
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(item="Barack Obama", by="president of the United States"): "not json",
            _DEFAULT_TEMPLATE.format(item="Miley Cyrus", by="president of the United States"): '{"decision": false}',
            _DEFAULT_TEMPLATE.format(item="Jeff Dean", by="president of the United States"): '{"decision": false}',
        }
    )

    with mocker.patch_llm(), pytest.raises(pydantic.ValidationError):
        find_sync(people, by="president of the United States")


@pytest.mark.asyncio
async def test_find_task_cleanup_on_exception() -> None:
    task_cancelled = asyncio.Event()

    async def mock_acompletion(*_args: Any, **kwargs: Any) -> Any:
        content = kwargs.get("messages", [{}])[0].get("content", "")

        if "item_1" in content:
            # first task fails after a short delay
            await asyncio.sleep(0.1)
            msg = "Mock API error"
            raise ValueError(msg)

        # "item_2" case: other task takes a long time to complete
        try:
            q: asyncio.Queue[None] = asyncio.Queue()
            await q.get()
        except asyncio.CancelledError:
            task_cancelled.set()
            raise

    with patch("litellm.acompletion", side_effect=mock_acompletion), patch("litellm.completion_cost", return_value=0.0):
        with pytest.raises(ValueError, match="Mock API error"):
            await find(["item_1", "item_2"], by="test")

        await task_cancelled.wait()  # this will hang if the task wasn't cancelled
