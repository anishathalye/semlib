import asyncio
import datetime
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import pydantic
import pytest
from litellm.types.utils import Message

from semlib._internal.base import Base
from semlib._internal.util import gather
from semlib.bare import Bare
from semlib.cache import InMemoryCache
from tests.conftest import LLMMocker


@pytest.mark.asyncio
async def test_base_no_cache(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    base = Base()
    assert base.total_cost() == 0.0

    mocker = llm_mocker(
        {
            "What is 1 + 1?": "2",
            "What is 2 + 2?": "4",
        }
    )

    with mocker.patch_llm():
        result1: str = await base._acompletion(messages=[Message(role="user", content="What is 1 + 1?")])  # noqa: SLF001
        assert result1 == "2"
        assert base.total_cost() == 1.0
        result2: str = await base._acompletion(messages=[Message(role="user", content="What is 2 + 2?")])  # noqa: SLF001
        assert result2 == "4"
        assert base.total_cost() == 2.0
        # repeat query, still incurs cost because no cache
        result2 = await base._acompletion(messages=[Message(role="user", content="What is 2 + 2?")])  # noqa: SLF001
        assert result2 == "4"
        assert base.total_cost() == 3.0


@pytest.mark.asyncio
async def test_base_cache(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    base = Base(cache=InMemoryCache())
    assert base.total_cost() == 0.0

    mocker = llm_mocker(
        {
            "What is 1 + 1?": "2",
            "What is 2 + 2?": "4",
        }
    )

    with mocker.patch_llm():
        result1: str = await base._acompletion(messages=[Message(role="user", content="What is 1 + 1?")])  # noqa: SLF001
        assert result1 == "2"
        assert base.total_cost() == 1.0
        result2: str = await base._acompletion(messages=[Message(role="user", content="What is 2 + 2?")])  # noqa: SLF001
        assert result2 == "4"
        assert base.total_cost() == 2.0
        # repeat query, doesn't incur additional cost
        result2 = await base._acompletion(messages=[Message(role="user", content="What is 2 + 2?")])  # noqa: SLF001
        assert result2 == "4"
        assert base.total_cost() == 2.0
        # but it does when the model is different
        result2 = await base._acompletion(  # noqa: SLF001
            messages=[Message(role="user", content="What is 2 + 2?")], model="gpt-4.1-nano"
        )
        assert result2 == "4"
        assert base.total_cost() == 3.0


@pytest.mark.asyncio
async def test_base_return_type(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    class Person(pydantic.BaseModel):
        name: str
        date_of_birth: datetime.date

    base = Base()
    assert base.total_cost() == 0.0

    mocker = llm_mocker(
        {
            "Who was the 40th president of the United States?": '{"name": "Ronald Reagan", "date_of_birth": "1911-02-06"}',
            "Who was the 41th president of the United States?": '{"name": "George H. W. Bush", "date_of_birth": "1924-06-12"}',
        }
    )

    with mocker.patch_llm():
        result1: Person = await base._acompletion(  # noqa: SLF001
            messages=[Message(role="user", content="Who was the 40th president of the United States?")],
            return_type=Person,
        )
        assert result1 == Person(name="Ronald Reagan", date_of_birth=datetime.date(1911, 2, 6))
        assert base.total_cost() == 1.0
        result2: Person = await base._acompletion(  # noqa: SLF001
            messages=[Message(role="user", content="Who was the 41th president of the United States?")],
            return_type=Person,
        )
        assert result2 == Person(name="George H. W. Bush", date_of_birth=datetime.date(1924, 6, 12))
        assert base.total_cost() == 2.0


@pytest.mark.asyncio
async def test_base_prompt(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is 1 + 1?": "2",
        }
    )
    base = Base()
    with mocker.patch_llm():
        response = await base.prompt("What is 1 + 1?")
    assert response == "2"


@pytest.mark.asyncio
async def test_base_prompt_typed(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    class Number(pydantic.BaseModel):
        value: int

    mocker = llm_mocker(
        {
            "What is 1 + 1?": '{"value": 2}',
        }
    )
    base = Base()
    with mocker.patch_llm():
        response = await base.prompt("What is 1 + 1?", return_type=Number)
    assert response == Number(value=2)


def test_base_ollama() -> None:
    base = Base(model="ollama/gemma3:12b")
    assert base._max_concurrency == 1  # noqa: SLF001


def test_base_bad_args() -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        Base(max_concurrency=-3)


@pytest.mark.asyncio
async def test_base_raises(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    base = Base()

    mocker = llm_mocker(
        {
            "What is 1 + 1?": "2",
        }
    )

    class Number(pydantic.BaseModel):
        value: int

    with mocker.patch_llm(), pytest.raises(pydantic.ValidationError):
        await base._acompletion(  # noqa: SLF001
            messages=[Message(role="user", content="What is 1 + 1?")],
            return_type=Number,
        )


@pytest.mark.asyncio
async def test_prompt_bare_cached(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "Populate the list": '{"primes": [2, 3, 5]}',
        }
    )
    b: Bare[list[int]] = Bare(list[int], class_name="list_of_first_three", field_name="primes")
    base = Base(cache=InMemoryCache())
    with mocker.patch_llm():
        x: list[int] = await base.prompt("Populate the list", return_type=b)
        x = await base.prompt("Populate the list", return_type=b)
    assert x == [2, 3, 5]
    assert base.total_cost() == 1.0


def test_base_env_max_concurrency() -> None:
    import os

    os.environ["SEMLIB_MAX_CONCURRENCY"] = "7"
    base = Base()
    assert base._max_concurrency == 7  # noqa: SLF001

    os.environ["SEMLIB_MAX_CONCURRENCY"] = "not_an_int"
    with pytest.raises(ValueError, match="must be an integer"):
        Base()

    del os.environ["SEMLIB_MAX_CONCURRENCY"]


@pytest.mark.asyncio
async def test_pending_requests_deduplication() -> None:
    """Test that concurrent identical requests wait for the first to complete instead of making duplicate API calls.

    This test relies on timing / asyncio.sleep to catch buggy implementations, but with a correct implementation, it
    should pass regardless of thread scheduling / timing. I don't think there's a good way to test this
    deterministically without glass-box testing. In practice, this should effectively catch buggy implementations (e.g.,
    the Semlib code pre-deduplication).
    """
    call_count = 0

    async def mock_acompletion(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.2)  # simulate slow completion
        response = AsyncMock()
        response.choices = [AsyncMock()]
        response.choices[0].message.content = "42"
        return response

    base = Base(cache=InMemoryCache())

    async def coro() -> str:
        return await base._acompletion(messages=[Message(role="user", content="What is the answer?")])  # noqa: SLF001

    with patch("litellm.acompletion", side_effect=mock_acompletion), patch("litellm.completion_cost", return_value=1.0):
        res1, res2 = await gather(coro(), coro())
        assert res1 == "42"
        assert res2 == "42"
        assert call_count == 1
        assert base.total_cost() == 1.0
