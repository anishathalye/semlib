from collections.abc import Callable

import pydantic
import pytest

from semlib.filter import _DEFAULT_TEMPLATE, filter, filter_sync
from tests.conftest import LLMMocker


@pytest.mark.parametrize("negate", [True, False])
def test_filter(llm_mocker: Callable[[dict[str, str]], LLMMocker], negate: bool) -> None:  # noqa: FBT001
    people = ["Barack Obama", "Miley Cyrus", "Jeff Dean", "Jeff Bezos", "Bill Clinton", "Elon Musk"]
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(item="Barack Obama", by="president of the United States"): '{"decision": true}',
            _DEFAULT_TEMPLATE.format(item="Miley Cyrus", by="president of the United States"): '{"decision": false}',
            _DEFAULT_TEMPLATE.format(item="Jeff Dean", by="president of the United States"): '{"decision": false}',
            _DEFAULT_TEMPLATE.format(item="Jeff Bezos", by="president of the United States"): '{"decision": false}',
            _DEFAULT_TEMPLATE.format(item="Bill Clinton", by="president of the United States"): '{"decision": true}',
            _DEFAULT_TEMPLATE.format(item="Elon Musk", by="president of the United States"): '{"decision": false}',
        }
    )
    with mocker.patch_llm():
        result = filter_sync(people, by="president of the United States", negate=negate)

    if negate:
        assert result == ["Miley Cyrus", "Jeff Dean", "Jeff Bezos", "Elon Musk"]
    else:
        assert result == ["Barack Obama", "Bill Clinton"]


def test_filter_template_str(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
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
        result = filter_sync(people, template="Is {} a famous computer scientist?")
    assert result == ["Jeff Dean"]


@pytest.mark.asyncio
async def test_filter_callable(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
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
        result = await filter(pairs, template=lambda pair: f"Is {pair[0]} the reverse of {pair[1]}?")
    assert result == [(123, 321), (12, 21)]


def test_filter_bad_args() -> None:
    with pytest.raises(ValueError, match="must specify either"):
        filter_sync([1, 2, 3])
    with pytest.raises(ValueError, match="cannot provide"):
        filter_sync([1, 2, 3], template=lambda x: f"Is {x} odd?", to_str=str)
    with pytest.raises(ValueError, match="cannot provide"):
        filter_sync([1, 2, 3], by="odd", template="Is {} odd?")


def test_filter_raises(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    people = ["Barack Obama", "Miley Cyrus", "Jeff Dean"]
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(item="Barack Obama", by="president of the United States"): "not json",
            _DEFAULT_TEMPLATE.format(item="Miley Cyrus", by="president of the United States"): '{"decision": false}',
            _DEFAULT_TEMPLATE.format(item="Jeff Dean", by="president of the United States"): '{"decision": false}',
        }
    )
    with mocker.patch_llm(), pytest.raises(pydantic.ValidationError):
        filter_sync(people, by="president of the United States")
