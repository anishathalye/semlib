from collections.abc import Callable

import pydantic
import pytest

from semlib.compare import _DEFAULT_TEMPLATE, Order, Task, compare, compare_sync
from tests.conftest import LLMMocker


def test_compare(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(a=1, b=2): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(a=2, b=1): '{"choice": "A"}',
            _DEFAULT_TEMPLATE.format(a=1, b=1): '{"choice": "neither"}',
        }
    )

    with mocker.patch_llm():
        assert compare_sync(2, 1) == Order.GREATER
        assert compare_sync(1, 2, task="choose_greater_or_abstain") == Order.LESS
        assert compare_sync(1, 1, task="choose_greater_or_abstain") == Order.NEITHER


def test_compare_choose_lesser(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    template = "Which is smaller, (a) {} or (b) {}?"
    mocker = llm_mocker(
        {
            template.format(2, 1): '{"choice": "B"}',
            template.format(1, 2): '{"choice": "A"}',
            template.format(1, 1): '{"choice": "neither"}',
            template.format(3, 9): '{"choice": "neither"}',
        }
    )

    with mocker.patch_llm():
        assert compare_sync(2, 1, template=template, task=Task.CHOOSE_LESSER) == Order.GREATER
        assert compare_sync(2, 1, template=template, task=Task.CHOOSE_LESSER_OR_ABSTAIN) == Order.GREATER
        assert compare_sync(1, 2, template=template, task=Task.CHOOSE_LESSER) == Order.LESS
        assert compare_sync(1, 2, template=template, task=Task.CHOOSE_LESSER_OR_ABSTAIN) == Order.LESS
        assert compare_sync(1, 1, template=template, task=Task.CHOOSE_LESSER_OR_ABSTAIN) == Order.NEITHER
        assert compare_sync(3, 9, template=template, task=Task.CHOOSE_LESSER_OR_ABSTAIN) == Order.NEITHER


def test_compare_compare(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "Consider the numbers (a) 1 and (b) 2. Is (a) greater than (b), or (a) less than (b)?": '{"order": "less"}',
            "Consider the numbers (a) 2 and (b) 1. Is (a) greater than (b), or (a) less than (b)?": '{"order": "greater"}',
            "Consider the numbers (a) 1 and (b) 1. Is (a) greater than (b), (a) less than (b), or neither?": '{"order": "neither"}',
        }
    )

    with mocker.patch_llm():
        assert (
            compare_sync(
                1,
                2,
                template="Consider the numbers (a) {} and (b) {}. Is (a) greater than (b), or (a) less than (b)?",
                task=Task.COMPARE,
            )
            == Order.LESS
        )
        assert (
            compare_sync(
                2,
                1,
                template="Consider the numbers (a) {} and (b) {}. Is (a) greater than (b), or (a) less than (b)?",
                task=Task.COMPARE,
            )
            == Order.GREATER
        )
        assert (
            compare_sync(
                1,
                1,
                template="Consider the numbers (a) {} and (b) {}. Is (a) greater than (b), (a) less than (b), or neither?",
                task=Task.COMPARE_OR_ABSTAIN,
            )
            == Order.NEITHER
        )


@pytest.mark.asyncio
async def test_compare_callable(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    def template(a: int, b: int) -> str:
        return f"Which is greater, (a) {a} or (b) {b}?"

    mocker = llm_mocker(
        {
            template(1, 2): '{"choice": "B"}',
        }
    )

    with mocker.patch_llm():
        result = await compare(1, 2, template=template)
    assert result == Order.LESS


def test_compare_bad_args() -> None:
    with pytest.raises(ValueError, match="if 'task' is not"):
        compare_sync(1, 2, task=Task.CHOOSE_LESSER)
    with pytest.raises(ValueError, match="cannot provide"):
        compare_sync(1, 2, template=lambda a, b: f"which is bigger, (a) {a} or (b) {b}?", to_str=str)
    with pytest.raises(ValueError, match="cannot provide"):
        compare_sync("car", "house", template="which is bigger, (a) {} or (b) {}?", by="weight")


def test_compare_raises(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(a=1, b=2): "not a json",
        }
    )

    with mocker.patch_llm(), pytest.raises(pydantic.ValidationError):
        compare_sync(1, 2)
