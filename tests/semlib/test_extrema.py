from collections.abc import Callable

import pydantic
import pytest

from semlib.compare import _DEFAULT_TEMPLATE, _DEFAULT_TEMPLATE_BY
from semlib.extrema import max, max_sync, min, min_sync  # noqa: A004
from tests.conftest import LLMMocker


def test_min_max(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(a="1", b="3"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(a="3", b="2"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE.format(a="1", b="2"): '{"choice": "B"}',
        }
    )

    with mocker.patch_llm():
        biggest = max_sync([1, 3, 2])
        smallest = min_sync([1, 3, 2])

    assert biggest == 3
    assert smallest == 1


@pytest.mark.asyncio
async def test_min_max_async(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(a="1", b="3"): '{"choice": "B"}',
            _DEFAULT_TEMPLATE.format(a="3", b="2"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE.format(a="1", b="2"): '{"choice": "B"}',
        }
    )

    with mocker.patch_llm():
        biggest = await max([1, 3, 2])
        smallest = await min([1, 3, 2])

    assert biggest == 3
    assert smallest == 1


def test_equal(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    class Person(pydantic.BaseModel):
        name: str
        age: int

    people = [Person(name="Alice", age=30), Person(name="Bob", age=30), Person(name="Charlie", age=25)]
    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(a="30", b="30"): '{"choice": "neither"}',
            _DEFAULT_TEMPLATE.format(a="30", b="25"): '{"choice": "A"}',
        }
    )

    with mocker.patch_llm():
        oldest: Person = max_sync(people, to_str=lambda p: str(p.age), task="choose_greater_or_abstain")

    assert oldest == people[0]  # left-biased


def test_min_max_callable(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    class University(pydantic.BaseModel):
        name: str
        year_founded: int

    universities = [
        University(name="Harvard", year_founded=1636),
        University(name="Stanford", year_founded=1885),
        University(name="MIT", year_founded=1861),
    ]

    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE_BY.format(a="Harvard", b="Stanford", criteria="endowment"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE_BY.format(a="Stanford", b="MIT", criteria="endowment"): '{"choice": "A"}',
            _DEFAULT_TEMPLATE_BY.format(a="Harvard", b="MIT", criteria="endowment"): '{"choice": "A"}',
        }
    )

    with mocker.patch_llm():
        biggest: University = max_sync(universities, by="endowment", to_str=lambda u: u.name)
        smallest: University = min_sync(universities, by="endowment", to_str=lambda u: u.name)

    assert biggest == universities[0]  # Harvard
    assert smallest == universities[2]  # MIT


def test_min_max_empty() -> None:
    with pytest.raises(ValueError, match="iterable argument is empty"):
        max_sync([])

    with pytest.raises(ValueError, match="iterable argument is empty"):
        min_sync([])


def test_min_raises(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    class Item(pydantic.BaseModel):
        name: str
        value: int

    items = [Item(name="A", value=10), Item(name="B", value=20)]

    mocker = llm_mocker(
        {
            _DEFAULT_TEMPLATE.format(a="10", b="20"): "invalid json",
        }
    )

    with mocker.patch_llm(), pytest.raises(pydantic.ValidationError):
        min_sync(items, to_str=lambda item: str(item.value))
