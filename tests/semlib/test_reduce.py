from collections.abc import Callable

import pydantic
import pytest

from semlib.bare import Bare
from semlib.box import Box
from semlib.reduce import reduce, reduce_sync
from tests.conftest import LLMMocker


def test_reduce_str(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is 0 + 1? Respond with just the number.": "1",
            "What is 1 + 2? Respond with just the number.": "3",
            "What is 3 + 3? Respond with just the number.": "6",
            "What is 6 + 4? Respond with just the number.": "10",
        }
    )
    with mocker.patch_llm():
        result: str = reduce_sync(map(str, range(5)), template="What is {} + {}? Respond with just the number.")
    assert result == "10"


def test_reduce_str_associative(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is 0 + 1? Respond with just the number.": "1",
            "What is 3 + 4? Respond with just the number.": "7",
            "What is 2 + 7? Respond with just the number.": "9",
            "What is 1 + 9? Respond with just the number.": "10",
        }
    )
    with mocker.patch_llm():
        result = reduce_sync(
            map(str, range(5)), template="What is {} + {}? Respond with just the number.", associative=True
        )
    assert result == "10"


@pytest.mark.asyncio
@pytest.mark.parametrize("associative", [True, False])
async def test_reduce_str_fn(llm_mocker: Callable[[dict[str, str]], LLMMocker], associative: bool) -> None:  # noqa: FBT001
    mocker = llm_mocker(
        {
            "What is 0 + 1? Respond with just the number.": "1",
            "What is 1 + 2? Respond with just the number.": "3",
            "What is 3 + 3? Respond with just the number.": "6",
            "What is 6 + 4? Respond with just the number.": "10",
            "What is 3 + 4? Respond with just the number.": "7",
            "What is 2 + 7? Respond with just the number.": "9",
            "What is 1 + 9? Respond with just the number.": "10",
        }
    )
    with mocker.patch_llm():
        result = await reduce(
            map(str, range(5)),
            template=lambda a, b: f"What is {a} + {b}? Respond with just the number.",
            associative=associative,
        )
    assert result == "10"


@pytest.mark.asyncio
async def test_reduce_t_fn(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is 0 + 1? Respond with just the number.": '{"value": 1}',
            "What is 1 + 2? Respond with just the number.": '{"value": 3}',
            "What is 3 + 3? Respond with just the number.": '{"value": 6}',
            "What is 6 + 4? Respond with just the number.": '{"value": 10}',
        }
    )

    class Number(pydantic.BaseModel):
        value: int

    with mocker.patch_llm():
        result: Number = await reduce(
            (Number(value=n) for n in range(5)),
            template=lambda a, b: f"What is {a.value} + {b.value}? Respond with just the number.",
            return_type=Number,
            associative=False,
        )

    assert result == Number(value=10)


@pytest.mark.asyncio
async def test_reduce_t_fn_associative(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is 0 + 1? Respond with just the number.": '{"value": 1}',
            "What is 3 + 4? Respond with just the number.": '{"value": 7}',
            "What is 2 + 7? Respond with just the number.": '{"value": 9}',
            "What is 1 + 9? Respond with just the number.": '{"value": 10}',
        }
    )

    class Number(pydantic.BaseModel):
        value: int

    with mocker.patch_llm():
        result: Number = await reduce(
            (Number(value=n) for n in range(5)),
            template=lambda a, b: f"What is {a.value} + {b.value}? Respond with just the number.",
            return_type=Number,
            associative=True,
        )

    assert result == Number(value=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("associative", [True, False])
async def test_reduce_t(llm_mocker: Callable[[dict[str, str]], LLMMocker], associative: bool) -> None:  # noqa: FBT001
    mocker = llm_mocker(
        {
            "What is value=0 + value=1? Respond with just the number.": '{"value": 1}',
            "What is value=1 + value=2? Respond with just the number.": '{"value": 3}',
            "What is value=3 + value=3? Respond with just the number.": '{"value": 6}',
            "What is value=6 + value=4? Respond with just the number.": '{"value": 10}',
            "What is value=3 + value=4? Respond with just the number.": '{"value": 7}',
            "What is value=2 + value=7? Respond with just the number.": '{"value": 9}',
            "What is value=1 + value=9? Respond with just the number.": '{"value": 10}',
        }
    )

    class Number(pydantic.BaseModel):
        value: int

    with mocker.patch_llm():
        result: Number = await reduce(
            (Number(value=n) for n in range(5)),
            template="What is {} + {}? Respond with just the number.",
            return_type=Number,
            associative=associative,
        )

    assert result == Number(value=10)


def test_reduce_tu(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "If 0 is prime, append it to this list: [].": '{"items": []}',
            "If 1 is prime, append it to this list: [].": '{"items": []}',
            "If 2 is prime, append it to this list: [].": '{"items": [2]}',
            "If 3 is prime, append it to this list: [2].": '{"items": [2, 3]}',
            "If 4 is prime, append it to this list: [2, 3].": '{"items": [2, 3]}',
            "If 5 is prime, append it to this list: [2, 3].": '{"items": [2, 3, 5]}',
            "If 6 is prime, append it to this list: [2, 3, 5].": '{"items": [2, 3, 5]}',
            "If 7 is prime, append it to this list: [2, 3, 5].": '{"items": [2, 3, 5, 7]}',
            "If 8 is prime, append it to this list: [2, 3, 5, 7].": '{"items": [2, 3, 5, 7]}',
            "If 9 is prime, append it to this list: [2, 3, 5, 7].": '{"items": [2, 3, 5, 7]}',
        }
    )

    class List(pydantic.BaseModel):
        items: list[int]

    with mocker.patch_llm():
        result = reduce_sync(
            range(10),
            template=lambda acc, n: f"If {n} is prime, append it to this list: {acc.items}.",
            initial=List(items=[]),
            return_type=List,
        )

    assert result == List(items=[2, 3, 5, 7])


@pytest.mark.asyncio
async def test_reduce_tu2(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "Add Stanford to this set of universities: [].": '{"universities": ["Stanford"]}',
            "Add MIT to this set of universities: ['Stanford'].": '{"universities": ["Stanford", "MIT"]}',
            "Add Harvard to this set of universities: ['Stanford', 'MIT'].": '{"universities": ["Stanford", "MIT", "Harvard"]}',
            "Add MIT to this set of universities: ['Stanford', 'MIT', 'Harvard'].": '{"universities": ["Stanford", "MIT", "Harvard"]}',
        }
    )

    class Person(pydantic.BaseModel):
        name: str
        university: str

    people = [
        Person(name="David Mazieres", university="Stanford"),
        Person(name="Frans Kaashoek", university="MIT"),
        Person(name="Eddie Kohler", university="Harvard"),
        Person(name="Nickolai Zeldovich", university="MIT"),
    ]

    class Universities(pydantic.BaseModel):
        universities: list[str]

    with mocker.patch_llm():
        result = await reduce(
            people,
            template=lambda acc, p: f"Add {p.university} to this set of universities: {acc.universities}.",
            initial=Universities(universities=[]),
            return_type=Universities,
        )

    assert result == Universities(universities=["Stanford", "MIT", "Harvard"])


def test_reduce_bare(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is 1 + 2? Respond with just the number.": '{"value": 3}',
            "What is 3 + 4? Respond with just the number.": '{"value": 7}',
            "What is 3 + 7? Respond with just the number.": '{"value": 10}',
        }
    )

    def template(a: int, b: int) -> str:
        return f"What is {a} + {b}? Respond with just the number."

    with mocker.patch_llm():
        result = reduce_sync(
            [1, 2, 3, 4],
            template=template,
            associative=True,
            return_type=Bare(int),
        )
    assert result == 10


def test_reduce_bare_2(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "If 1 is prime, append it to this list: [].": '{"value": []}',
            "If 2 is prime, append it to this list: [].": '{"value": [2]}',
            "If 3 is prime, append it to this list: [2].": '{"value": [2, 3]}',
            "If 4 is prime, append it to this list: [2, 3].": '{"value": [2, 3]}',
            "If 5 is prime, append it to this list: [2, 3].": '{"value": [2, 3, 5]}',
            "If 6 is prime, append it to this list: [2, 3, 5].": '{"value": [2, 3, 5]}',
        }
    )

    def template(acc: list[int], n: int) -> str:
        return f"If {n} is prime, append it to this list: {acc}."

    with mocker.patch_llm():
        result = reduce_sync(
            [1, 2, 3, 4, 5, 6],
            template=template,
            return_type=Bare(list[int]),
            initial=[],
        )
    assert result == [2, 3, 5]


def test_reduce_raises() -> None:
    with pytest.raises(ValueError, match="empty iterable"):
        reduce_sync([], template="What is {} + {}?")
    with pytest.raises(ValueError, match="empty iterable"):
        reduce_sync([], template="What is {} + {}?", associative=True)

    class Number(pydantic.BaseModel):
        value: int

    with pytest.raises(ValueError, match="empty iterable"):
        reduce_sync([], template="What is {} + {}?", return_type=Number)
    with pytest.raises(ValueError, match="empty iterable"):
        reduce_sync([], template="What is {} + {}?", return_type=Number, associative=True)


def test_reduce_box(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "Construct a set from the numbers '1' and 'two'.": "{1, 2}",
            "Construct a set from the numbers '2' and 'three'.": "{2, 3}",
            "Add the number '3' to the set {2, 3}.": "{2, 3}",
            "Compute the union of the sets {1, 2} and {2, 3}.": "{1, 2, 3}",
        }
    )

    items: list[str] = ["1", "two", "3", "2", "three"]

    def template(a: Box[str] | str, b: Box[str] | str) -> str:
        if isinstance(a, Box) and isinstance(b, Box):
            return f"Construct a set from the numbers '{a.value}' and '{b.value}'."
        if not isinstance(a, Box) and not isinstance(b, Box):
            return f"Compute the union of the sets {a} and {b}."
        if isinstance(a, Box):
            a, b = b, a
        assert isinstance(b, Box)
        assert isinstance(a, str)
        return f"Add the number '{b.value}' to the set {a}."

    with mocker.patch_llm():
        result: Box[str] | str = reduce_sync(map(Box, items), template=template, associative=True)
    assert result == "{1, 2, 3}"
