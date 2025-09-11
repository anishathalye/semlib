import datetime
from collections.abc import Callable

import pydantic
import pytest

from semlib.apply import apply, apply_sync
from semlib.bare import Bare
from tests.conftest import LLMMocker


def test_apply_sync(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    class Person(pydantic.BaseModel):
        name: str
        date_of_birth: datetime.date

    people = [
        Person(name="George Washington", date_of_birth=datetime.date(1732, 2, 22)),
        Person(name="John Adams", date_of_birth=datetime.date(1735, 10, 30)),
        Person(name="Thomas Jefferson", date_of_birth=datetime.date(1743, 4, 13)),
        Person(name="James Madison", date_of_birth=datetime.date(1751, 3, 16)),
        Person(name="James Monroe", date_of_birth=datetime.date(1758, 4, 28)),
    ]

    class Count(pydantic.BaseModel):
        count: int

    mocker = llm_mocker(
        {
            "Of the following people, how many have names that start with 'J'?\nGeorge Washington,John Adams,Thomas Jefferson,James Madison,James Monroe": '{"count": 3}',
        }
    )
    with mocker.patch_llm():
        result: Count = apply_sync(
            people,
            lambda people: f"Of the following people, how many have names that start with 'J'?\n{','.join([p.name for p in people])}",
            return_type=Count,
        )

    assert result == Count(count=3)


def test_apply_sync_item(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    class Person(pydantic.BaseModel):
        name: str
        date_of_birth: datetime.date

    person = Person(name="George Washington", date_of_birth=datetime.date(1732, 2, 22))

    class BirthYear(pydantic.BaseModel):
        year: int

    mocker = llm_mocker(
        {
            "What year was George Washington born?": '{"year": 1732}',
        }
    )
    with mocker.patch_llm():
        result: BirthYear = apply_sync(
            person,
            lambda p: f"What year was {p.name} born?",
            return_type=BirthYear,
        )

    assert result == BirthYear(year=1732)


def test_apply_sync_str(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is 7 + 12?": "19",
        }
    )
    with mocker.patch_llm():
        result: str = apply_sync(12, "What is 7 + {}?")

    assert result == "19"


@pytest.mark.asyncio
async def test_apply_async(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is 7 + 12?": "19",
        }
    )
    with mocker.patch_llm():
        result: str = await apply(
            (7, 12),
            lambda xs: "What is {} + {}?".format(*xs),
        )

    assert result == "19"


@pytest.mark.asyncio
async def test_apply_async_raises(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    class Count(pydantic.BaseModel):
        count: int

    mocker = llm_mocker(
        {
            "What is 7 + 12?": "nineteen",
        }
    )
    with mocker.patch_llm(), pytest.raises(pydantic.ValidationError):
        await apply(
            (7, 12),
            lambda xs: "What is {} + {}?".format(*xs),
            return_type=Count,
        )


def test_apply_sync_bare(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is 7 + 12?": '{"value": 19}',
        }
    )
    with mocker.patch_llm():
        result = apply_sync(12, "What is 7 + {}?", return_type=Bare(int))

    assert result == 19
