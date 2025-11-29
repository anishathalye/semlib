import asyncio
import datetime
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pydantic
import pytest

from semlib.bare import Bare
from semlib.map import map, map_sync  # noqa: A004
from tests.conftest import LLMMocker


def test_map_str(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is the number that comes after one? Reply with a single number, e.g., '5'.": "2",
            "What is the number that comes after two? Reply with a single number, e.g., '5'.": "3",
            "What is the number that comes after three? Reply with a single number, e.g., '5'.": "4",
        }
    )

    with mocker.patch_llm():
        result: list[str] = map_sync(
            ["one", "two", "three"], "What is the number that comes after {}? Reply with a single number, e.g., '5'."
        )

    assert result == ["2", "3", "4"]


def test_map_structured_output(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    class Person(pydantic.BaseModel):
        name: str
        date_of_birth: datetime.date

    mocker = llm_mocker(
        {
            "Who was the 40th president of the United States?": '{"name": "Ronald Reagan", "date_of_birth": "1911-02-06"}',
            "Who was the 41th president of the United States?": '{"name": "George H. W. Bush", "date_of_birth": "1924-06-12"}',
            "Who was the 42th president of the United States?": '{"name": "William Jefferson Clinton", "date_of_birth": "1946-08-19"}',
            "Who was the 43th president of the United States?": '{"name": "George W. Bush", "date_of_birth": "1946-07-06"}',
        }
    )

    with mocker.patch_llm():
        result: list[Person] = map_sync(
            [40, 41, 42, 43], "Who was the {}th president of the United States?", return_type=Person
        )

    assert result == [
        Person(name="Ronald Reagan", date_of_birth=datetime.date(1911, 2, 6)),
        Person(name="George H. W. Bush", date_of_birth=datetime.date(1924, 6, 12)),
        Person(name="William Jefferson Clinton", date_of_birth=datetime.date(1946, 8, 19)),
        Person(name="George W. Bush", date_of_birth=datetime.date(1946, 7, 6)),
    ]


def test_map_callable_formatter(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    class Car(pydantic.BaseModel):
        make: str
        model: str
        year: int

    def template(car: Car) -> str:
        return f"What is the horsepower of a {car.year} {car.make} {car.model}?"

    cars: list[Car] = [
        Car(make="Toyota", model="Camry", year=2020),
        Car(make="Honda", model="Pilot", year=2019),
        Car(make="Ford", model="F-450", year=2021),
    ]

    class Engine(pydantic.BaseModel):
        horsepower: int

    mocker = llm_mocker(
        {
            "What is the horsepower of a 2020 Toyota Camry?": '{"horsepower": 203}',
            "What is the horsepower of a 2019 Honda Pilot?": '{"horsepower": 280}',
            "What is the horsepower of a 2021 Ford F-450?": '{"horsepower": 475}',
        }
    )

    with mocker.patch_llm():
        result: list[Engine] = map_sync(cars, template, return_type=Engine)

    assert result == [Engine(horsepower=203), Engine(horsepower=280), Engine(horsepower=475)]


@pytest.mark.asyncio
async def test_map_async(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is 1 + 1?": "2",
            "What is 1 + 3?": "4",
            "What is 1 + 5?": "6",
        }
    )

    with mocker.patch_llm():
        result: list[str] = await map([1, 3, 5], "What is 1 + {:d}?")

    assert result == ["2", "4", "6"]


def test_map_raises(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is the number that comes after one?": "not json",
            "What is the number that comes after two?": '{"value": 3}',
            "What is the number that comes after three?": '{"value": 4}',
        }
    )

    class Number(pydantic.BaseModel):
        value: int

    with mocker.patch_llm(), pytest.raises(pydantic.ValidationError):
        map_sync(["one", "two", "three"], "What is the number that comes after {}?", return_type=Number)


def test_map_bare(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is the number that comes after one?": '{"value": 2}',
            "What is the number that comes after two?": '{"value": 3}',
            "What is the number that comes after three?": '{"value": 4}',
        }
    )

    with mocker.patch_llm():
        result = map_sync(["one", "two", "three"], "What is the number that comes after {}?", return_type=Bare(int))

    assert result == [2, 3, 4]


@pytest.mark.asyncio
async def test_map_task_cleanup_on_exception() -> None:
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
            await map(["item_1", "item_2"], template="{}")

        await task_cancelled.wait()  # this will hang if the task wasn't cancelled
