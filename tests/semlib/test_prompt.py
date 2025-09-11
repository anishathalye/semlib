from collections.abc import Callable

import pydantic
import pytest

from semlib import Bare
from semlib.prompt import prompt, prompt_sync
from tests.conftest import LLMMocker


def test_prompt_sync(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is 1 + 1?": "2",
        }
    )
    with mocker.patch_llm():
        response = prompt_sync("What is 1 + 1?")
    assert response == "2"


@pytest.mark.asyncio
async def test_prompt_async(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    class Number(pydantic.BaseModel):
        value: int

    mocker = llm_mocker(
        {
            "What is 1 + 1?": '{"value": 2}',
        }
    )
    with mocker.patch_llm():
        response = await prompt("What is 1 + 1?", return_type=Number)
    assert response == Number(value=2)


@pytest.mark.asyncio
async def test_prompt_bare(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "Return the first 3 primes": '{"value": [2, 3, 5]}',
        }
    )
    b: Bare[list[int]] = Bare(list[int])
    with mocker.patch_llm():
        x: list[int] = await prompt("Return the first 3 primes", return_type=b)
    assert x == [2, 3, 5]


@pytest.mark.asyncio
async def test_prompt_bare_named(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "Return the list": '{"primes": [2, 3, 5]}',
        }
    )
    b: Bare[list[int]] = Bare(list[int], class_name="list_of_length_3", field_name="primes")
    with mocker.patch_llm():
        x: list[int] = await prompt("Return the list", return_type=b)
    assert x == [2, 3, 5]
