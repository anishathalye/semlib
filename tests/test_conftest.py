from collections.abc import Callable

import litellm
import pytest

from tests.conftest import LLMMocker


@pytest.mark.asyncio
async def test_exact_match_returns_expected_response(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    response_map = {"Hello world": "Hi there", "What is 2+2?": "4"}
    mocker = llm_mocker(response_map)

    with mocker.patch_llm():
        response = await litellm.acompletion(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}]
        )
        assert response.choices[0].message.content == "Hi there"


@pytest.mark.asyncio
async def test_no_match_raises_error(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    response_map = {"Hello world": "Hi there"}
    mocker = llm_mocker(response_map)

    with mocker.patch_llm(), pytest.raises(ValueError, match="no mock response found for prompt: Goodbye world"):
        await litellm.acompletion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Goodbye world"}])


@pytest.mark.asyncio
async def test_partial_match_does_not_work(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    response_map = {"Hello": "Hi there"}
    mocker = llm_mocker(response_map)

    with mocker.patch_llm(), pytest.raises(ValueError, match="no mock response found for prompt: Hello world"):
        await litellm.acompletion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}])
