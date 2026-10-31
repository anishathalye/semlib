import os
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


class LLMMocker:
    def __init__(self, response_map: dict[str, str]):
        self.response_map = response_map

    def _create_mock_response(self, content: str) -> AsyncMock:
        response = AsyncMock()
        response.choices = [AsyncMock()]
        response.choices[0].message.content = content
        return response

    async def mock_acompletion(self, *_args: Any, **kwargs: Any) -> AsyncMock:
        messages = kwargs.get("messages", [])
        if messages and isinstance(messages, list) and len(messages) > 0:
            prompt = messages[0].get("content", "")

            if prompt in self.response_map:
                return self._create_mock_response(self.response_map[prompt])

            msg = f"no mock response found for prompt: {prompt}"
            raise ValueError(msg)
        msg = "no content in LLM query"
        raise ValueError(msg)

    def patch_llm(self) -> ExitStack:
        def context_manager() -> ExitStack:
            stack = ExitStack()
            if os.getenv("SEMLIB_TEST_REAL_LLM") != "1":
                stack.enter_context(patch("litellm.acompletion", side_effect=self.mock_acompletion))
            stack.enter_context(patch("litellm.completion_cost", return_value=1.0))
            return stack

        return context_manager()


@pytest.fixture
def llm_mocker() -> Callable[[dict[str, str]], LLMMocker]:
    def _create_mocker(response_map: dict[str, str]) -> LLMMocker:
        return LLMMocker(response_map)

    return _create_mocker
