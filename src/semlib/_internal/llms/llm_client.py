from typing import Union

import litellm
from litellm.caching.caching_handler import CustomStreamWrapper
from litellm.types.utils import Message, ModelResponse

from semlib._internal.llms.custom.claudecode import ClaudeCodeModelResponse, LocalClaudeCode

CUSTOM_MODEL_PROVIDERS = [LocalClaudeCode()]


class LLMClient:
    def __init__(self) -> None:
        litellm.litellm.custom_provider_map += [
            {"provider": model.name(), "custom_handler": model} for model in CUSTOM_MODEL_PROVIDERS
        ]
        pass

    @staticmethod
    async def acompletion(model: str, messages: list[Message], **kwargs) -> Union[ModelResponse, CustomStreamWrapper]:
        return await litellm.acompletion(model=model, messages=messages, **kwargs)

    @staticmethod
    def completion_cost(response: Union[ModelResponse, CustomStreamWrapper]) -> float:
        if isinstance(response, ClaudeCodeModelResponse):
            return response.cost_usd

        return litellm.completion_cost(response)
