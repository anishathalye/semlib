import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
from typing import Union

from litellm import CustomLLM, ModelResponse
from litellm.caching.caching_handler import CustomStreamWrapper
from litellm.types.utils import Message

DEFAULT_TIMEOUT_SECONDS = 60 * 30  # 30 minutes


class LocalClaudeCode(CustomLLM):
    @staticmethod
    def name() -> str:
        return "claudecode"

    @staticmethod
    async def acompletion(model: str, messages: list[Message], **kwargs) -> Union[ModelResponse, CustomStreamWrapper]:
        try:
            process = await asyncio.create_subprocess_exec(
                "claude",
                f"--model={model}",
                "-p",
                json.dumps(messages),
                "--output-format=json",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy()
            )

            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(),
                timeout=kwargs.get('timeout', DEFAULT_TIMEOUT_SECONDS)
            )

            if process.returncode != 0:
                raise Exception(f"Claude CLI failed: {stderr_data.decode()}")

            raw_response = stdout_data.decode().strip()
            parsed = ClaudeCLIResponseObj.from_json(raw_response)

            input_tokens = int(parsed.usage.input_tokens or 0)
            output_tokens = int(parsed.usage.output_tokens or 0)

            response = ClaudeCodeModelResponse(
                id=f"{LocalClaudeCode.name()}_{uuid.uuid4().hex[:8]}",
                model=model,
                created=int(time.time()),
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": parsed.result
                        },
                        "finish_reason": "stop"
                    }
                ],
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cache_creation_input_tokens": int(parsed.usage.cache_creation_input_tokens or 0),
                    "cache_read_input_tokens": int(parsed.usage.cache_read_input_tokens or 0),
                },
                cost_usd=parsed.total_cost_usd,
            )

            return response

        except asyncio.TimeoutError:
            raise Exception("Claude CLI command timed out")
        except FileNotFoundError:
            raise Exception("Claude CLI not found. Please ensure 'claude' command is available in PATH")
        except Exception as e:
            raise Exception(f"Error executing Claude CLI: {str(e)}")


@dataclass
class ClaudeCLIUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            input_tokens=int(d.get("input_tokens", d.get("prompt_tokens", 0) or 0) or 0),
            output_tokens=int(d.get("output_tokens", d.get("completion_tokens", 0) or 0) or 0),
            cache_creation_input_tokens=int(d.get("cache_creation_input_tokens", 0) or 0),
            cache_read_input_tokens=int(d.get("cache_read_input_tokens", 0) or 0),
        )


@dataclass
class ClaudeCLIResponseObj:
    result: str = ""
    usage: ClaudeCLIUsage = field(default_factory=ClaudeCLIUsage)
    total_cost_usd: float = 0.0
    subtype: Optional[str] = None

    @classmethod
    def from_json(cls, raw: str):
        try:
            data = json.loads(raw) if raw else {}
        except Exception:
            data = {}
        usage = ClaudeCLIUsage.from_dict(data.get("usage", {}))
        return cls(
            result=str(data.get("result", "")),
            usage=usage,
            total_cost_usd=float(data.get("total_cost_usd", 0.0) or 0.0),
            subtype=data.get("subtype"),
        )


class ClaudeCodeModelResponse(ModelResponse):
    cost_usd: float = 0.0
