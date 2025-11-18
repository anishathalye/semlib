import asyncio
import os
from typing import cast, overload

from litellm.types.utils import Message
from pydantic import BaseModel
from pydantic_core import ValidationError  # noqa: F401 # used in docstrings

from semlib._internal.constants import DEFAULT_MODEL
from semlib._internal.llms.llm_client import LLMClient
from semlib._internal.util import parse_max_concurrency
from semlib.bare import Bare
from semlib.cache import QueryCache


class Base:
    def __init__(
        self, *, model: str | None = None, max_concurrency: int | None = None, cache: QueryCache | None = None
    ):
        """Initialize.

        Args:
            model: The language model to use for completions. If not specified, uses the value from the
                `SEMLIB_DEFAULT_MODEL` environment variable, or falls back to the default model (currently
                `"openai/gpt-4o"`). This is used as the `model` argument for
                [litellm](https://docs.litellm.ai/docs/providers) unless overridden in individual method calls.
            max_concurrency: Maximum number of concurrent API requests. If not specified, uses the value from the
                `SEMLIB_MAX_CONCURRENCY` environment variable, or defaults to 10 for most models, or 1 for Ollama
                models.
            cache: If provided, this is used to cache LLM responses to avoid redundant API calls.

        Raises:
            ValueError: If `max_concurrency` is provided but is not a positive integer.
        """
        self._model = model or os.getenv("SEMLIB_DEFAULT_MODEL") or DEFAULT_MODEL
        if max_concurrency is None and (env_max_concurrency := os.getenv("SEMLIB_MAX_CONCURRENCY")) is not None:
            try:
                max_concurrency = int(env_max_concurrency)
            except ValueError:
                msg = "SEMLIB_MAX_CONCURRENCY must be an integer"
                raise ValueError(msg) from None
        self._max_concurrency = parse_max_concurrency(max_concurrency, self._model)
        self._sem = asyncio.Semaphore(self._max_concurrency)
        self._pending_requests: set[bytes] = set()
        self._cond = asyncio.Condition()  # for pending requests deduplication
        self._cache = cache
        self._llm_client = LLMClient()

        self._total_cost: float = 0.0

    def total_cost(self) -> float:
        """Get the total cost incurred so far for API calls made through this instance.

        Returns:
            The total cost in USD.
        """
        return self._total_cost

    def _add_cost(self, cost: float) -> None:
        self._total_cost += cost

    @overload
    async def _acompletion[T: BaseModel](
        self, *, messages: list[Message], return_type: type[T], model: str | None = None
    ) -> T: ...

    @overload
    async def _acompletion[T](
        self, *, messages: list[Message], return_type: Bare[T], model: str | None = None
    ) -> T: ...

    @overload
    async def _acompletion(
        self,
        *,
        messages: list[Message],
        return_type: None = None,
        model: str | None = None,
    ) -> str: ...

    async def _acompletion[T: BaseModel, U](
        self, *, messages: list[Message], model: str | None = None, return_type: type[T] | Bare[U] | None = None
    ) -> str | T | U:
        model = model if model is not None else self._model
        cache_key = (messages, return_type._model if isinstance(return_type, Bare) else return_type, model)  # noqa: SLF001

        # check cache / pending requests
        if self._cache is not None:
            hashed_key = self._cache._hash_key(cache_key)  # noqa: SLF001
            async with self._cond:
                # if there's already a request in flight for this key, wait for it to complete
                while hashed_key in self._pending_requests:
                    await self._cond.wait()
                # then, check the cache
                if (cached_response := self._cache._get(cache_key)) is not None:  # noqa: SLF001
                    # cache hit
                    if return_type is None:
                        return cached_response
                    if isinstance(return_type, Bare):
                        return return_type._extract(return_type._model.model_validate_json(cached_response))  # noqa: SLF001
                    return return_type.model_validate_json(cached_response)
                # if we've gotten to this point, we need to make a request; mark it as pending
                self._pending_requests.add(hashed_key)

        # make the request
        if return_type is None:
            kwargs = {}
        elif isinstance(return_type, Bare):
            kwargs = {"response_format": return_type._model}  # noqa: SLF001
        else:
            kwargs = {"response_format": return_type}
        try:
            async with self._sem:
                response = await self._llm_client.acompletion(model=model, messages=messages, **kwargs)
            self._add_cost(self._llm_client.completion_cost(response))
            content = cast(str, response.choices[0].message.content)
            if return_type is None:
                typed_response: str | T | U = content
            elif isinstance(return_type, Bare):
                typed_response = return_type._extract(return_type._model.model_validate_json(content))  # noqa: SLF001
            else:
                typed_response = return_type.model_validate_json(content)
            # cache after parsing, to ensure we don't cache a broken response
            if self._cache is not None:
                async with self._cond:
                    self._cache._set(cache_key, content)  # noqa: SLF001
            return typed_response
        finally:
            if self._cache is not None:
                async with self._cond:
                    self._pending_requests.remove(hashed_key)
                    self._cond.notify_all()

    @property
    def model(self) -> str:
        """Get the current model being used for completions.

        Returns:
            The model name as a string.
        """
        return self._model

    def clear_cache(self) -> None:
        """Clear the internal cache of LLM responses, if caching is enabled."""
        if self._cache is not None:
            self._cache.clear()

    @overload
    async def prompt[T: BaseModel](self, prompt: str, /, *, return_type: type[T], model: str | None = None) -> T: ...

    @overload
    async def prompt[T](self, prompt: str, /, *, return_type: Bare[T], model: str | None = None) -> T: ...

    @overload
    async def prompt(self, prompt: str, /, *, return_type: None = None, model: str | None = None) -> str: ...

    async def prompt[T: BaseModel, U](
        self, prompt: str, /, *, return_type: type[T] | Bare[U] | None = None, model: str | None = None
    ) -> str | T | U:
        """Send a prompt to the language model and get a response.

        This method sends a single user message to the language model and returns the response. The response can be
        returned as a raw string, parsed into a [Pydantic](https://pydantic.dev/) model, or extracted as a bare value
        using the [Bare][semlib.bare.Bare] marker.

        Args:
            prompt: The text prompt to send to the language model.
            return_type: If not specified, the response is returned as a raw string. If a Pydantic model class is
                provided, the response is parsed into an instance of that model. If a [Bare][semlib.bare.Bare] instance
                is provided, a single value of the specified type is extracted from the response.
            model: If specified, overrides the default model for this call.

        Returns:
            The language model's response in the format specified by return_type.

        Raises:
            ValidationError: If `return_type` is a Pydantic model or a Bare type and the response cannot be
                parsed into the specified type.

        Examples:
            Get raw string response:
            >>> await session.prompt("What is 2+2?")
            '2 + 2 equals 4.'

            Get structured value:
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> await session.prompt("Who is Barack Obama?", return_type=Person)
            Person(name='Barack Obama', age=62)

            Get bare value:
            >>> await session.prompt("What is 2+2?", return_type=Bare(int))
            4
        """
        return await self._acompletion(
            messages=[Message(role="user", content=prompt)], return_type=return_type, model=model
        )
