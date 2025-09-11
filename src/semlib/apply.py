import asyncio
from collections.abc import Callable
from typing import overload

from pydantic import BaseModel
from pydantic_core import ValidationError  # noqa: F401 # used in docstrings

from semlib._internal.base import Base
from semlib.bare import Bare


class Apply(Base):
    @overload
    async def apply[T, U: BaseModel](
        self,
        item: T,
        /,
        template: str | Callable[[T], str],
        *,
        return_type: type[U],
        model: str | None = None,
    ) -> U: ...

    @overload
    async def apply[T, U](
        self,
        item: T,
        /,
        template: str | Callable[[T], str],
        *,
        return_type: Bare[U],
        model: str | None = None,
    ) -> U: ...

    @overload
    async def apply[T](
        self,
        item: T,
        /,
        template: str | Callable[[T], str],
        *,
        return_type: None = None,
        model: str | None = None,
    ) -> str: ...

    async def apply[T, U: BaseModel, V](
        self,
        item: T,
        /,
        template: str | Callable[[T], str],
        *,
        return_type: type[U] | Bare[V] | None = None,
        model: str | None = None,
    ) -> U | V | str:
        """Apply a language model prompt to a single item.

        This method formats a prompt template with the given item, sends it to the language model, and returns the
        response. The response can be returned as a raw string, parsed into a [Pydantic](https://pydantic.dev/) model,
        or extracted as a bare value using the [Bare][semlib.bare.Bare] marker.

        This method is a simple wrapper around [prompt][semlib._internal.base.Base.prompt].

        Args:
            item: The item to apply the `template` to.
            template: A template to format with the item. This can be either a string template with a single positional
                placeholder, or a callable that takes the item and returns a formatted string.
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
            Basic usage:
            >>> await session.apply(
            ...     [1, 2, 3, 4, 5],
            ...     template="What is the sum of these numbers: {}?",
            ...     return_type=Bare(int),
            ... )
            15
        """
        formatter = template.format if isinstance(template, str) else template

        model = model if model is not None else self._model

        return await self.prompt(formatter(item), return_type=return_type, model=model)


@overload
async def apply[T, U: BaseModel](
    item: T,
    /,
    template: str | Callable[[T], str],
    *,
    return_type: type[U],
    model: str | None = None,
) -> U: ...


@overload
async def apply[T, U](
    item: T,
    /,
    template: str | Callable[[T], str],
    *,
    return_type: Bare[U],
    model: str | None = None,
) -> U: ...


@overload
async def apply[T](
    item: T,
    /,
    template: str | Callable[[T], str],
    *,
    return_type: None = None,
    model: str | None = None,
) -> str: ...


async def apply[T, U: BaseModel, V](
    item: T,
    /,
    template: str | Callable[[T], str],
    *,
    return_type: type[U] | Bare[V] | None = None,
    model: str | None = None,
) -> U | V | str:
    """Standalone version of [apply][semlib.apply.Apply.apply]."""
    applier = Apply(model=model)
    return await applier.apply(item, template, return_type=return_type)


@overload
def apply_sync[T, U: BaseModel](
    item: T,
    /,
    template: str | Callable[[T], str],
    *,
    return_type: type[U],
    model: str | None = None,
) -> U: ...


@overload
def apply_sync[T, U](
    item: T,
    /,
    template: str | Callable[[T], str],
    *,
    return_type: Bare[U],
    model: str | None = None,
) -> U: ...


@overload
def apply_sync[T](
    item: T,
    /,
    template: str | Callable[[T], str],
    *,
    return_type: None = None,
    model: str | None = None,
) -> str: ...


def apply_sync[T, U: BaseModel, V](
    item: T,
    /,
    template: str | Callable[[T], str],
    *,
    return_type: type[U] | Bare[V] | None = None,
    model: str | None = None,
) -> U | V | str:
    """Standalone synchronous version of [apply][semlib.apply.Apply.apply]."""
    applier = Apply(model=model)
    return asyncio.run(applier.apply(item, template, return_type=return_type))  # type: ignore[return-value]
