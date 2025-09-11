import asyncio
from collections.abc import Callable, Iterable
from typing import overload

from pydantic import BaseModel
from pydantic_core import ValidationError  # noqa: F401 # used in docstrings

from semlib._internal import util
from semlib._internal.base import Base
from semlib.bare import Bare


class Map(Base):
    @overload
    async def map[T, U: BaseModel](
        self,
        iterable: Iterable[T],
        /,
        template: str | Callable[[T], str],
        *,
        return_type: type[U],
        model: str | None = None,
    ) -> list[U]: ...

    @overload
    async def map[T, U](
        self,
        iterable: Iterable[T],
        /,
        template: str | Callable[[T], str],
        *,
        return_type: Bare[U],
        model: str | None = None,
    ) -> list[U]: ...

    @overload
    async def map[T](
        self,
        iterable: Iterable[T],
        /,
        template: str | Callable[[T], str],
        *,
        return_type: None = None,
        model: str | None = None,
    ) -> list[str]: ...

    async def map[T, U: BaseModel, V](
        self,
        iterable: Iterable[T],
        /,
        template: str | Callable[[T], str],
        *,
        return_type: type[U] | Bare[V] | None = None,
        model: str | None = None,
    ) -> list[U] | list[V] | list[str]:
        """Map a prompt template over an iterable and get responses from the language model.

        This method applies a prompt template to each item in the provided iterable, sends the resulting prompts to the
        language model, and collects the responses. The responses can be returned as raw strings, parsed into
        [Pydantic](https://pydantic.dev/) models, or extracted as bare values using the [Bare][semlib.bare.Bare] marker.

        This method is analogous to Python's built-in [`map`](https://docs.python.org/3/library/functions.html#map)
        function.

        Args:
            iterable: The collection of items to map over.
            template: A prompt template to apply to each item. This can be either a string template with a single
                positional placeholder, or a callable that takes an item and returns a formatted string.
            return_type: If not specified, the responses are returned as raw strings. If a Pydantic model class is provided,
                the responses are parsed into instances of that model. If a [Bare][semlib.bare.Bare] instance is
                provided, single values of the specified type are extracted from the responses.
            model: If specified, overrides the default model for this call.

        Returns:
            A list of responses from the language model in the format specified by return_type.

        Raises:
            ValidationError: If `return_type` is a Pydantic model or a Bare type and the response cannot be
                parsed into the specified type.

        Examples:
            Basic map:
            >>> await session.map(
            ...     ["apple", "banana", "kiwi"],
            ...     template="What color is {}? Reply in a single word.",
            ... )
            ['Red.', 'Yellow.', 'Green.']

            Map with structured return type:
            >>> class Person(pydantic.BaseModel):
            ...     name: str
            ...     age: int
            >>> await session.map(
            ...     ["Barack Obama", "Angela Merkel"],
            ...     template="Who is {}?",
            ...     return_type=Person,
            ... )
            [Person(name='Barack Obama', age=62), Person(name='Angela Merkel', age=69)]

            Map with bare return type:
            >>> await session.map(
            ...     [42, 1337, 2025],
            ...     template="What are the unique prime factors of {}?",
            ...     return_type=Bare(list[int]),
            ... )
            [[2, 3, 7], [7, 191], [3, 5]]
        """

        formatter = template.format if isinstance(template, str) else template
        model = model if model is not None else self._model
        # case analysis for type checker
        if return_type is None:
            return await util.gather(*[self.prompt(formatter(item), model=model) for item in iterable])
        if isinstance(return_type, Bare):
            return await util.gather(
                *[self.prompt(formatter(item), return_type=return_type, model=model) for item in iterable]
            )
        return await util.gather(
            *[self.prompt(formatter(item), return_type=return_type, model=model) for item in iterable]
        )


@overload
async def map[T, U: BaseModel](  # noqa: A001
    iterable: Iterable[T],
    /,
    template: str | Callable[[T], str],
    *,
    return_type: type[U],
    model: str | None = None,
    max_concurrency: int | None = None,
) -> list[U]: ...


@overload
async def map[T, U](  # noqa: A001
    iterable: Iterable[T],
    /,
    template: str | Callable[[T], str],
    *,
    return_type: Bare[U],
    model: str | None = None,
    max_concurrency: int | None = None,
) -> list[U]: ...


@overload
async def map[T](  # noqa: A001
    iterable: Iterable[T],
    /,
    template: str | Callable[[T], str],
    *,
    return_type: None = None,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> list[str]: ...


async def map[T, U: BaseModel, V](  # noqa: A001
    iterable: Iterable[T],
    /,
    template: str | Callable[[T], str],
    *,
    return_type: type[U] | Bare[V] | None = None,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> list[U] | list[V] | list[str]:
    """Standalone version of [map][semlib.map.Map.map]."""
    mapper = Map(model=model, max_concurrency=max_concurrency)
    return await mapper.map(iterable, template, return_type=return_type)


@overload
def map_sync[T, U: BaseModel](
    iterable: Iterable[T],
    /,
    template: str | Callable[[T], str],
    *,
    return_type: type[U],
    model: str | None = None,
    max_concurrency: int | None = None,
) -> list[U]: ...


@overload
def map_sync[T, U](
    iterable: Iterable[T],
    /,
    template: str | Callable[[T], str],
    *,
    return_type: Bare[U],
    model: str | None = None,
    max_concurrency: int | None = None,
) -> list[U]: ...


@overload
def map_sync[T](
    iterable: Iterable[T],
    /,
    template: str | Callable[[T], str],
    *,
    return_type: None = None,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> list[str]: ...


def map_sync[T, U: BaseModel, V](
    iterable: Iterable[T],
    /,
    template: str | Callable[[T], str],
    *,
    return_type: type[U] | Bare[V] | None = None,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> list[U] | list[V] | list[str]:
    """Standalone synchronous version of [map][semlib.map.Map.map]."""
    mapper = Map(model=model, max_concurrency=max_concurrency)
    return asyncio.run(mapper.map(iterable, template, return_type=return_type))  # type: ignore[return-value]
