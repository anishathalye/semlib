import asyncio
from collections.abc import Callable, Iterable
from typing import Any, overload

from pydantic import BaseModel
from pydantic_core import ValidationError  # noqa: F401 # used in docstrings

from semlib._internal.base import Base
from semlib._internal.util import gather
from semlib.bare import Bare


class Reduce(Base):
    @overload
    async def reduce(
        self,
        iterable: Iterable[str],
        /,
        template: str | Callable[[str, str], str],
        *,
        associative: bool = False,
        model: str | None = None,
    ) -> str: ...

    @overload
    async def reduce[T](
        self,
        iterable: Iterable[str | T],
        /,
        template: str | Callable[[str | T, str | T], str],
        *,
        associative: bool = False,
        model: str | None = None,
    ) -> str | T: ...

    @overload
    async def reduce[T: BaseModel](
        self,
        iterable: Iterable[T],
        /,
        template: str | Callable[[T, T], str],
        *,
        return_type: type[T],
        associative: bool = False,
        model: str | None = None,
    ) -> T: ...

    @overload
    async def reduce[T](
        self,
        iterable: Iterable[T],
        /,
        template: str | Callable[[T, T], str],
        *,
        return_type: Bare[T],
        associative: bool = False,
        model: str | None = None,
    ) -> T: ...

    @overload
    async def reduce[T, U: BaseModel](
        self,
        iterable: Iterable[T],
        /,
        template: str | Callable[[U, T], str],
        initial: U,
        *,
        return_type: type[U],
        model: str | None = None,
    ) -> U: ...

    @overload
    async def reduce[T, U](
        self,
        iterable: Iterable[T],
        /,
        template: str | Callable[[U, T], str],
        initial: U,
        *,
        return_type: Bare[U],
        model: str | None = None,
    ) -> U: ...

    async def reduce(
        self,
        iterable: Iterable[Any],
        /,
        template: str | Callable[[Any, Any], str],
        initial: Any = None,
        *,
        return_type: Any = None,
        associative: bool = False,
        model: str | None = None,
    ) -> Any:
        """Reduce an iterable to a single value using a language model.

        This method is analogous to Python's
        [`functools.reduce`](https://docs.python.org/3/library/functools.html#functools.reduce) function.

        Args:
            iterable: The collection of items to reduce.
            template: A prompt template to apply to each item. This can be either a string template with two positional
                placeholders (with the first placeholder being the accumulator and the second placeholder being an
                item), or a callable that takes an accumulator and an item and returns a formatted string.
            initial: If provided, this value is placed before the items of the iterable in the calculation, and serves
                as a default when the iterable is empty.
            return_type: The return type is also the type of the accumulator. If not specified, the responses are
                returned as raw strings. If a Pydantic model class is provided, the responses are parsed into instances
                of that model. If a [Bare][semlib.bare.Bare] instance is provided, single values of the specified type
                are extracted from the responses.
            associative: If `True`, the reduction is performed in a balanced tree manner, which unlocks concurrency and
                can provide significant speedups for large iterables. This requires the reduction operation to be
                associative.
            model: If specified, overrides the default model for this call.

        Returns:
            The final accumulated value.

        Raises:
            ValidationError: If `return_type` is a Pydantic model or a Bare type and the response cannot be
                parsed into the specified type.

        Examples:
            Basic reduce:
            >>> await session.reduce(
            ...     ["one", "three", "seven", "twelve"], "{} + {} = ?", return_type=Bare(int)
            ... )
            23

            Reduce with initial value:
            >>> await session.reduce(
            ...     range(20),
            ...     template=lambda acc, n: f"If {n} is prime, append it to this list: {acc}.",
            ...     initial=[],
            ...     return_type=Bare(list[int]),
            ...     model="openai/o4-mini",
            ... )
            [2, 3, 5, 7, 11, 13, 17, 19]

            Associative reduce:
            >>> await session.reduce(
            ...     [[i] for i in range(20)],
            ...     template=lambda acc,
            ...     n: f"Compute the union of these two sets, and then remove any non-prime numbers: {acc} and {n}. Return the result as a list.",
            ...     return_type=Bare(list[int]),
            ...     associative=True,
            ...     model="openai/o4-mini",
            ... )
            [2, 3, 5, 7, 11, 13, 17, 19]

            Distinguishing between leaf nodes and internal nodes in an associative reduce with [Box][semlib.box.Box]:

            >>> reviews: list[str] = [
            ...     "The instructions are a bit confusing. It took me a while to figure out how to use it.",
            ...     "It's so loud!",
            ...     "I regret buying this microwave. It's the worst appliance I've ever owned.",
            ...     "This microwave is great! It heats up food quickly and evenly.",
            ...     "This microwave is a waste of money. It doesn't work at all.",
            ...     "I hate the design of this microwave. It looks cheap and ugly.",
            ...     "The turntable is a bit small, so I can't fit larger plates in it.",
            ...     "The microwave is a bit noisy when it's running.",
            ...     "The microwave is a bit expensive compared to other models with similar features.",
            ...     "The turntable is useless, so I can't fit any plates in it.",
            ...     "I love the sleek design of this microwave. It looks great in my kitchen.",
            ... ]
            >>> def template(a: str | Box[str], b: str | Box[str]) -> str:
            ...     # leaf nodes (raw reviews)
            ...     if isinstance(a, Box) and isinstance(b, Box):
            ...         return f'''
            ... Consider the following two product reviews, and return a bulleted list
            ... summarizing any actionable product improvements that could be made based on
            ... the reviews. If there are no actionable product improvements, return an empty
            ... string.
            ...
            ... - Review 1: {a.value}
            ... - Review 2: {b.value}'''
            ...     # summaries of reviews
            ...     if not isinstance(a, Box) and not isinstance(b, Box):
            ...         return f'''
            ... Consider the following two lists of ideas for product improvements, and
            ... combine them while de-duplicating similar ideas. If there are no ideas, return
            ... an empty string.
            ...
            ... # List 1:
            ... {a}
            ...
            ... # List 2:
            ... {b}'''
            ...     # one is a summary, the other is a raw review
            ...     if isinstance(a, Box) and not isinstance(b, Box):
            ...         ideas = b
            ...         review = a.value
            ...     if not isinstance(a, Box) and isinstance(b, Box):
            ...         ideas = a
            ...         review = b.value
            ...     return f'''
            ... Consider the following list of ideas for product improvements, and a product
            ... review. Update the list of ideas based on the review, de-duplicating similar
            ... ideas. If there are no ideas, return an empty string.
            ...
            ... # List of ideas:
            ... {ideas}
            ...
            ... # Review:
            ... {review}'''
            >>> result = await session.reduce(
            ...     map(Box, reviews), template=template, associative=True
            ... )
            >>> print(result)
            - Clarify and simplify the product instructions to make them easier to understand.
            - Consider reducing the noise level of the product to make it quieter during operation.
            - Improve product reliability to ensure the microwave functions correctly for all users.
            - Increase the size or adjust the design of the turntable to accommodate larger plates.
            - Improve the design to enhance the aesthetic appeal and make it look more premium.
        """
        if initial is not None:
            return await self._reduce2(
                iterable,
                template,
                initial,
                return_type=return_type,
                model=model,
            )
        return await self._reduce1(
            iterable,
            template,
            return_type=return_type,
            associative=associative,
            model=model,
        )

    async def _reduce1(
        self,
        iterable: Iterable[Any],
        /,
        template: str | Callable[[Any, Any], str],
        *,
        return_type: Any,
        associative: bool = False,
        model: str | None = None,
    ) -> Any:
        formatter = template.format if isinstance(template, str) else template
        if not associative:
            it = iter(iterable)
            try:
                acc = next(it)
            except StopIteration:
                msg = "reduce of empty iterable with no initial value"
                raise ValueError(msg) from None
            for item in it:
                acc = await self.prompt(formatter(acc, item), model=model, return_type=return_type)
            return acc
        # associative
        items = list(iterable)
        if len(items) == 0:
            msg = "reduce of empty iterable with no initial value"
            raise ValueError(msg) from None
        if len(items) == 1:
            return items[0]
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]
        left_acc, right_acc = await gather(
            self._reduce1(left, template, return_type=return_type, associative=True, model=model),
            self._reduce1(right, template, return_type=return_type, associative=True, model=model),
        )
        return await self.prompt(formatter(left_acc, right_acc), model=model, return_type=return_type)

    async def _reduce2(
        self,
        iterable: Iterable[Any],
        /,
        template: str | Callable[[Any, Any], str],
        initial: Any,
        *,
        return_type: Any,
        model: str | None = None,
    ) -> Any:
        acc = initial
        formatter = template.format if isinstance(template, str) else template
        for item in iterable:
            acc = await self.prompt(formatter(acc, item), model=model, return_type=return_type)
        return acc


@overload
async def reduce(
    iterable: Iterable[str],
    /,
    template: str | Callable[[str, str], str],
    *,
    associative: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> str: ...


@overload
async def reduce[T](
    iterable: Iterable[str | T],
    /,
    template: str | Callable[[str | T, str | T], str],
    *,
    associative: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> str | T: ...


@overload
async def reduce[T: BaseModel](
    iterable: Iterable[T],
    /,
    template: str | Callable[[T, T], str],
    *,
    return_type: type[T],
    associative: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> T: ...


@overload
async def reduce[T](
    iterable: Iterable[T],
    /,
    template: str | Callable[[T, T], str],
    *,
    return_type: Bare[T],
    associative: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> T: ...


@overload
async def reduce[T, U: BaseModel](
    iterable: Iterable[T],
    /,
    template: str | Callable[[U, T], str],
    initial: U,
    *,
    return_type: type[U],
    model: str | None = None,
    max_concurrency: int | None = None,
) -> U: ...


@overload
async def reduce[T, U](
    iterable: Iterable[T],
    /,
    template: str | Callable[[U, T], str],
    initial: U,
    *,
    return_type: Bare[U],
    model: str | None = None,
    max_concurrency: int | None = None,
) -> U: ...


async def reduce(
    iterable: Iterable[Any],
    /,
    template: str | Callable[[Any, Any], str],
    initial: Any = None,
    *,
    return_type: Any = None,
    associative: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> Any:
    """Standalone version of [reduce][semlib.reduce.Reduce.reduce]."""
    reducer = Reduce(model=model, max_concurrency=max_concurrency)
    return await reducer.reduce(  # type: ignore[call-overload]
        iterable,
        template,
        initial,
        return_type=return_type,
        associative=associative,
    )


@overload
def reduce_sync(
    iterable: Iterable[str],
    /,
    template: str | Callable[[str, str], str],
    *,
    associative: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> str: ...


@overload
def reduce_sync[T](
    iterable: Iterable[str | T],
    /,
    template: str | Callable[[str | T, str | T], str],
    *,
    associative: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> str | T: ...


@overload
def reduce_sync[T: BaseModel](
    iterable: Iterable[T],
    /,
    template: str | Callable[[T, T], str],
    *,
    return_type: type[T],
    associative: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> T: ...


@overload
def reduce_sync[T](
    iterable: Iterable[T],
    /,
    template: str | Callable[[T, T], str],
    *,
    return_type: Bare[T],
    associative: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> T: ...


@overload
def reduce_sync[T, U: BaseModel](
    iterable: Iterable[T],
    /,
    template: str | Callable[[U, T], str],
    initial: U,
    *,
    return_type: type[U],
    model: str | None = None,
    max_concurrency: int | None = None,
) -> U: ...


@overload
def reduce_sync[T, U](
    iterable: Iterable[T],
    /,
    template: str | Callable[[U, T], str],
    initial: U,
    *,
    return_type: Bare[U],
    model: str | None = None,
    max_concurrency: int | None = None,
) -> U: ...


def reduce_sync(
    iterable: Iterable[Any],
    /,
    template: str | Callable[[Any, Any], str],
    initial: Any = None,
    *,
    return_type: Any = None,
    associative: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> Any:
    """Standalone synchronous version of [reduce][semlib.reduce.Reduce.reduce]."""
    reducer = Reduce(model=model, max_concurrency=max_concurrency)
    return asyncio.run(
        reducer.reduce(  # type: ignore[call-overload]
            iterable,
            template,
            initial,
            return_type=return_type,
            associative=associative,
        )
    )
