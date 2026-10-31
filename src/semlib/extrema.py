import asyncio
from collections.abc import Callable, Iterable

from pydantic_core import ValidationError  # noqa: F401 # used in docstrings

from semlib._internal import util
from semlib.compare import Compare, Order, Task


class Extrema(Compare):
    async def _get_extreme[T](
        self,
        iterable: list[T],
        /,
        *,
        find_min: bool,
        by: str | None = None,
        to_str: Callable[[T], str] | None = None,
        template: str | Callable[[T, T], str] | None = None,
        task: Task | str | None = None,
        model: str | None = None,
    ) -> T:
        if not iterable:
            msg = "iterable argument is empty"
            raise ValueError(msg)

        async def compare(a: T, b: T) -> Order:
            return await self.compare(
                a,
                b,
                by=by,
                to_str=to_str,
                template=template,
                task=task,
                model=model,
            )

        async def rec(lst: list[T]) -> T:
            if len(lst) == 1:
                return lst[0]
            mid = len(lst) // 2
            left, right = await util.gather(rec(lst[:mid]), rec(lst[mid:]))
            comparison = await compare(left, right)
            if (find_min and comparison == Order.LESS) or (not find_min and comparison == Order.GREATER):
                return left
            if (find_min and comparison == Order.GREATER) or (not find_min and comparison == Order.LESS):
                return right
            # neither is greater; we arbitrarily choose to be left-biased
            return left

        return await rec(iterable)

    async def min[T](
        self,
        iterable: Iterable[T],
        /,
        *,
        by: str | None = None,
        to_str: Callable[[T], str] | None = None,
        template: str | Callable[[T, T], str] | None = None,
        task: Task | str | None = None,
        model: str | None = None,
    ) -> T:
        """Get the smallest item in an iterable.

        This method finds the smallest item in a collection by using a language model to perform pairwise comparisons.

        This method is analogous to Python's built-in [`min`](https://docs.python.org/3/library/functions.html#min)
        function.

        Args:
            iterable: The collection of items to search.
            by: A criteria specifying what aspect to compare by. If this is provided, `template` cannot be
                provided.
            to_str: If specified, used to convert items to string representation. Otherewise, uses `str()` on each item.
                If this is provided, a callable template cannot be provided.
            template: A custom prompt template for comparisons. Must be either a string template with two positional
                placeholders, or a callable that takes two items and returns a formatted string. If this is provided,
                `by` cannot be provided.
            task: The type of comparison task that is being performed in `template`. This allows for writing the
                template in the most convenient way possible (e.g., in some scenarios, it's easier to specify a criteria
                for which item is lesser, and in others, it's easier to specify a criteria for which item is greater).
                If this is provided, a custom `template` must also be provided.  Defaults to
                [Task.CHOOSE_GREATER][semlib.compare.Task.CHOOSE_GREATER] if not specified.
            model: If specified, overrides the default model for this call.

        Returns:
            The smallest item.

        Raises:
            ValidationError: If parsing any LLM response fails.

        Examples:
            Basic usage:
            >>> await session.min(["blue", "red", "green"], by="wavelength")
            'blue'
        """
        return await self._get_extreme(
            list(iterable),
            find_min=True,
            by=by,
            to_str=to_str,
            template=template,
            task=task,
            model=model,
        )

    async def max[T](
        self,
        iterable: Iterable[T],
        /,
        *,
        by: str | None = None,
        to_str: Callable[[T], str] | None = None,
        template: str | Callable[[T, T], str] | None = None,
        task: Task | str | None = None,
        model: str | None = None,
    ) -> T:
        """Get the largest item in an iterable.

        This method finds the largest item in a collection by using a language model to perform pairwise comparisons.

        This method is analogous to Python's built-in [`max`](https://docs.python.org/3/library/functions.html#max)
        function.

        Args:
            iterable: The collection of items to search.
            by: A criteria specifying what aspect to compare by. If this is provided, `template` cannot be
                provided.
            to_str: If specified, used to convert items to string representation. Otherewise, uses `str()` on each item.
                If this is provided, a callable template cannot be provided.
            template: A custom prompt template for comparisons. Must be either a string template with two positional
                placeholders, or a callable that takes two items and returns a formatted string. If this is provided,
                `by` cannot be provided.
            task: The type of comparison task that is being performed in `template`. This allows for writing the
                template in the most convenient way possible (e.g., in some scenarios, it's easier to specify a criteria
                for which item is lesser, and in others, it's easier to specify a criteria for which item is greater).
                If this is provided, a custom `template` must also be provided.  Defaults to
                [Task.CHOOSE_GREATER][semlib.compare.Task.CHOOSE_GREATER] if not specified.
            model: If specified, overrides the default model for this call.

        Returns:
            The largest item.

        Raises:
            ValidationError: If parsing any LLM response fails.

        Examples:
            Basic usage:
            >>> await session.max(
            ...     ["LeBron James", "Kobe Bryant", "Magic Johnson"], by="assists"
            ... )
            'Magic Johnson'
        """
        return await self._get_extreme(
            list(iterable),
            find_min=False,
            by=by,
            to_str=to_str,
            template=template,
            task=task,
            model=model,
        )


async def min[T](  # noqa: A001
    iterable: Iterable[T],
    /,
    *,
    by: str | None = None,
    to_str: Callable[[T], str] | None = None,
    template: str | Callable[[T, T], str] | None = None,
    task: Task | str | None = None,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> T:
    """Standalone version of [min][semlib.extrema.Extrema.min]."""
    extrema = Extrema(model=model, max_concurrency=max_concurrency)
    return await extrema.min(iterable, by=by, to_str=to_str, template=template, task=task)


def min_sync[T](
    iterable: Iterable[T],
    /,
    *,
    by: str | None = None,
    to_str: Callable[[T], str] | None = None,
    template: str | Callable[[T, T], str] | None = None,
    task: Task | str | None = None,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> T:
    """Standalone synchronous version of [min][semlib.extrema.Extrema.min]."""
    extrema = Extrema(model=model, max_concurrency=max_concurrency)
    return asyncio.run(extrema.min(iterable, by=by, to_str=to_str, template=template, task=task))


async def max[T](  # noqa: A001
    iterable: Iterable[T],
    /,
    *,
    by: str | None = None,
    to_str: Callable[[T], str] | None = None,
    template: str | Callable[[T, T], str] | None = None,
    task: Task | str | None = None,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> T:
    """Standalone version of [max][semlib.extrema.Extrema.max]."""
    extrema = Extrema(model=model, max_concurrency=max_concurrency)
    return await extrema.max(iterable, by=by, to_str=to_str, template=template, task=task)


def max_sync[T](
    iterable: Iterable[T],
    /,
    *,
    by: str | None = None,
    to_str: Callable[[T], str] | None = None,
    template: str | Callable[[T, T], str] | None = None,
    task: Task | str | None = None,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> T:
    """Standalone synchronous version of [max][semlib.extrema.Extrema.max]."""
    extrema = Extrema(model=model, max_concurrency=max_concurrency)
    return asyncio.run(extrema.max(iterable, by=by, to_str=to_str, template=template, task=task))
