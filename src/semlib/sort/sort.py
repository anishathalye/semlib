import asyncio
from collections.abc import Callable, Iterable

from pydantic_core import ValidationError  # noqa: F401 # used in docstrings

from semlib.compare import Compare, Order, Task
from semlib.sort.algorithm import Algorithm, BordaCount


class Sort(Compare):
    async def sort[T](
        self,
        iterable: Iterable[T],
        /,
        *,
        by: str | None = None,
        to_str: Callable[[T], str] | None = None,
        template: str | Callable[[T, T], str] | None = None,
        task: Task | str | None = None,
        algorithm: Algorithm | None = None,
        reverse: bool = False,
        model: str | None = None,
    ) -> list[T]:
        """Sort an iterable.

        This method sorts a collection of items by using a language model to perform pairwise comparisons. The sorting
        algorithm determines which comparisons to make and how to aggregate the results into a final ranking.

        This method is analogous to Python's built-in
        [`sorted`](https://docs.python.org/3/library/functions.html#sorted) function.

        Args:
            iterable: The collection of items to sort.
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
            algorithm: The sorting algorithm to use. If not specified, defaults to
                [BordaCount][semlib.sort.algorithm.BordaCount]. Different algorithms make different tradeoffs between
                accuracy, latency, and cost. See the documentation for each algorithm for details.
            reverse: If `True`, sort in descending order. If `False`, sort in ascending order.
            model: If specified, overrides the default model for this call.

        Returns:
            A new list containing all items from the iterable in sort order.

        Raises:
            ValidationError: If parsing any LLM response fails.

        Examples:
            Basic sort:
            >>> await session.sort(["blue", "red", "green"], by="wavelength", reverse=True)
            ['red', 'green', 'blue']

            Custom template and task:
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Person:
            ...     name: str
            ...     job: str
            >>> people = [
            ...     Person(name="Barack Obama", job="President of the United States"),
            ...     Person(name="Dalai Lama", job="Spiritual Leader of Tibet"),
            ...     Person(name="Sundar Pichai", job="CEO of Google"),
            ... ]
            >>> await session.sort(
            ...     people,
            ...     template=lambda a, b: f"Which job earns more, (a) {a.job} or (b) {b.job}?",
            ... )
            [
                Person(name='Dalai Lama', job='Spiritual Leader of Tibet'),
                Person(name='Barack Obama', job='President of the United States'),
                Person(name='Sundar Pichai', job='CEO of Google'),
            ]
        """
        algorithm = algorithm if algorithm is not None else BordaCount()

        async def comparator(a: T, b: T) -> Order:
            return await self.compare(a, b, by=by, to_str=to_str, template=template, task=task, model=model)

        return await algorithm._sort(  # noqa: SLF001
            iterable, reverse=reverse, comparator=comparator, max_concurrency=self._max_concurrency
        )


async def sort[T](
    iterable: Iterable[T],
    /,
    *,
    by: str | None = None,
    to_str: Callable[[T], str] | None = None,
    template: str | Callable[[T, T], str] | None = None,
    task: Task | str | None = None,
    algorithm: Algorithm | None = None,
    reverse: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> list[T]:
    """Standalone version of [sort][semlib.sort.Sort.sort]."""
    sorter = Sort(
        model=model,
        max_concurrency=max_concurrency,
    )
    return await sorter.sort(
        iterable, by=by, to_str=to_str, template=template, task=task, algorithm=algorithm, reverse=reverse
    )


def sort_sync[T](
    iterable: Iterable[T],
    /,
    *,
    by: str | None = None,
    to_str: Callable[[T], str] | None = None,
    template: str | Callable[[T, T], str] | None = None,
    task: Task | str | None = None,
    algorithm: Algorithm | None = None,
    reverse: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> list[T]:
    """Standalone synchronous version of [sort][semlib.sort.Sort.sort]."""
    sorter = Sort(
        model=model,
        max_concurrency=max_concurrency,
    )
    return asyncio.run(
        sorter.sort(iterable, by=by, to_str=to_str, template=template, task=task, algorithm=algorithm, reverse=reverse)
    )
