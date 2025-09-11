import asyncio
from collections.abc import Callable, Iterable

from pydantic_core import ValidationError  # noqa: F401 # used in docstrings

from semlib._internal.base import Base
from semlib.filter import _DEFAULT_TEMPLATE, _Decision


class Find(Base):
    async def find[T](
        self,
        iterable: Iterable[T],
        /,
        *,
        by: str | None = None,
        to_str: Callable[[T], str] | None = None,
        template: str | Callable[[T], str] | None = None,
        negate: bool = False,
        model: str | None = None,
    ) -> T | None:
        """Find an item in an iterable based on a criteria.

        This method searches through the provided iterable and returns some item (not necessarily the first) that
        matches the specified criteria.

        Args:
            iterable: The collection of items to search.
            by: A criteria specifying a predicate to search by. If this is provided, `template` cannot be provided.
            to_str: If specified, used to convert items to string representation. Otherewise, uses `str()` on each item.
                If this is provided, a callable template cannot be provided.
            template: A custom prompt template for predicates. Must be either a string template with a single positional
                placeholder, or a callable that takes an item and returns a formatted string. If this is provided, `by`
                cannot be provided.
            negate: If `True`, find an item that does **not** match the criteria. If `False`, find an item that does
                match the criteria.
            model: If specified, overrides the default model for this call.

        Returns:
            An item from the iterable if it matches the criteria, or `None` if no such item is found.

        Raises:
            ValidationError: If parsing any LLM response fails.

        Examples:
            Basic find:
            >>> await session.find(["Tom Hanks", "Tom Cruise", "Tom Brady"], by="actor?")
            'Tom Cruise'  # nondeterministic, could also return "Tom Hanks"

            Custom template:
            >>> await session.find(
            ...     [(123, 321), (384, 483), (134, 431)],
            ...     template=lambda pair: f"Is {pair[0]} backwards {pair[1]}?",
            ...     negate=True,
            ... )
            (384, 483)
        """
        if template is None:
            if by is None:
                msg = "must specify either 'by' or 'template'"
                raise ValueError(msg)
        else:
            if callable(template) and to_str is not None:
                msg = "cannot provide 'to_str' when a template function is provided"
                raise ValueError(msg)
            if by is not None:
                msg = "cannot provide 'by' when a custom template is provided"
                raise ValueError(msg)

        to_str = to_str if to_str is not None else str

        if template is None:

            def map_template(item: T, /) -> str:
                return _DEFAULT_TEMPLATE.format(by=by or "", item=to_str(item))
        elif isinstance(template, str):

            def map_template(item: T, /) -> str:
                return template.format(to_str(item))
        else:
            # callable
            map_template = template

        model = model if model is not None else self._model

        async def fn(item: T) -> tuple[T, bool]:
            decision = await self.prompt(
                map_template(item),
                return_type=_Decision,
                model=model,
            )
            if negate:
                decision.decision = not decision.decision
            return item, decision.decision

        tasks: list[asyncio.Task[tuple[T, bool]]] = [asyncio.create_task(fn(item)) for item in iterable]
        try:
            for next_finished in asyncio.as_completed(tasks):
                item, decision = await next_finished
                if decision:
                    return item
            return None
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.wait(tasks)


async def find[T](
    iterable: Iterable[T],
    /,
    *,
    by: str | None = None,
    to_str: Callable[[T], str] | None = None,
    template: str | Callable[[T], str] | None = None,
    negate: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> T | None:
    """Standalone version of [find][semlib.find.Find.find]."""
    finder = Find(model=model, max_concurrency=max_concurrency)
    result = await finder.find(iterable, by=by, to_str=to_str, template=template, negate=negate)
    # binding to intermediate variable coro to avoid mypy bug, see https://github.com/python/mypy/issues/19716 and
    # https://github.com/python/mypy/pull/19767 (fixed now, but not shipped yet)
    return result  # noqa: RET504


def find_sync[T](
    iterable: Iterable[T],
    /,
    *,
    by: str | None = None,
    to_str: Callable[[T], str] | None = None,
    template: str | Callable[[T], str] | None = None,
    negate: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> T | None:
    """Standalone synchronous version of [find][semlib.find.Find.find]."""
    finder = Find(model=model, max_concurrency=max_concurrency)
    coro = finder.find(iterable, by=by, to_str=to_str, template=template, negate=negate)
    # binding to intermediate variable coro to avoid mypy bug, see https://github.com/python/mypy/issues/19716 and
    # https://github.com/python/mypy/pull/19767 (fixed now, but not shipped yet)
    return asyncio.run(coro)
