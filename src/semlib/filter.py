import asyncio
from collections.abc import Callable, Iterable

from pydantic import BaseModel
from pydantic_core import ValidationError  # noqa: F401 # used in docstrings

from semlib.map import Map


class _Decision(BaseModel):
    decision: bool


_DEFAULT_TEMPLATE = """
Given a criteria and an item, determine whether the item meets the criteria.

<criteria>
{by}
</criteria>

<item>
{item}
</item>
""".strip()


class Filter(Map):
    async def filter[T](
        self,
        iterable: Iterable[T],
        /,
        *,
        by: str | None = None,
        to_str: Callable[[T], str] | None = None,
        template: str | Callable[[T], str] | None = None,
        negate: bool = False,
        model: str | None = None,
    ) -> list[T]:
        """Filter an iterable based on a criteria.

        This method is analogous to Python's built-in
        [`filter`](https://docs.python.org/3/library/functions.html#filter) function.

        Args:
            iterable: The collection of items to filter.
            by: A criteria specifying a predicate to filter by. If this is provided, `template` cannot be provided.
            to_str: If specified, used to convert items to string representation. Otherewise, uses `str()` on each item.
                If this is provided, a callable template cannot be provided.
            template: A custom prompt template for predicates. Must be either a string template with a single positional
                placeholder, or a callable that takes an item and returns a formatted string. If this is provided, `by`
                cannot be provided.
            negate: If `True`, keep items that do **not** match the criteria. If `False`, keep items that match the
                criteria.
            model: If specified, overrides the default model for this call.

        Returns:
            A new list containing items from the iterable if they match the criteria.

        Raises:
            ValidationError: If parsing any LLM response fails.

        Examples:
            Basic filter:
            >>> await session.filter(["Tom Hanks", "Tom Cruise", "Tom Brady"], by="actor?")
            ['Tom Hanks', 'Tom Cruise']

            Custom template:
            >>> await session.filter(
            ...     [(123, 321), (384, 483), (134, 431)],
            ...     template=lambda pair: f"Is {pair[0]} backwards {pair[1]}?",
            ...     negate=True,
            ... )
            [(384, 483)]
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

        decisions = await self.map(iterable, map_template, return_type=_Decision, model=model)
        return [
            item
            for item, decision in zip(iterable, decisions, strict=False)
            if ((not decision.decision) if negate else decision.decision)
        ]


async def filter[T](  # noqa: A001
    iterable: Iterable[T],
    /,
    *,
    by: str | None = None,
    to_str: Callable[[T], str] | None = None,
    template: str | Callable[[T], str] | None = None,
    negate: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> list[T]:
    """Standalone version of [filter][semlib.filter.Filter.filter]."""
    filterer = Filter(model=model, max_concurrency=max_concurrency)
    return await filterer.filter(iterable, by=by, to_str=to_str, template=template, negate=negate)


def filter_sync[T](
    iterable: Iterable[T],
    /,
    *,
    by: str | None = None,
    to_str: Callable[[T], str] | None = None,
    template: str | Callable[[T], str] | None = None,
    negate: bool = False,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> list[T]:
    """Standalone synchronous version of [filter][semlib.filter.Filter.filter]."""
    filterer = Filter(model=model, max_concurrency=max_concurrency)
    return asyncio.run(filterer.filter(iterable, by=by, to_str=to_str, template=template, negate=negate))
