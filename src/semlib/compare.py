import asyncio
from collections.abc import Callable
from enum import Enum
from typing import cast

from pydantic import BaseModel
from pydantic_core import ValidationError  # noqa: F401 # used in docstrings

from semlib._internal.base import Base

_DEFAULT_TEMPLATE = """
Given two items, determine which item is greater.

<item A>
{a}
</item A>

<item B>
{b}
</item B>
""".strip()

_DEFAULT_TEMPLATE_BY = """
Given a criteria and two items, determine which item is greater according to the criteria.

<criteria>
{criteria}
</criteria>

<item A>
{a}
</item A>

<item B>
{b}
</item B>
""".strip()


class Task(str, Enum):
    """Comparison task to perform.

    Intended to be passed to [compare][semlib.compare.Compare.compare] and similar methods, this specifies how the LLM
    should compare two items.
    """

    COMPARE = "compare"
    """Ask the model to compare two items and determine their relative order.

    The model must choose either `"less"` or `"greater"`.
    """

    COMPARE_OR_ABSTAIN = "compare_or_abstain"
    """Ask the model to compare two items and determine their relative order, or abstain if unsure.

    The model must choose `"less"`, `"greater"`, or `"neither"`.
    """

    CHOOSE_GREATER = "choose_greater"
    """Ask the model to choose which of the two items (a) or (b) is greater.

    The model must choose either `"A"` or `"B"`.
    """

    CHOOSE_GREATER_OR_ABSTAIN = "choose_greater_or_abstain"
    """Ask the model to choose which of the two items (a) or (b) is greater, or abstain if unsure.

    The model must choose `"A"`, `"B"`, or `"neither"`.
    """

    CHOOSE_LESSER = "choose_lesser"
    """Ask the model to choose which of the two items (a) or (b) is lesser.

    The model must choose either `"A"` or `"B"`.
    """

    CHOOSE_LESSER_OR_ABSTAIN = "choose_lesser_or_abstain"
    """Ask the model to choose which of the two items (a) or (b) is lesser, or abstain if unsure.

    The model must choose `"A"`, `"B"`, or `"neither"`.
    """


class Order(str, Enum):
    """Result of a comparison."""

    LESS = "less"
    """Item A is less than Item B."""

    GREATER = "greater"
    """Item A is greater than Item B."""

    NEITHER = "neither"
    """Items A and B are equivalent, or the model abstained from choosing."""


class _StrictOrder(str, Enum):
    LESS = "less"
    GREATER = "greater"

    def to_order(self) -> Order:
        match self:
            case _StrictOrder.LESS:
                return Order.LESS
            case _StrictOrder.GREATER:
                return Order.GREATER


class _CompareResult(BaseModel):
    order: Order


class _StrictCompareResult(BaseModel):
    order: _StrictOrder


class _Choice(str, Enum):
    A = "A"
    B = "B"
    NEITHER = "neither"


class _StrictChoice(str, Enum):
    A = "A"
    B = "B"


class _ChooseResult(BaseModel):
    choice: _Choice


class _StrictChooseResult(BaseModel):
    choice: _StrictChoice


_RETURN_TYPE_BY_TASK: dict[Task, type[BaseModel]] = {
    Task.COMPARE: _StrictCompareResult,
    Task.COMPARE_OR_ABSTAIN: _CompareResult,
    Task.CHOOSE_GREATER: _StrictChooseResult,
    Task.CHOOSE_GREATER_OR_ABSTAIN: _ChooseResult,
    Task.CHOOSE_LESSER: _StrictChooseResult,
    Task.CHOOSE_LESSER_OR_ABSTAIN: _ChooseResult,
}


class Compare(Base):
    async def compare[T](
        self,
        a: T,
        b: T,
        /,
        *,
        by: str | None = None,
        to_str: Callable[[T], str] | None = None,
        template: str | Callable[[T, T], str] | None = None,
        task: Task | str | None = None,
        model: str | None = None,
    ) -> Order:
        """Compare two items.

        This method uses a language model to compare two items and determine the relative ordering of the two items.
        The comparison can be customized by specifying either a criteria to compare by, or a custom prompt template. The
        comparison task can be framed in a number of ways (choosing the greater item, lesser item, or the ordering).

        Args:
            a: The first item to compare.
            b: The second item to compare.
            by: A criteria specifying what aspect to compare by. If this is provided, `template` cannot be
                provided.
            to_str: If specified, used to convert items to string representation. Otherewise, uses `str()` on each item.
                If this is provided, a callable template cannot be provided.
            template: A custom prompt template for the comparison. Must be either a string template with two positional
                placeholders, or a callable that takes two items and returns a formatted string. If this is provided,
                `by` cannot be provided.
            task: The type of comparison task that is being performed in `template`. This allows for writing the
                template in the most convenient way possible (e.g., in some scenarios, it's easier to specify a criteria
                for which item is lesser, and in others, it's easier to specify a criteria for which item is greater).
                If this is provided, a custom `template` must also be provided.  Defaults to
                [Task.CHOOSE_GREATER][semlib.compare.Task.CHOOSE_GREATER] if not specified.
            model: If specified, overrides the default model for this call.

        Returns:
            The ordering of the two items.

        Raises:
            ValidationError: If parsing the LLM response fails.

        Examples:
            Basic comparison:
            >>> await session.compare("twelve", "seventy two")
            <Order.LESS: 'less'>

            Custom criteria:
            >>> await session.compare("California condor", "Bald eagle", by="wingspan")
            <Order.GREATER: 'greater'>

            Custom template and task:
            >>> await session.compare(
            ...     "proton",
            ...     "electron",
            ...     template="Which is smaller, (A) {} or (B) {}?",
            ...     task=Task.CHOOSE_LESSER,
            ... )
            <Order.GREATER: 'greater'>
        """
        if task is not None and task not in {Task.CHOOSE_GREATER, Task.CHOOSE_GREATER_OR_ABSTAIN} and template is None:
            msg = "if 'task' is not CHOOSE_GREATER or CHOOSE_GREATER_OR_ABSTAIN, 'template' must also be provided"
            raise ValueError(msg)
        if template is not None:
            if callable(template) and to_str is not None:
                msg = "cannot provide 'to_str' when a template function is provided"
                raise ValueError(msg)
            if by is not None:
                msg = "cannot provide 'by' when a custom template is provided"
                raise ValueError(msg)

        to_str = to_str if to_str is not None else str
        if task is None:
            task = Task.CHOOSE_GREATER
        elif isinstance(task, str):
            task = Task(task)
        model = model if model is not None else self._model

        if isinstance(template, str):
            prompt = template.format(to_str(a), to_str(b))
        elif template is not None:
            # callable
            prompt = template(a, b)
        elif by is None:
            prompt = _DEFAULT_TEMPLATE.format(a=to_str(a), b=to_str(b))
        else:
            prompt = _DEFAULT_TEMPLATE_BY.format(criteria=by, a=to_str(a), b=to_str(b))

        response = await self.prompt(
            prompt,
            model=model,
            return_type=_RETURN_TYPE_BY_TASK[task],
        )

        match task:
            case Task.COMPARE:
                strict_compare_result = cast(_StrictCompareResult, response)
                return strict_compare_result.order.to_order()
            case Task.COMPARE_OR_ABSTAIN:
                compare_result = cast(_CompareResult, response)
                return compare_result.order
            case Task.CHOOSE_GREATER:
                strict_choose_result = cast(_StrictChooseResult, response)
                match strict_choose_result.choice:
                    case _StrictChoice.A:
                        return Order.GREATER
                    case _StrictChoice.B:
                        return Order.LESS
            case Task.CHOOSE_GREATER_OR_ABSTAIN:
                choose_result = cast(_ChooseResult, response)
                match choose_result.choice:
                    case _Choice.A:
                        return Order.GREATER
                    case _Choice.B:
                        return Order.LESS
                    case _Choice.NEITHER:
                        return Order.NEITHER
            case Task.CHOOSE_LESSER:
                strict_choose_result = cast(_StrictChooseResult, response)
                match strict_choose_result.choice:
                    case _StrictChoice.A:
                        return Order.LESS
                    case _StrictChoice.B:
                        return Order.GREATER
            case Task.CHOOSE_LESSER_OR_ABSTAIN:
                choose_result = cast(_ChooseResult, response)
                match choose_result.choice:
                    case _Choice.A:
                        return Order.LESS
                    case _Choice.B:
                        return Order.GREATER
                    case _Choice.NEITHER:
                        return Order.NEITHER


async def compare[T](
    a: T,
    b: T,
    /,
    *,
    by: str | None = None,
    to_str: Callable[[T], str] | None = None,
    template: str | Callable[[T, T], str] | None = None,
    task: Task | str | None = None,
    model: str | None = None,
) -> Order:
    """Standalone version of [compare][semlib.compare.Compare.compare]."""
    comparator = Compare(model=model)
    return await comparator.compare(a, b, by=by, to_str=to_str, template=template, task=task)


def compare_sync[T](
    a: T,
    b: T,
    /,
    *,
    by: str | None = None,
    to_str: Callable[[T], str] | None = None,
    template: str | Callable[[T, T], str] | None = None,
    task: Task | str | None = None,
    model: str | None = None,
) -> Order:
    """Standalone synchronous version of [compare][semlib.compare.Compare.compare]."""
    comparator = Compare(model=model)
    return asyncio.run(comparator.compare(a, b, by=by, to_str=to_str, template=template, task=task))
