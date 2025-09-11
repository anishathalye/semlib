from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Iterable

from semlib.compare import Order


class Algorithm(ABC):
    """Abstract base class for sorting algorithms.

    Sorting algorithms can be used with [sort][semlib.sort.Sort.sort]."""

    def __init__(self) -> None:
        """Initialize."""

    @abstractmethod
    async def _sort[T](
        self,
        iterable: Iterable[T],
        /,
        *,
        reverse: bool = False,
        comparator: Callable[[T, T], Coroutine[None, None, Order]],
        max_concurrency: int,
    ) -> list[T]: ...
