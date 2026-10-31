import secrets
from collections.abc import Callable, Coroutine, Iterable
from typing import override

from semlib._internal import util
from semlib.compare import Order
from semlib.sort.algorithm.algorithm import Algorithm


class QuickSort(Algorithm):
    """Quicksort sorting algorithm.

    This algorithm uses the [Quicksort](https://en.wikipedia.org/wiki/Quicksort) method to sort items. This algorithm
    does **not** provide theoretical guarantees with noisy pairwise comparisons, but standard sorting algorithms can
    perform well in practice ([Qin et al., 2024](https://aclanthology.org/2024.findings-naacl.97/)) even with noisy
    pariwise comparisons using LLMs.

    This algorithm requires O(n log n) pairwise comparisons on average. If you want higher-quality rankings and can
    tolerate increased costs and latency, you can consider using the [BordaCount][semlib.sort.algorithm.BordaCount]
    algorithm instead.
    """

    @override
    def __init__(self, *, randomized: bool = False):
        """Initialize.

        Args:
            randomized: If `True`, uses a randomized pivot selection strategy. This can help avoid worst-case O(n^2)
                performance on certain inputs, but results may be non-deterministic. If False, always uses the first
                item as the pivot.
        """
        super().__init__()
        self._randomized = randomized

    @override
    async def _sort[T](
        self,
        iterable: Iterable[T],
        /,
        *,
        reverse: bool = False,
        comparator: Callable[[T, T], Coroutine[None, None, Order]],
        max_concurrency: int,
    ) -> list[T]:
        lst = iterable if isinstance(iterable, list) else list(iterable)

        async def quicksort(lst: list[T]) -> list[T]:
            if len(lst) <= 1:
                return lst
            pivot_index = secrets.randbelow(len(lst)) if self._randomized else 0
            pivot = lst[pivot_index]
            less = []
            greater = []
            equal = [pivot]  # to handle "neither" case
            comparisons = await util.gather(
                *(comparator(item, pivot) for i, item in enumerate(lst) if i != pivot_index)
            )
            for i, item in enumerate(lst):
                if i == pivot_index:
                    continue
                comparison = comparisons[i if i < pivot_index else i - 1]
                if comparison == Order.LESS:
                    less.append(item)
                elif comparison == Order.GREATER:
                    greater.append(item)
                else:
                    equal.append(item)
            sort_less, sort_greater = await util.gather(quicksort(less), quicksort(greater))
            return sort_less + equal + sort_greater

        sort_list = await quicksort(lst)
        return sort_list[::-1] if reverse else sort_list
