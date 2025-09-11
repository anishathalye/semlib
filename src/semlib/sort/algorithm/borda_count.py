from collections.abc import Callable, Coroutine, Iterable
from typing import override

from semlib._internal import util
from semlib._internal.util import foreach
from semlib.compare import Order
from semlib.sort.algorithm.algorithm import Algorithm


class BordaCount(Algorithm):
    """Borda count sorting algorithm.

    This algorithm uses the [Borda count](https://en.wikipedia.org/wiki/Borda_count) method to rank items.  The
    algorithm has good theoretical properties for finding approximate rankings based on noisy pairwise comparisons
    ([Shah and Wainwright, 2018](https://jmlr.org/papers/volume18/16-206/16-206.pdf)).

    This algorithm requires O(n^2) pairwise comparisons. If you want to reduce the number of comparisons (to reduce LLM
    costs), you can consider using the [QuickSort][semlib.sort.algorithm.QuickSort] algorithm instead.

    This algorithm is carefully implemented so that it has O(n) space complexity.
    """

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
        scores: list[int] = [0 for _ in lst]

        async def fn(item: tuple[int, int]) -> None:
            i, j = item
            result_ij, result_ji = await util.gather(comparator(lst[i], lst[j]), comparator(lst[j], lst[i]))
            if result_ij == Order.LESS and result_ji == Order.GREATER:
                scores[j] += 1
                scores[i] -= 1
            elif result_ij == Order.GREATER and result_ji == Order.LESS:
                scores[i] += 1
                scores[j] -= 1

        await foreach(
            fn,
            ((i, j) for i in range(len(lst)) for j in range(i + 1, len(lst))),
            max_concurrency=max(1, max_concurrency // 2),  # because each worker does two comparisons concurrently
        )

        # stable sort
        sort_by_score = sorted([(scores[i], i, lst[i]) for i in range(len(lst))], reverse=reverse)
        return [item for _, _, item in sort_by_score]
