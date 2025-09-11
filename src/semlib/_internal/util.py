import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from typing import Any

from semlib._internal.constants import DEFAULT_MAX_CONCURRENCY


class Poison:
    pass


poison = Poison()


async def foreach[T](fn: Callable[[T], Awaitable[None]], iterable: Iterable[T], /, *, max_concurrency: int) -> None:
    q: asyncio.Queue[T | Poison] = asyncio.Queue(maxsize=max_concurrency)

    async def worker() -> None:
        while True:
            item = await q.get()
            if isinstance(item, Poison):
                q.task_done()
                break
            await fn(item)
            q.task_done()

    async def producer() -> None:
        for item in iterable:
            await q.put(item)
        for _ in range(max_concurrency):
            await q.put(poison)

    try:
        async with asyncio.TaskGroup() as tg:
            for _ in range(max_concurrency):
                tg.create_task(worker())
            tg.create_task(producer())
    except ExceptionGroup as eg:
        raise eg.exceptions[0] from eg


async def gather[T](*coros: Coroutine[Any, Any, T]) -> list[T]:
    tasks: list[asyncio.Task[T]] = [asyncio.create_task(coro) for coro in coros]
    try:
        return await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.wait(tasks)
        raise


def parse_max_concurrency(max_concurrency: int | None, model: str | None) -> int:
    if max_concurrency is not None and max_concurrency <= 0:
        msg = "max_concurrency must be a positive integer or None"
        raise ValueError(msg)

    if max_concurrency is not None:
        return max_concurrency
    if model is not None and model.startswith("ollama"):
        return 1
    return DEFAULT_MAX_CONCURRENCY
