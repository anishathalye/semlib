import pytest

from semlib._internal.util import foreach


@pytest.mark.asyncio
async def test_foreach_raise() -> None:
    async def fn(_: int) -> None:
        msg = "foo"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="foo"):
        await foreach(fn, range(10), max_concurrency=2)
