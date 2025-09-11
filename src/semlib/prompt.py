import asyncio
from typing import overload

from pydantic import BaseModel

from semlib._internal.base import Base
from semlib.bare import Bare


@overload
async def prompt[T: BaseModel](prompt: str, /, *, return_type: type[T], model: str | None = None) -> T: ...


@overload
async def prompt[T](prompt: str, /, *, return_type: Bare[T], model: str | None = None) -> T: ...


@overload
async def prompt(prompt: str, /, *, return_type: None = None, model: str | None = None) -> str: ...


async def prompt[T: BaseModel, U](
    prompt: str, /, *, return_type: type[T] | Bare[U] | None = None, model: str | None = None
) -> str | T | U:
    """Standalone version of [prompt][semlib._internal.base.Base.prompt]."""
    base = Base(model=model)
    return await base.prompt(prompt, return_type=return_type)


@overload
def prompt_sync[T: BaseModel](prompt: str, /, *, return_type: type[T], model: str | None = None) -> T: ...


@overload
def prompt_sync[T](prompt: str, /, *, return_type: Bare[T], model: str | None = None) -> T: ...


@overload
def prompt_sync(prompt: str, /, *, return_type: None = None, model: str | None = None) -> str: ...


def prompt_sync[T: BaseModel, U](
    prompt: str, /, *, return_type: type[T] | Bare[U] | None = None, model: str | None = None
) -> str | T | U:
    """Standalone synchronous version of [prompt][semlib._internal.base.Base.prompt]."""
    base = Base(model=model)
    return asyncio.run(base.prompt(prompt, return_type=return_type))  # type: ignore[return-value]
