import pathlib
import sqlite3

import pytest
from litellm.types.utils import Message
from pydantic import BaseModel

from semlib.cache import _VERSION, _VERSION_KEY, InMemoryCache, OnDiskCache, QueryCache

# ruff: noqa: SLF001


def test_in_memory_cache() -> None:
    cache: QueryCache = InMemoryCache()
    key: tuple[list[Message], None, str] = ([], None, "gpt-4")
    assert cache._get(key) is None
    cache._set(key, "value1")
    assert cache._get(key) == "value1"
    cache._set(key, "value2")
    assert cache._get(key) == "value2"
    cache.clear()
    assert cache._get(key) is None


def test_on_disk_cache(tmp_path: pathlib.Path) -> None:
    cache_path = tmp_path / "cache.db"
    cache: QueryCache = OnDiskCache(str(cache_path))
    key: tuple[list[Message], None, str] = ([], None, "gpt-4")
    assert cache._get(key) is None
    cache._set(key, "value1")
    assert cache._get(key) == "value1"
    cache._set(key, "value2")
    assert cache._get(key) == "value2"

    # re-open and check persistence
    cache = OnDiskCache(str(cache_path))
    assert cache._get(key) == "value2"

    # check version mismatch
    conn = sqlite3.connect(str(cache_path), autocommit=True)
    conn.execute(f"UPDATE metadata SET value = '{int(_VERSION)+1}' WHERE key = '{_VERSION_KEY}'")
    conn.close()
    with pytest.raises(ValueError, match="cache version mismatch"):
        OnDiskCache(str(cache_path))


@pytest.mark.parametrize("cache_type", ["in_memory", "on_disk"])
def test_cache_interface(cache_type: str, tmp_path: pathlib.Path) -> None:
    if cache_type == "on_disk":
        cache_path = tmp_path / "cache.db"
        cache: QueryCache = OnDiskCache(str(cache_path))
    else:
        cache = InMemoryCache()

    # test that different messages/return types/models are cached separately
    class Foo(BaseModel):
        bar: int

    key1: tuple[list[Message], None, str] = ([Message(role="user", content="Hello")], None, "gpt-4.1")
    key2: tuple[list[Message], None, str] = ([Message(role="user", content="Hello")], None, "gpt-4o")  # different model
    key3: tuple[list[Message], None, str] = (
        [Message(role="user", content="Hello"), Message(role="user", content="world")],
        None,
        "gpt-4",
    )  # different content
    key4: tuple[list[Message], type[Foo], str] = (
        [Message(role="user", content="Hello")],
        Foo,
        "gpt-4o",
    )  # different return type

    assert len(cache) == 0

    cache._set(key1, "value1")
    assert cache._get(key1) == "value1"
    assert cache._get(key2) is None
    assert cache._get(key3) is None
    assert cache._get(key4) is None

    cache._set(key2, "value2")
    assert cache._get(key1) == "value1"
    assert cache._get(key2) == "value2"
    assert cache._get(key3) is None
    assert cache._get(key4) is None

    cache._set(key3, "value3")
    assert cache._get(key1) == "value1"
    assert cache._get(key2) == "value2"
    assert cache._get(key3) == "value3"
    assert cache._get(key4) is None

    cache._set(key4, "value4")
    assert cache._get(key1) == "value1"
    assert cache._get(key2) == "value2"
    assert cache._get(key3) == "value3"
    assert cache._get(key4) == "value4"

    assert len(cache) == 4
    cache.clear()
    assert len(cache) == 0
