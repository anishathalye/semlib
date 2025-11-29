import json
import sqlite3
from abc import ABC, abstractmethod
from hashlib import sha256
from typing import cast, override

from litellm.types.utils import Message
from pydantic import BaseModel

type CacheKey[T: BaseModel] = tuple[list[Message], type[T] | None, str]


class QueryCache(ABC):
    """Abstract base class for a cache of LLM query results.

    Caches can be used with [Session][semlib.session.Session] to avoid repeating identical queries to the LLM.
    """

    @abstractmethod
    def _set[T: BaseModel](self, key: CacheKey[T], value: str) -> None: ...

    @abstractmethod
    def _get[T: BaseModel](self, key: CacheKey[T]) -> str | None: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def _hash_key[T: BaseModel](self, key: CacheKey[T]) -> bytes:
        messages, pydantic_model, llm_model = key
        key_components: list[str] = [llm_model]
        key_components.extend(message.to_json() for message in messages)
        if pydantic_model is not None:
            key_components.append(json.dumps(pydantic_model.model_json_schema()))
        h = sha256()
        for part in key_components:
            h.update(part.encode("utf-8"))
        return h.digest()


class InMemoryCache(QueryCache):
    """An in-memory cache of LLM query results."""

    @override
    def __init__(self) -> None:
        """Initialize an in-memory cache."""
        self._data: dict[bytes, str] = {}

    @override
    def _set[T: BaseModel](self, key: CacheKey[T], value: str) -> None:
        self._data[self._hash_key(key)] = value

    @override
    def _get[T: BaseModel](self, key: CacheKey[T]) -> str | None:
        return self._data.get(self._hash_key(key))

    @override
    def clear(self) -> None:
        self._data.clear()

    @override
    def __len__(self) -> int:
        return len(self._data)


_VERSION_KEY = "version"
_VERSION = "1"


class OnDiskCache(QueryCache):
    """A persistent on-disk cache of LLM query results, backed by SQLite."""

    @override
    def __init__(self, path: str) -> None:
        """Initialize an on-disk cache.

        Args:
            path: Path to the SQLite database file. If the file does not exist, it will be created. By convention, the
                filename should have a ".db" or ".sqlite" extension.
        """
        self._conn = sqlite3.connect(path, autocommit=True)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        cur = self._conn.execute("SELECT value FROM metadata WHERE key = ?", (_VERSION_KEY,))
        row = cur.fetchone()
        if row is None:
            self._conn.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", (_VERSION_KEY, _VERSION))
        elif row[0] != _VERSION:
            msg = f"cache version mismatch: expected {_VERSION}, got {row[0]}"
            raise ValueError(msg)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data (
                key BLOB PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

    @override
    def _set[T: BaseModel](self, key: CacheKey[T], value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO data (key, value) VALUES (?, ?)",
            (self._hash_key(key), value),
        )

    @override
    def _get[T: BaseModel](self, key: CacheKey[T]) -> str | None:
        cur = self._conn.execute("SELECT value FROM data WHERE key = ?", (self._hash_key(key),))
        row = cur.fetchone()
        if row is None:
            return None
        return cast("str", row[0])

    @override
    def clear(self) -> None:
        self._conn.execute("DELETE FROM data")

    @override
    def __len__(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM data")
        row = cur.fetchone()
        return cast("int", row[0])
