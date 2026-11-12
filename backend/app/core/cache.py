"""
Small in-process caches for expensive read paths.

The dashboard polls every few seconds while the underlying data only changes
when a bid lands or a round settles. ``VersionedTTLCache`` keeps the last
computed value and hands it back as long as a cheap "data version" fingerprint
is unchanged and the entry is younger than ``ttl`` seconds. A version bump
(new bid, new round, new allocation, new agent) invalidates immediately, so
callers never see stale results after a write; the TTL only bounds staleness
of values that change without a new row (e.g. live LLM telemetry).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Hashable
from typing import Any


class VersionedTTLCache:
    def __init__(self, ttl: float = 5.0, max_entries: int = 64):
        self.ttl = ttl
        self.max_entries = max_entries
        self._entries: dict[Hashable, tuple[Hashable, float, Any]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: Hashable, version: Hashable, compute: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                cached_version, stamp, value = entry
                if cached_version == version and now - stamp < self.ttl:
                    self.hits += 1
                    return value
            self.misses += 1

        value = compute()
        with self._lock:
            if len(self._entries) >= self.max_entries and key not in self._entries:
                oldest = min(self._entries, key=lambda k: self._entries[k][1])
                self._entries.pop(oldest, None)
            self._entries[key] = (version, now, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


dashboard_cache = VersionedTTLCache(ttl=5.0)
