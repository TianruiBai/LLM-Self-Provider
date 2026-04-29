"""In-process event bus for broadcasting live API activity to UI subscribers.

The gateway publishes events (request started, deltas, finished) and any
number of subscribers (the web UI's /events SSE stream) receive them.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

log = logging.getLogger("provider.events")


class EventBus:
    def __init__(self, max_queue: int = 512) -> None:
        self._subs: set[asyncio.Queue[dict[str, Any]]] = set()
        self._max = max_queue
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._max)
        async with self._lock:
            self._subs.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self._subs.discard(q)

    def publish(self, event: dict[str, Any]) -> None:
        # Stamp with server time so clients don't depend on local clocks.
        event = {"ts": time.time(), **event}
        # Snapshot subscribers without awaiting the lock — publish is hot path.
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest then enqueue the new event.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass

    async def stream(self, q: asyncio.Queue[dict[str, Any]]) -> AsyncIterator[bytes]:
        """Yield SSE-formatted bytes for one subscriber. Sends a heartbeat
        every 15s so proxies/browsers keep the connection alive."""
        try:
            # Initial hello so the client knows the channel is live.
            yield b": connected\n\n"
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield b": ping\n\n"
                    continue
                payload = json.dumps(ev, ensure_ascii=False, default=str)
                yield f"data: {payload}\n\n".encode("utf-8")
        finally:
            await self.unsubscribe(q)
