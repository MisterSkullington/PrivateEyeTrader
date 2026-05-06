"""Async pub/sub event bus. All modules communicate exclusively via this bus."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine

from privateye.core.types import EventType


Handler = Callable[[Any], Coroutine[Any, Any, None]]


class AsyncEventBus:
    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._queue: asyncio.Queue[tuple[EventType, Any]] = asyncio.Queue()
        self._running = False

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type] = [
            h for h in self._handlers[event_type] if h is not handler
        ]

    async def publish(self, event_type: EventType, payload: Any) -> None:
        await self._queue.put((event_type, payload))

    def publish_sync(self, event_type: EventType, payload: Any) -> None:
        """Fire-and-forget from sync contexts. Creates a task on the running loop."""
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(self._queue.put_nowait, (event_type, payload))

    async def dispatch(self, event_type: EventType, payload: Any) -> None:
        """Immediately dispatch to all subscribers without queuing."""
        for handler in self._handlers.get(event_type, []):
            await handler(payload)

    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                event_type, payload = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                for handler in self._handlers.get(event_type, []):
                    try:
                        await handler(payload)
                    except Exception as exc:
                        from privateye.utils.logging import get_logger
                        get_logger().error(f"Handler error [{event_type}]: {exc}")
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        self._running = False

    async def wait_empty(self) -> None:
        await self._queue.join()
