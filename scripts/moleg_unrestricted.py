"""Remove application admission queues before an isolated app starts serving."""
from contextlib import asynccontextmanager


class ActiveCalls:
    """Track execution without a semaphore, tickets, capacity, or waiting."""
    max_concurrency = None
    queued = 0

    def __init__(self):
        self.active = 0

    @asynccontextmanager
    async def limit(self, *, request_id="-"):
        self.active += 1
        try:
            yield
        finally:
            self.active -= 1


def remove_admission_limits(app, generate_queue):
    """Preserve per-session state serialization and unrelated middleware.

    The archived app's GenerateQueue.limit uses _global.limit dynamically.
    Replacing that component also covers modules holding generate_queue aliases.
    """
    if app.middleware_stack is not None:
        raise RuntimeError("remove admission queues before the ASGI stack is built")
    if generate_queue._global.active or generate_queue._global.queued:
        raise RuntimeError("cannot replace admission accounting while requests are running")
    app.user_middleware[:] = [m for m in app.user_middleware
                             if m.cls.__name__ != "GlobalWaitQueueMiddleware"]
    generate_queue._global = ActiveCalls()
