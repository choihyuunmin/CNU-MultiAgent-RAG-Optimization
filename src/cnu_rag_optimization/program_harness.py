import asyncio
import contextvars
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps


@dataclass
class _Node:
    name: str
    scheduled: float
    dependency_ready: float | None = None
    started: float | None = None
    finished: float | None = None
    joined: float | None = None
    join_finished: float | None = None
    status: str = "waiting"
    release: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _Workflow:
    started: float = field(default_factory=time.monotonic)
    workflow_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    request_id: str | None = None
    nodes: dict[str, _Node] = field(default_factory=dict)
    tasks: list[asyncio.Task] = field(default_factory=list)


class ProgramHarness:
    def __init__(self, emit=lambda event: None, *, start_when_ready=True):
        """Schedule declared work, without inferring dependencies or branch guards.

        ``False`` is an instrumented control: execute the same operation only at
        its original join. Callers must only declare work whose execution is
        already required; conditional/unused branches must not be speculated.
        This is not an admission limit. Synchronous work sent to a thread cannot
        be stopped by cancelling its asyncio waiter.
        """
        self.emit = emit
        self.start_when_ready = start_when_ready
        self.scope = contextvars.ContextVar("program_harness", default=None)

    def _emit(self, event):
        try:
            self.emit(event)
        except Exception:
            pass

    def wrap(self, function, *, request_key=None):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            workflow = _Workflow()
            if request_key is not None:
                try:
                    workflow.request_id = request_key(*args, **kwargs)
                except Exception:
                    pass  # Observability must not change application behavior.
            token = self.scope.set(workflow)
            try:
                return await function(*args, **kwargs)
            finally:
                cancelled = 0
                for task in workflow.tasks:
                    if not task.done():
                        cancelled += 1
                        task.cancel()
                await asyncio.gather(*workflow.tasks, return_exceptions=True)
                nodes = []
                for node in workflow.nodes.values():
                    overlap = None
                    if node.started is not None and node.joined is not None:
                        overlap = max(0.0, min(node.joined, node.finished or node.joined) - node.started) * 1000
                    nodes.append({
                        "name": node.name, "status": node.status,
                        "scheduled_ms": (node.scheduled - workflow.started) * 1000,
                        "dependency_ready_ms": None if node.dependency_ready is None else
                            (node.dependency_ready - workflow.started) * 1000,
                        "started_ms": None if node.started is None else
                            (node.started - workflow.started) * 1000,
                        "finished_ms": None if node.finished is None else
                            (node.finished - workflow.started) * 1000,
                        "joined_ms": None if node.joined is None else
                            (node.joined - workflow.started) * 1000,
                        "ready_wait_ms": None if node.started is None or node.dependency_ready is None else
                            (node.started - node.dependency_ready) * 1000,
                        "run_ms": None if node.finished is None or node.started is None else
                            (node.finished - node.started) * 1000,
                        "join_wait_ms": None if node.join_finished is None or node.joined is None else
                            (node.join_finished - node.joined) * 1000,
                        "overlap_before_join_ms": overlap,
                    })
                self._emit({"event": "workflow_closed", "nodes": nodes, "cancelled": cancelled,
                            "workflow_id": workflow.workflow_id, "request_id": workflow.request_id,
                            "schedule": "ready" if self.start_when_ready else "barrier",
                            "elapsed_ms": (time.monotonic() - workflow.started) * 1000})
                self.scope.reset(token)
        return wrapped

    def after(self, name, predecessor, operation):
        workflow = self.scope.get()
        if workflow is None:
            raise RuntimeError("program harness requires request scope")
        if not name or name in workflow.nodes:
            raise ValueError("unique nonempty node name required")
        node = _Node(name=name, scheduled=time.monotonic())
        workflow.nodes[name] = node

        async def run():
            try:
                value = await predecessor
                node.dependency_ready = time.monotonic()
                node.status = "ready"
                if not self.start_when_ready:
                    await node.release.wait()
                node.started = time.monotonic()
                node.status = "running"
                result = await operation(value)
                node.status = "ok"
                return result
            except asyncio.CancelledError:
                node.status = "cancelled"
                raise
            except BaseException:
                node.status = "error"
                raise
            finally:
                node.finished = time.monotonic()
        task = asyncio.create_task(run(), name="cnu-program-" + name)
        workflow.tasks.append(task)
        return task

    async def join(self, task):
        workflow = self.scope.get()
        if workflow is None or task not in workflow.tasks:
            raise ValueError("task is not owned by current workflow")
        node = next(node for node in workflow.nodes.values()
                    if task.get_name() == "cnu-program-" + node.name)
        node.joined = time.monotonic()
        node.release.set()
        try:
            return await task
        finally:
            node.join_finished = time.monotonic()
