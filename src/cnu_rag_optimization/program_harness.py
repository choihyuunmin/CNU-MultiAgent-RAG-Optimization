"""Request-local dataflow harness for application programs.

Ready successors start when their declared dependencies finish, instead of
waiting for an unrelated barrier. This module owns neither RPC payloads nor
results and applies no admission/concurrency limit.
"""
import asyncio
import contextvars
import time
from dataclasses import dataclass, field
from functools import wraps


@dataclass
class _Node:
    name: str
    scheduled: float
    started: float | None = None
    finished: float | None = None
    joined: float | None = None
    join_finished: float | None = None
    status: str = "waiting"


@dataclass
class _Workflow:
    nodes: dict[str, _Node] = field(default_factory=dict)
    tasks: list[asyncio.Task] = field(default_factory=list)


class ProgramHarness:
    def __init__(self, emit=lambda event: None):
        self.emit = emit
        self.scope = contextvars.ContextVar("program_harness", default=None)

    def _emit(self, event):
        try:
            self.emit(event)
        except Exception:
            pass

    def wrap(self, function):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            workflow = _Workflow()
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
                        "run_ms": None if node.finished is None or node.started is None else
                            (node.finished - node.started) * 1000,
                        "join_wait_ms": None if node.join_finished is None or node.joined is None else
                            (node.join_finished - node.joined) * 1000,
                        "overlap_before_join_ms": overlap,
                    })
                self._emit({"event": "workflow_closed", "nodes": nodes, "cancelled": cancelled})
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
        try:
            return await task
        finally:
            node.join_finished = time.monotonic()
