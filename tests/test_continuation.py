import asyncio
import pytest

from cnu_rag_optimization.continuation import ContinuationWindow


@pytest.mark.parametrize('ordering,aged,expected', [
    ('continuation',False,['continuation','fresh']),
    ('continuation',True,['fresh','continuation']),
    ('fifo',False,['fresh','continuation']),
])
def test_progress_priority_and_aged_fifo(ordering,aged,expected):
    async def run():
        now=[0.0]
        window=ContinuationWindow(1,ordering=ordering,aging_s=10,clock=lambda:now[0])
        progress=asyncio.Event();start_next=asyncio.Event();seen=[]
        async def progressed():
            with window.flow():
                async with window.slot():pass
                progress.set()
                await start_next.wait()
                async with window.slot():seen.append('continuation')
        async def fresh():
            with window.flow():
                async with window.slot():seen.append('fresh')
        task=asyncio.create_task(progressed());await progress.wait()
        async with window.slot():
            new=asyncio.create_task(fresh());await asyncio.sleep(0)
            if aged:now[0]=11
            start_next.set();await asyncio.sleep(0)
        await asyncio.gather(task,new)
        assert seen==expected
        assert window.active==0 and not window.pending
    asyncio.run(run())


def test_waiter_cancellation_and_dispatch_race_release_credit():
    async def run():
        window=ContinuationWindow(1)
        async def invoke():
            async with window.slot():await asyncio.sleep(10)
        async with window.slot():
            cancelled=asyncio.create_task(invoke());await asyncio.sleep(0)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):await cancelled
            raced=asyncio.create_task(invoke());await asyncio.sleep(0)
        raced.cancel()  # granted a slot but not resumed yet
        with pytest.raises(asyncio.CancelledError):await raced
        assert window.active==0 and not window.pending
        async with window.slot():pass
    asyncio.run(run())


def test_child_tasks_cannot_borrow_parent_credit():
    async def run():
        window=ContinuationWindow(1);entered=asyncio.Event()
        async def child():
            async with window.slot():entered.set()
        async with window.slot():
            async with window.slot():assert window.active==1
            task=asyncio.create_task(child());await asyncio.sleep(0)
            assert not entered.is_set()
        await task
        assert entered.is_set() and window.active==0
    asyncio.run(run())


def test_local_wait_does_not_consume_inner_execution_timeout():
    async def run():
        window=ContinuationWindow(1)
        async def invoke():
            with window.flow():
                async with window.slot():
                    await asyncio.wait_for(asyncio.sleep(.025),timeout=.05)
        await asyncio.wait_for(asyncio.gather(*(invoke() for _ in range(4))),timeout=.5)
        assert window.active==0
    asyncio.run(run())


def test_failed_call_does_not_count_as_completed_progress():
    async def run():
        events=[];window=ContinuationWindow(1,on_event=events.append)
        with window.flow():
            with pytest.raises(ValueError):
                async with window.slot():raise ValueError('bad output')
            async with window.slot():pass
        assert [e['completed_before'] for e in events if e['event']=='dispatch']==[0,0]
    asyncio.run(run())


def test_zero_clock_wait_time_is_not_confused_with_service_time():
    async def run():
        now=[0.0];events=[]
        window=ContinuationWindow(1,clock=lambda:now[0],on_event=events.append)
        async with window.slot():now[0]=5.0
        released=events[-1]
        assert released['wait_s']==0 and released['service_s']==5
    asyncio.run(run())


@pytest.mark.parametrize('shape', ['sequential', 'fanout', 'dynamic_loop'])
def test_independent_workflows_preserve_results_across_execution_shapes(shape):
    async def run():
        window=ContinuationWindow(2)
        running=set();peak=0
        async def invoke(value):
            nonlocal peak
            async with window.slot():
                running.add(value);peak=max(peak,len(running))
                try:
                    await asyncio.sleep(0)
                    return value
                finally:running.remove(value)
        async def workflow(i):
            with window.flow():
                if shape=='fanout':
                    first=await invoke((i,0))
                    branches=await asyncio.gather(invoke((i,1)),invoke((i,2)))
                    last=await invoke((i,3))
                    return [first,*branches,last]
                out=[]
                count=4 if shape=='sequential' else 2+i%3
                while len(out)<count:
                    out.append(await invoke((i,len(out))))
                return out
        results=await asyncio.gather(*(workflow(i) for i in range(12)))
        assert results==[[(i,j) for j in range(2+i%3 if shape=='dynamic_loop' else 4)] for i in range(12)]
        assert peak==2 and not running
        assert window.snapshot()['active']==0 and window.snapshot()['queued']==0
    asyncio.run(run())
