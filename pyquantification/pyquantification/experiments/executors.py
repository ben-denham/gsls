import concurrent.futures
from concurrent.futures import Future, Executor, ProcessPoolExecutor
from typing import cast, Any, Callable, Sequence, List, Iterator


class LazyFuture(Future):
    """A future with a task that will be executed once run() is called."""

    def __init__(self, fn: Callable, /, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        if not self.set_running_or_notify_cancel():
            return
        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException as ex:
            self.set_exception(ex)
        else:
            self.set_result(result)


class SerialExecutor(Executor):
    """An executor that runs all tasks in a single thread - useful when
    profiling resource usage of tasks."""

    def __init__(self) -> None:
        self.futures: List[LazyFuture] = []

    def submit(self, fn: Callable, /, *args: Any, **kwargs: Any) -> LazyFuture:  # type: ignore
        """Save the task as a LazyFuture to be executed later."""
        future = LazyFuture(fn, *args, **kwargs)
        self.futures.append(future)
        return future

    def shutdown(self, wait: bool = True, *,
                 cancel_futures: bool = False) -> None:
        if cancel_futures:
            for future in self.futures:
                future.cancel()


def as_completed(executor: Executor, futures: Sequence[Future]) -> Iterator[Future]:
    """Extend as_completed() with support for SerialExecutor's LazyFutures."""
    if isinstance(executor, SerialExecutor):
        for future in futures:
            cast(LazyFuture, future).run()
            yield future
    else:
        yield from concurrent.futures.as_completed(futures)


def get_executor(max_workers: int) -> Executor:
    """Return an appropriate *Executor object for max_workers."""
    if max_workers == 1:
        # When there is only one worker, use a SerialExecutor so that
        # we can get a full stack trace from a single process (aids
        # debugging and profiling).
        return SerialExecutor()
    return ProcessPoolExecutor(max_workers=max_workers)
