import time
from collections import defaultdict
from contextlib import contextmanager
from typing import DefaultDict, Dict, Generator, List


class Tracer:
    def __init__(self, cuda: bool = True) -> None:
        if cuda:
            try:
                import torch

                cuda = torch.cuda.is_available()
            except ImportError:
                cuda = False

        self.start_time: List[float] = []
        self.callstack: List[str] = []
        self.total_time: DefaultDict[str, float] = defaultdict(float)
        self.cuda = cuda

    def start(self, name: str) -> None:
        self.callstack.append(name)
        self.start_time.append(time.time())

    def end(self, name: str) -> None:
        if self.cuda:
            import torch

            torch.cuda.synchronize()
        self.total_time[self.stack] += time.time() - self.start_time.pop()
        actual_name = self.callstack.pop()
        assert (
            actual_name == name
        ), f"Expected to complete {name}, but currently active span is {actual_name}"

    def finish(self) -> Dict[str, float]:
        assert (
            len(self.callstack) == 0
        ), f"Cannot finish when there are open traces: {self.stack}"
        self_times: Dict[str, float] = {}
        # Traverse the tree depth-first
        for name in reversed(sorted(self.total_time.keys())):
            time_in_children = sum(
                t for child, t in self_times.items() if child.startswith(name)
            )
            self_times[f"{name}[self]"] = self.total_time[name] - time_in_children

        self_times.update(self.total_time)
        self.total_time = defaultdict(float)
        return self_times

    @contextmanager
    def span(self, name: str) -> Generator[None, None, None]:
        self.start(name)
        yield
        self.end(name)

    @property
    def stack(self) -> str:
        return ".".join(self.callstack)
