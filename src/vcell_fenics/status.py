"""Report run status the way VCell listens for it ([ADR 011] §4).

VCell learns about a running solver in one of two ways:

- **Locally** it scans the solver's *stdout* for ``[[[message]]]`` markers (``MathExecutable
  .checkForNewApplicationMessages``) and parses ``data:<t>`` (the value after the last ``:``) and
  ``progress:<p>%`` (between the last ``:`` and the ``%``). The scanner is unforgiving: a stray ``]]]``
  before the next ``[[[`` stops it for good, and any other message text throws. So stdout must carry
  markers **only** — :func:`isolate_stdout` points file descriptor 1 at stderr (Netgen and PETSc print
  there from C) and hands back a private stream for the markers.
- **On the cluster** the solver posts REST *WorkerEvents* to the VCell message broker — the pattern of
  the Langevin solver (``--vc-send-status-config=FILE``): STARTING 999, DATA 1000, PROGRESS 1001,
  FAILURE 1002, COMPLETED 1003. A job is complete only on 1003 (``SimulationStateMachine``; VCell's
  postprocessor sends only worker-exit), so the solver sends it — after its results are finalized.

Messaging must never break a solve: every send failure is swallowed (and logged once).

[ADR 011]: ../../docs/decisions/011-vcell-solver-contract.md
"""

from __future__ import annotations

import base64
import io
import os
import socket
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

# WorkerEvent codes (vcell-core cbit/rmi/event/WorkerEvent.java).
JOB_STARTING = 999
JOB_DATA = 1000
JOB_PROGRESS = 1001
JOB_FAILURE = 1002
JOB_COMPLETED = 1003

_PERSISTENT = {JOB_STARTING, JOB_FAILURE, JOB_COMPLETED}  # delivered even if the broker restarts
_MESSAGE_LIMIT = 2048


class StatusReporter(Protocol):
    def starting(self) -> None: ...
    def progress(self, fraction: float, t: float) -> None: ...
    def data(self, t: float, fraction: float) -> None: ...
    def completed(self, t: float) -> None: ...
    def failed(self, message: str, t: float, fraction: float) -> None: ...


class NullReporter:
    """Reports nothing (the default, and every rank but 0)."""

    def starting(self) -> None:
        pass

    def progress(self, fraction: float, t: float) -> None:
        pass

    def data(self, t: float, fraction: float) -> None:
        pass

    def completed(self, t: float) -> None:
        pass

    def failed(self, message: str, t: float, fraction: float) -> None:
        pass


class StdoutMarkers:
    """``[[[progress:NN.N%]]]`` / ``[[[data:<t>]]]`` markers for VCell's local stdout scanner.

    Progress is throttled to one marker per ``interval`` seconds (and to changes of ≥ 0.1%); data
    markers are never throttled — each one tells VCell a new output row is readable."""

    def __init__(self, stream: TextIO, *, interval: float = 1.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._stream = stream
        self._interval = interval
        self._clock = clock
        self._last_time = -float("inf")
        self._last_percent = -1.0

    def _emit(self, message: str) -> None:
        self._stream.write(f"[[[{message}]]]\n")
        self._stream.flush()

    def starting(self) -> None:
        self._progress_marker(0.0, force=True)

    def progress(self, fraction: float, t: float) -> None:
        self._progress_marker(fraction)

    def data(self, t: float, fraction: float) -> None:
        self._emit(f"data:{t:.17g}")
        self._progress_marker(fraction)

    def completed(self, t: float) -> None:
        self._progress_marker(1.0, force=True)

    def failed(self, message: str, t: float, fraction: float) -> None:
        pass  # the failure text goes to stderr; VCell reads the exit status

    def _progress_marker(self, fraction: float, *, force: bool = False) -> None:
        percent = round(100.0 * min(max(fraction, 0.0), 1.0), 1)
        now = self._clock()
        if not force and (now - self._last_time < self._interval or abs(percent - self._last_percent) < 0.1):
            return
        self._last_time, self._last_percent = now, percent
        self._emit(f"progress:{percent:.1f}%")


@dataclass(frozen=True)
class MessagingConfig:
    """The broker coordinates VCell writes for a cluster job — the Langevin ``.langevinMessagingConfig``
    properties format (``LangevinSolver.writeLangevinMessagingConfig``)."""

    broker_host: str
    broker_port: int
    broker_username: str
    broker_password: str
    vc_username: str
    sim_key: str
    task_id: int
    job_index: int

    @classmethod
    def from_properties(cls, path: Path) -> MessagingConfig:
        values: dict[str, str] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
        try:
            return cls(
                broker_host=values["broker_host"],
                broker_port=int(values["broker_port"]),
                broker_username=values["broker_username"],
                broker_password=values["broker_password"],
                vc_username=values["vc_username"],
                sim_key=values["simKey"],
                task_id=int(values["taskID"]),
                job_index=int(values["jobIndex"]),
            )
        except (KeyError, ValueError) as error:
            raise ValueError(f"{path}: incomplete messaging config ({error})") from None


class RestWorkerEvents:
    """POST VCell WorkerEvents to the broker's REST endpoint (``/api/message/workerEvent``), exactly as
    ``docker/build/batch/entrypoint.sh`` and Langevin's ``VCellMessagingRest`` do. Progress is throttled
    to one event per ``interval`` seconds; data events to one per second; failures to send are logged
    once to stderr and otherwise ignored."""

    def __init__(
        self,
        config: MessagingConfig,
        *,
        interval: float = 5.0,
        timeout: float = 10.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
        clock: Callable[[], float] = time.monotonic,
        hostname: str | None = None,
    ) -> None:
        self._config = config
        self._interval = interval
        self._timeout = timeout
        self._opener = opener
        self._clock = clock
        self._hostname = hostname or socket.gethostname()
        self._last_progress = -float("inf")
        self._last_data = -float("inf")
        self._warned = False
        token = base64.b64encode(f"{config.broker_username}:{config.broker_password}".encode()).decode()
        self._authorization = f"Basic {token}"

    def starting(self) -> None:
        self._send(JOB_STARTING, progress=0.0, t=0.0, message="Starting Job")

    def progress(self, fraction: float, t: float) -> None:
        now = self._clock()
        if now - self._last_progress >= self._interval:
            self._last_progress = now
            self._send(JOB_PROGRESS, progress=fraction, t=t)

    def data(self, t: float, fraction: float) -> None:
        now = self._clock()
        if now - self._last_data >= 1.0:
            self._last_data = now
            self._send(JOB_DATA, progress=fraction, t=t)

    def completed(self, t: float) -> None:
        self._send(JOB_COMPLETED, progress=1.0, t=t)

    def failed(self, message: str, t: float, fraction: float) -> None:
        self._send(JOB_FAILURE, progress=fraction, t=t, message=message)

    def url(self, status: int, *, progress: float, t: float, message: str | None = None) -> str:
        """The request URL for one event (public for tests and for debugging a broker)."""

        persistent = status in _PERSISTENT
        params: list[tuple[str, str]] = [
            ("type", "queue"),
            ("JMSPriority", "5"),
            ("JMSTimeToLive", "600000" if persistent else "60000"),
            ("JMSDeliveryMode", "persistent" if persistent else "nonpersistent"),
            ("MessageType", "WorkerEvent"),
            ("UserName", self._config.vc_username),
            ("HostName", self._hostname),
            ("SimKey", self._config.sim_key),
            ("TaskID", str(self._config.task_id)),
            ("JobIndex", str(self._config.job_index)),
            ("WorkerEvent_Status", str(status)),
        ]
        if message is not None:
            params.append(("WorkerEvent_StatusMsg", _sanitize(message)))
        params += [("WorkerEvent_Progress", repr(float(progress))), ("WorkerEvent_TimePoint", repr(float(t)))]
        base = f"http://{self._config.broker_host}:{self._config.broker_port}/api/message/workerEvent"
        return f"{base}?{urllib.parse.urlencode(params)}"

    def _send(self, status: int, *, progress: float, t: float, message: str | None = None) -> None:
        request = urllib.request.Request(
            self.url(status, progress=progress, t=t, message=message),
            data=b"",
            method="POST",
            headers={"Authorization": self._authorization},
        )
        try:
            with self._opener(request, timeout=self._timeout):
                pass
        except Exception as error:  # never let messaging break a solve
            if not self._warned:
                self._warned = True
                print(f"[vcell-fenics] warning: status message to the VCell broker failed: {error}", file=sys.stderr)


class Fanout:
    """Send every event to several reporters (e.g. stdout markers *and* the broker), remembering the
    last time and progress reported — what a failure is reported at."""

    def __init__(self, reporters: Sequence[StatusReporter]) -> None:
        self._reporters = list(reporters)
        self.last_t = 0.0
        self.last_fraction = 0.0

    def starting(self) -> None:
        for reporter in self._reporters:
            reporter.starting()

    def progress(self, fraction: float, t: float) -> None:
        self.last_t, self.last_fraction = t, fraction
        for reporter in self._reporters:
            reporter.progress(fraction, t)

    def data(self, t: float, fraction: float) -> None:
        self.last_t, self.last_fraction = t, fraction
        for reporter in self._reporters:
            reporter.data(t, fraction)

    def completed(self, t: float) -> None:
        for reporter in self._reporters:
            reporter.completed(t)

    def failed(self, message: str, t: float, fraction: float) -> None:
        for reporter in self._reporters:
            reporter.failed(message, t, fraction)


def isolate_stdout() -> TextIO:
    """Reserve stdout for status markers: return a stream on a duplicate of file descriptor 1, then point
    descriptor 1 — and ``sys.stdout`` — at stderr, so C libraries (Netgen, PETSc) and stray prints
    cannot corrupt the marker stream VCell parses."""

    sys.stdout.flush()
    markers_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return io.TextIOWrapper(os.fdopen(markers_fd, "wb", buffering=0), encoding="utf-8", line_buffering=True)


def _sanitize(message: str) -> str:
    """The Langevin client's rule: at most 2048 characters, no newlines or quotes."""

    cleaned = message.translate({ord(c): " " for c in "\n\r'\""})
    return cleaned[:_MESSAGE_LIMIT]


__all__ = [
    "JOB_COMPLETED",
    "JOB_DATA",
    "JOB_FAILURE",
    "JOB_PROGRESS",
    "JOB_STARTING",
    "Fanout",
    "MessagingConfig",
    "NullReporter",
    "RestWorkerEvents",
    "StatusReporter",
    "StdoutMarkers",
    "isolate_stdout",
]
