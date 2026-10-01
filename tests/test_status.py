"""The status protocol (ADR 011 §4): stdout markers VCell's local scanner accepts, and REST WorkerEvents
exactly as VCell's broker expects them.

The stdout side is checked against a Python port of VCell's own parser
(`MathExecutable.checkForNewApplicationMessages` + `LangevinSolver.getApplicationMessage`); the REST side
against the URLs Langevin's `VCellMessagingRestTest` asserts, and a captive HTTP server standing in for
the broker.
"""

from __future__ import annotations

import contextlib
import http.server
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pytest

from vcell_fenics.runner import _PhasedProgress
from vcell_fenics.status import (
    COMPILING,
    JOB_COMPLETED,
    JOB_DATA,
    JOB_FAILURE,
    JOB_PROGRESS,
    JOB_STARTING,
    LOADING,
    MESHING,
    PHASES,
    SOLVING,
    WRITING,
    Fanout,
    MessagingConfig,
    RestWorkerEvents,
    StdoutMarkers,
)

_CONFIG = MessagingConfig(
    broker_host="broker.example",
    broker_port=8165,
    broker_username="msg_user",
    broker_password="msg_pswd",
    vc_username="vcell_user",
    sim_key="12334483837",
    task_id=0,
    job_index=0,
)


# -- a port of VCell's stdout parser ---------------------------------------------------------------------


def vcell_scan(stdout: str) -> list[str]:
    """`MathExecutable.checkForNewApplicationMessages`: the `[[[…]]]` messages VCell would see."""

    messages: list[str] = []
    position = 0
    while True:
        begin, end = stdout.find("[[[", position), stdout.find("]]]", position)
        if begin >= 0 and end > begin:
            messages.append(stdout[begin + 3 : end])
            position = end + 3
        else:
            return messages


def vcell_parse(message: str) -> tuple[str, float]:
    """`LangevinSolver.getApplicationMessage`: data → the time after the last ':', progress → the
    fraction between the last ':' and '%'; anything else throws in VCell."""

    if message.startswith("data:"):
        return "data", float(message[message.rindex(":") + 1 :])
    if message.startswith("progress:"):
        return "progress", float(message[message.rindex(":") + 1 : message.index("%")]) / 100.0
    raise RuntimeError("unrecognized message")


def vcell_phase(message: str) -> str | None:
    """`FenicsSolver.progressPhase` (VCell with phase support): the text between `progress:` and the last
    ':' of a progress marker, or None for a bare `progress:NN.N%`."""

    if not message.startswith("progress:"):
        return None
    body = message[len("progress:") : message.rindex(":")] if message.count(":") > 1 else ""
    return body or None


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_stdout_markers_parse_as_vcell_parses_them() -> None:
    from io import StringIO

    stream, clock = StringIO(), _Clock()
    markers = StdoutMarkers(stream, clock=clock)
    markers.starting()
    clock.now = 2.0
    markers.progress(0.25, 0.0025)
    markers.data(0.005, 0.5)
    clock.now = 4.0
    markers.data(0.01, 1.0)
    markers.completed(0.01)
    parsed = [vcell_parse(m) for m in vcell_scan(stream.getvalue())]
    assert parsed == [
        ("progress", 0.0),  # starting: forced
        ("progress", 0.25),
        ("data", 0.005),  # its progress (50%) lands within 1 s of the last marker: throttled
        ("data", 0.01),
        ("progress", 1.0),
        ("progress", 1.0),  # completed: forced, always 100%
    ]


def test_progress_is_throttled_but_data_is_not() -> None:
    from io import StringIO

    stream, clock = StringIO(), _Clock()
    markers = StdoutMarkers(stream, interval=1.0, clock=clock)
    for k in range(100):  # 100 progress updates within 0.1 s: at most one marker
        clock.now = k * 0.001
        markers.progress(k / 100, k * 0.01)
    progress_markers = [m for m in vcell_scan(stream.getvalue()) if m.startswith("progress")]
    assert len(progress_markers) == 1
    for k in range(5):
        markers.data(float(k), 0.5)
    assert sum(m.startswith("data") for m in vcell_scan(stream.getvalue())) == 5


def test_phase_markers_parse_in_an_older_vcell_and_name_the_phase_in_a_newer_one() -> None:
    """`[[[progress:<phase>:NN.N%]]]` is still a progress marker to a VCell that predates phases (it takes
    the number between the last ':' and the '%'); a newer one also reads the phase."""

    from io import StringIO

    stream, clock = StringIO(), _Clock()
    markers = StdoutMarkers(stream, clock=clock)
    markers.starting()
    markers.phase(LOADING, 0.0, 0.0)
    markers.phase(MESHING, 0.0, 0.0)
    markers.phase(COMPILING, 0.0, 0.0)
    markers.phase(SOLVING, 0.0, 0.0)
    clock.now = 2.0
    markers.progress(0.37, 0.37)
    markers.data(0.5, 0.5)  # its progress marker is throttled (within 1 s)
    markers.phase(WRITING, 1.0, 1.0)
    markers.completed(1.0)
    messages = vcell_scan(stream.getvalue())
    assert [vcell_parse(m) for m in messages] == [
        ("progress", 0.0),
        ("progress", 0.0),
        ("progress", 0.0),
        ("progress", 0.0),
        ("progress", 0.0),
        ("progress", 0.37),
        ("data", 0.5),
        ("progress", 1.0),
        ("progress", 1.0),
    ]
    assert [vcell_phase(m) for m in messages] == [
        None,
        "loading model",
        "meshing",
        "compiling",
        "solving",
        "solving",
        None,
        "writing results",
        None,  # completion is the plain 100% marker
    ]


def test_a_phase_change_is_never_throttled() -> None:
    from io import StringIO

    stream = StringIO()
    markers = StdoutMarkers(stream, clock=_Clock())  # the clock never moves
    for name in PHASES:
        markers.phase(name, 0.0, 0.0)
    assert [vcell_phase(m) for m in vcell_scan(stream.getvalue())] == list(PHASES)


def test_a_phase_cannot_break_the_marker() -> None:
    from io import StringIO

    stream = StringIO()
    StdoutMarkers(stream).phase("odd: 50% [x]]]\nname", 0.5, 0.0)
    (message,) = vcell_scan(stream.getvalue())
    assert vcell_parse(message) == ("progress", 0.5)
    assert vcell_phase(message) == "odd 50 x name"


def test_phased_progress_reports_phases_in_order() -> None:
    """The runner's view: setup phases at 0%, the first step enters solving, progress is the fraction of
    simulated time, writing results at 100%."""

    events: list[tuple[str, object, float, float]] = []

    class Recorder:
        def starting(self) -> None:
            events.append(("starting", None, 0.0, 0.0))

        def phase(self, name: str, fraction: float, t: float) -> None:
            events.append(("phase", name, fraction, t))

        def progress(self, fraction: float, t: float) -> None:
            events.append(("progress", None, fraction, t))

        def data(self, t: float, fraction: float) -> None:
            events.append(("data", None, fraction, t))

        def completed(self, t: float) -> None:
            events.append(("completed", None, 1.0, t))

        def failed(self, message: str, t: float, fraction: float) -> None:
            events.append(("failed", message, fraction, t))

    status = _PhasedProgress(Recorder(), t_final=2.0)
    status.phase(MESHING)
    status.phase(COMPILING)
    status.row(0.0)  # the initial state is written before any step
    status.step(0.5)
    status.step(1.0)
    status.row(1.0)
    status.step(2.0)
    status.phase(WRITING)
    assert events == [
        ("phase", "meshing", 0.0, 0.0),
        ("phase", "compiling", 0.0, 0.0),
        ("data", None, 0.0, 0.0),
        ("phase", "solving", 0.25, 0.5),
        ("progress", None, 0.5, 1.0),
        ("data", None, 0.5, 1.0),
        ("progress", None, 1.0, 2.0),
        ("phase", "writing results", 1.0, 2.0),
    ]
    rows_only: list[tuple[str, object, float, float]] = []
    events = rows_only
    status = _PhasedProgress(Recorder(), t_final=2.0)
    status.row(0.0)
    status.row(1.0)  # an integrator that reports only rows still enters solving
    assert rows_only == [("data", None, 0.0, 0.0), ("phase", "solving", 0.5, 1.0), ("data", None, 0.5, 1.0)]


def test_a_stray_close_token_silences_the_vcell_scanner() -> None:
    """Why stdout must carry markers only: one `]]]` from a library before the next marker, and VCell
    never reads another message."""

    assert vcell_scan("[[[progress:10.0%]]] mesh]]] [[[data:1]]]") == ["progress:10.0%"]


def test_isolate_stdout_sends_c_level_output_to_stderr() -> None:
    code = (
        "import os\n"
        "from vcell_fenics.status import isolate_stdout\n"
        "markers = isolate_stdout()\n"
        "os.write(1, b'netgen says ]]] hello\\n')\n"
        "print('a stray print')\n"
        "markers.write('[[[progress:50.0%]]]\\n')\n"
        "markers.flush()\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "[[[progress:50.0%]]]\n"
    assert "netgen says" in result.stderr and "a stray print" in result.stderr


# -- REST WorkerEvents -------------------------------------------------------------------------------------


def _query(url: str) -> list[tuple[str, str]]:
    return urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query, keep_blank_values=True)


def test_starting_url_matches_the_langevin_client() -> None:
    events = RestWorkerEvents(_CONFIG, hostname="node07")
    url = events.url(JOB_STARTING, progress=0.0, t=0.0, message="Starting Job")
    expected = (
        "http://broker.example:8165/api/message/workerEvent?type=queue&JMSPriority=5&JMSTimeToLive=600000"
        "&JMSDeliveryMode=persistent&MessageType=WorkerEvent&UserName=vcell_user&HostName=node07"
        "&SimKey=12334483837&TaskID=0&JobIndex=0&WorkerEvent_Status=999&WorkerEvent_StatusMsg=Starting+Job"
        "&WorkerEvent_Progress=0.0&WorkerEvent_TimePoint=0.0"
    )
    assert url == expected  # VCellMessagingRestTest.testSendWorkerEvent_starting_good_creds


def test_progress_url_matches_the_langevin_client() -> None:
    events = RestWorkerEvents(_CONFIG, hostname="node07")
    query = dict(_query(events.url(JOB_PROGRESS, progress=0.4, t=2.0)))
    assert query["JMSTimeToLive"] == "60000" and query["JMSDeliveryMode"] == "nonpersistent"
    assert (query["WorkerEvent_Status"], query["WorkerEvent_Progress"], query["WorkerEvent_TimePoint"]) == (
        "1001",
        "0.4",
        "2.0",
    )
    assert "WorkerEvent_StatusMsg" not in query


def test_progress_events_carry_the_phase_as_a_serialized_simulation_message() -> None:
    """`WORKEREVENT_PROGRESS|<phase>` is what VCell's `WorkerEventMessage` turns into the job's status
    message (`SimulationMessage.fromSerializedMessage`), in every VCell that has the broker."""

    sent: list[str] = []

    def opener(request: urllib.request.Request, timeout: float) -> contextlib.nullcontext[None]:
        sent.append(request.full_url)
        return contextlib.nullcontext()

    clock = _Clock()
    events = RestWorkerEvents(_CONFIG, opener=opener, clock=clock, hostname="node07")
    events.progress(0.0, 0.0)  # before any phase: the bare number, as before
    for name in (LOADING, MESHING, COMPILING, SOLVING):
        events.phase(name, 0.0, 0.0)  # never throttled
    events.progress(0.2, 0.2)  # within 5 s of the last: throttled
    clock.now = 10.0
    events.progress(0.37, 0.37)
    events.phase(WRITING, 1.0, 1.0)
    events.completed(1.0)
    queries = [dict(_query(url)) for url in sent]
    assert [(q["WorkerEvent_Status"], q.get("WorkerEvent_StatusMsg"), q["WorkerEvent_Progress"]) for q in queries] == [
        ("1001", None, "0.0"),
        ("1001", "WORKEREVENT_PROGRESS|loading model", "0.0"),
        ("1001", "WORKEREVENT_PROGRESS|meshing", "0.0"),
        ("1001", "WORKEREVENT_PROGRESS|compiling", "0.0"),
        ("1001", "WORKEREVENT_PROGRESS|solving", "0.0"),
        ("1001", "WORKEREVENT_PROGRESS|solving", "0.37"),
        ("1001", "WORKEREVENT_PROGRESS|writing results", "1.0"),
        ("1003", None, "1.0"),
    ]


def test_failure_message_is_sanitized_and_truncated() -> None:
    events = RestWorkerEvents(_CONFIG, hostname="node07")
    query = dict(_query(events.url(JOB_FAILURE, progress=0.1, t=0.0, message="bad 'x'\n\"y\"" + "z" * 5000)))
    message = query["WorkerEvent_StatusMsg"]
    assert len(message) == 2048 and not set("\n\r'\"") & set(message)
    assert query["JMSDeliveryMode"] == "persistent"


class _Broker(http.server.BaseHTTPRequestHandler):
    received: ClassVar[list[tuple[str, str, str | None]]] = []

    def do_POST(self) -> None:  # the http.server API
        _Broker.received.append((self.command, self.path, self.headers.get("Authorization")))
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def broker() -> Iterator[int]:
    _Broker.received = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _Broker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    thread.join()


def test_events_reach_a_broker_with_basic_auth(broker: int) -> None:
    config = MessagingConfig(**{**_CONFIG.__dict__, "broker_host": "127.0.0.1", "broker_port": broker})
    clock = _Clock()
    events = RestWorkerEvents(config, clock=clock, hostname="node07")
    events.starting()
    events.progress(0.1, 0.001)
    events.progress(0.2, 0.002)  # within 5 s of the last: throttled
    clock.now = 10.0
    events.data(0.005, 0.5)
    events.progress(0.6, 0.006)
    events.completed(0.01)
    statuses = [dict(_query(path))["WorkerEvent_Status"] for _, path, _ in _Broker.received]
    assert statuses == [str(JOB_STARTING), str(JOB_PROGRESS), str(JOB_DATA), str(JOB_PROGRESS), str(JOB_COMPLETED)]
    methods, auths = {m for m, _, _ in _Broker.received}, {a for _, _, a in _Broker.received}
    assert methods == {"POST"} and auths == {"Basic bXNnX3VzZXI6bXNnX3Bzd2Q="}  # msg_user:msg_pswd


def test_an_unreachable_broker_never_raises(capsys: pytest.CaptureFixture[str]) -> None:
    config = MessagingConfig(**{**_CONFIG.__dict__, "broker_host": "127.0.0.1", "broker_port": 9})
    events = RestWorkerEvents(config, timeout=0.5, hostname="node07")
    events.starting()
    events.failed("boom", 0.0, 0.0)
    events.completed(1.0)
    assert capsys.readouterr().err.count("status message to the VCell broker failed") == 1  # logged once


def test_messaging_config_reads_the_langevin_properties_format(tmp_path: Path) -> None:
    path = tmp_path / "SimID_1_0_.fenicsMessagingConfig"
    path.write_text(
        "broker_host=localhost\nbroker_port=8165\nbroker_username=msg_user\nbroker_password=msg_pswd\n"
        "vc_username=vcell_user\nsimKey=12334483837\ntaskID=0\njobIndex=3\n"
    )
    config = MessagingConfig.from_properties(path)
    assert (config.broker_host, config.broker_port, config.job_index) == ("localhost", 8165, 3)
    path.write_text("broker_host=localhost\n")
    with pytest.raises(ValueError, match="incomplete"):
        MessagingConfig.from_properties(path)


def test_fanout_reaches_every_reporter() -> None:
    from io import StringIO

    a, b = StringIO(), StringIO()
    Fanout([StdoutMarkers(a), StdoutMarkers(b)]).data(0.5, 0.5)
    assert vcell_scan(a.getvalue()) == vcell_scan(b.getvalue()) == ["data:0.5", "progress:50.0%"]


# -- the CLI end to end (subprocesses: isolate_stdout rewires file descriptor 1) ---------------------------

_ROOT = Path(__file__).resolve().parent.parent
_SMOKE = _ROOT / "tests" / "fixtures" / "simtask" / "SimID_1585623750_0__0.simtask.xml"
# a task the solver refuses (a compartmental, non-spatial one): the refusal's reporting path
_REFUSED = _ROOT / "tests" / "fixtures" / "simtask" / "SimID_274631114_0__0.simtask.xml"


def _cli(*args: str, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "vcell_fenics.cli", *args], capture_output=True, text=True, timeout=timeout
    )


def _config_file(tmp_path: Path, port: int) -> Path:
    path = tmp_path / "SimID_1585623750_0_.fenicsMessagingConfig"
    path.write_text(
        f"broker_host=127.0.0.1\nbroker_port={port}\nbroker_username=msg_user\nbroker_password=msg_pswd\n"
        "vc_username=vcell_user\nsimKey=1585623750\ntaskID=0\njobIndex=0\n"
    )
    return path


def test_print_status_stdout_is_markers_only(tmp_path: Path) -> None:
    result = _cli("--simtask", str(_SMOKE), "--out", str(tmp_path), "--vc-print-status", "-tid", "0")
    assert result.returncode == 0, result.stderr
    messages = vcell_scan(result.stdout)
    assert result.stdout.strip() == "".join(f"[[[{m}]]]\n" for m in messages).strip()  # nothing else
    parsed = [vcell_parse(m) for m in messages]  # VCell would throw on anything unrecognized
    assert [t for kind, t in parsed if kind == "data"] == pytest.approx([0.0, 0.005, 0.01])
    assert parsed[0] == ("progress", 0.0) and parsed[-1] == ("progress", 1.0)
    phases = [p for p in (vcell_phase(m) for m in messages) if p is not None]
    assert list(dict.fromkeys(phases)) == list(PHASES)  # every phase, in order
    assert "wrote" in result.stderr  # the log went to stderr


def test_send_status_reports_the_run_to_the_broker(tmp_path: Path, broker: int) -> None:
    config = _config_file(tmp_path, broker)
    result = _cli("--simtask", str(_SMOKE), "--out", str(tmp_path), f"--vc-send-status-config={config}")
    assert result.returncode == 0, result.stderr
    statuses = [int(dict(_query(path))["WorkerEvent_Status"]) for _, path, _ in _Broker.received]
    assert statuses[0] == JOB_STARTING and statuses[-1] == JOB_COMPLETED
    assert JOB_DATA in statuses and JOB_FAILURE not in statuses
    messages = [dict(_query(path)).get("WorkerEvent_StatusMsg") for _, path, _ in _Broker.received]
    phases = [m.split("|", 1)[1] for m in messages if m and m.startswith("WORKEREVENT_PROGRESS|")]
    assert list(dict.fromkeys(phases)) == list(PHASES)
    last = dict(_query(_Broker.received[-1][1]))
    assert (last["SimKey"], last["JobIndex"], last["WorkerEvent_TimePoint"]) == ("1585623750", "0", "0.01")


def test_a_refused_task_reports_failure_and_exits_2(tmp_path: Path, broker: int) -> None:
    config = _config_file(tmp_path, broker)
    result = _cli("--simtask", str(_REFUSED), "--out", str(tmp_path), f"--vc-send-status-config={config}")
    assert result.returncode == 2
    statuses = [int(dict(_query(path))["WorkerEvent_Status"]) for _, path, _ in _Broker.received]
    assert statuses == [JOB_STARTING, JOB_PROGRESS, JOB_FAILURE]  # refused while loading the model
    assert dict(_query(_Broker.received[1][1]))["WorkerEvent_StatusMsg"] == "WORKEREVENT_PROGRESS|loading model"
    assert "non-spatial" in dict(_query(_Broker.received[-1][1]))["WorkerEvent_StatusMsg"]


def test_an_unreachable_broker_does_not_fail_the_run(tmp_path: Path) -> None:
    config = _config_file(tmp_path, 9)  # nothing listens on the discard port
    result = _cli("--simtask", str(_SMOKE), "--out", str(tmp_path), "--no-fields", f"--vc-send-status-config={config}")
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("status message to the VCell broker failed") == 1


def test_sigterm_marks_the_bundle_failed_and_exits_143(tmp_path: Path) -> None:
    """A Slurm cancel or timeout: the run unwinds, the manifest says failed, the exit status is 143."""

    import json
    import signal
    import time

    models = _ROOT / "examples" / "models"
    process = subprocess.Popen(
        [
            sys.executable, "-m", "vcell_fenics.cli",
            "--math", str(models / "diffusion2d_math.yaml"), "--geometry", str(models / "diffusion2d_geom.yaml"),
            "--t-final", "1000", "--output-dt", "1", "--dt", "0.01", "--h", "0.1", "--no-fields",
            "--out", str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )  # fmt: skip
    manifest = tmp_path / "results.fenics" / ".zattrs"
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline and process.poll() is None:  # wait until rows are being written
        if (
            manifest.is_file() and json.loads(manifest.read_text())["vcell_fenics"]["times"]
        ):  # atomic: always a manifest
            break
        time.sleep(0.2)
    assert process.poll() is None, "the run ended before it could be interrupted"
    process.send_signal(signal.SIGTERM)
    _, stderr = process.communicate(timeout=120)
    assert process.returncode == 143, stderr
    state = json.loads(manifest.read_text())["vcell_fenics"]
    assert state["status"] == "failed" and "SIGTERM" in stderr
