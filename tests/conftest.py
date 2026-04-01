# ---------------------------------------------------------------------------
# conftest.py — pytest fixtures and hooks for GPU integration tests and
# optional benchmarking.
#
# Provides:
#   - GPU availability detection (Metal on macOS, CUDA on Linux)
#   - rLLM binary discovery
#   - Model directory resolution
#   - Server lifecycle management (start/stop rllm serve, health checks)
#   - Available memory detection for MoE streaming decisions
#   - --bench mode: performance measurement integrated into pytest runs
#
# Usage:
#   pytest tests/ -v                         # test + benchmark all models
#   pytest tests/ -v --filter llama          # test + benchmark specific models
#
# Related: models.py (model registry), quality.py (validation),
#          benchmark.py (measurement engine), test_model_families.py (tests)
# ---------------------------------------------------------------------------

import os
import platform
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil
import pytest
import requests

from benchmark import BenchResult, detect_gpu_name, format_markdown_table, write_results, append_result_line, _resolve_output_path
from models import MODEL_REGISTRY, ModelConfig, is_base_model, check_model_shards


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_rllm_binary() -> Path | None:
    """Locate the rllm release binary relative to the repo root."""
    if env := os.environ.get("RLLM_BIN"):
        p = Path(env)
        return p if p.is_file() else None

    here = Path(__file__).resolve().parent
    for ancestor in [here] + list(here.parents):
        candidate = ancestor / "target" / "release" / "rllm"
        if candidate.is_file():
            return candidate
    return None


def _has_gpu() -> bool:
    """Return True if a Metal or CUDA GPU is available."""
    if platform.system() == "Darwin":
        return True
    try:
        subprocess.run(
            ["nvidia-smi"], capture_output=True, check=True, timeout=10
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _get_free_port() -> int:
    """Bind to port 0 and return the OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_gpu_count() -> int:
    """Detect the number of available GPUs.

    macOS: always 1 (Metal, unified memory).
    Linux + CUDA: count via nvidia-smi.
    """
    if platform.system() == "Darwin":
        return 1
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return len([l for l in result.stdout.strip().splitlines() if l.strip()])
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return 1


def _get_available_memory_gb() -> float:
    """Detect total available GPU/system memory in GB.

    macOS (unified memory): total system RAM.
    Linux + CUDA: sum of all GPU memory via nvidia-smi.
    """
    if platform.system() == "Darwin":
        return psutil.virtual_memory().total / (1024 ** 3)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        total_mib = sum(
            int(line.strip())
            for line in result.stdout.strip().splitlines()
            if line.strip().isdigit()
        )
        return total_mib / 1024
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return psutil.virtual_memory().total / (1024 ** 3)


def _get_model_disk_size_gb(model_dir: str) -> float:
    """Sum safetensors file sizes to get actual model size on disk."""
    total = sum(f.stat().st_size for f in Path(model_dir).glob("*.safetensors"))
    return total / (1024 ** 3)


def _should_stream_experts(model_size_gb: float, is_q4: bool, model_dir: str = "") -> bool:
    """Decide whether to use --stream-experts based on available memory.

    Uses actual model size on disk when available (accurate for Q4/Q8 variants),
    falling back to a bf16 size estimate otherwise.
    """
    if model_dir:
        effective_size = _get_model_disk_size_gb(model_dir)
    else:
        effective_size = model_size_gb / 1.5 if is_q4 else model_size_gb
    available = _get_available_memory_gb()
    return effective_size > available * 0.80


# ---------------------------------------------------------------------------
# Server manager — caches running servers to avoid restart per test.
# ---------------------------------------------------------------------------

@dataclass
class ServerProcess:
    """A running rllm serve instance."""
    proc: subprocess.Popen
    base_url: str
    config_key: str
    log_thread: "threading.Thread | None" = None


class ServerManager:
    """Manages rllm serve subprocesses — one server at a time.

    Stops the previous server before starting a new one to prevent OOM
    when running large models sequentially.  Reuses the existing server
    if the new request matches (same model + args).
    """

    def __init__(self, binary: Path, gpu_count: int = 1):
        self.binary = binary
        self.gpu_count = gpu_count
        self._servers: dict[str, ServerProcess] = {}

    def _config_key(self, model_dir: str, extra_args: list[str]) -> str:
        return f"{model_dir}|{'|'.join(extra_args)}"

    def get_or_start(
        self,
        model_dir: str,
        extra_args: list[str] | None = None,
        startup_timeout: float = 300,
        memory_gb: float = 0,
    ) -> str:
        """Return the base_url of a running server for this config.

        Reuses the server if the same config is already running.
        Otherwise stops ALL existing servers first, then starts fresh.
        """
        extra_args = extra_args or []
        key = self._config_key(model_dir, extra_args)

        if key in self._servers:
            srv = self._servers[key]
            try:
                r = requests.get(f"{srv.base_url}/health", timeout=5)
                if r.status_code == 200:
                    return srv.base_url
            except requests.ConnectionError:
                pass
            self._stop_one(key)

        self.stop_all()

        port = _get_free_port()
        base_url = f"http://127.0.0.1:{port}"

        cmd = [
            str(self.binary), "serve",
            "--model", model_dir,
            "--port", str(port),
            "--tp", str(self.gpu_count),
            *extra_args,
        ]

        print(f"\n  Starting rllm serve: {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                raise RuntimeError(
                    f"rllm serve exited with code {proc.returncode} "
                    f"before becoming healthy.\nstderr:\n{stderr[-2000:]}"
                )
            try:
                r = requests.get(f"{base_url}/health", timeout=2)
                if r.status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(1)
        else:
            stderr = ""
            proc.terminate()
            try:
                proc.wait(timeout=10)
                stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            except subprocess.TimeoutExpired:
                proc.kill()
            raise RuntimeError(
                f"rllm serve did not become healthy within {startup_timeout}s.\n"
                f"stderr:\n{stderr[-2000:]}"
            )

        self._servers[key] = ServerProcess(proc=proc, base_url=base_url, config_key=key)
        print(f"  Server healthy at {base_url}", file=sys.stderr)
        return base_url

    def _stop_one(self, key: str) -> None:
        srv = self._servers.pop(key, None)
        if srv is None:
            return
        print(f"\n  Stopping server {srv.base_url}", file=sys.stderr)
        try:
            srv.proc.terminate()
            srv.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            srv.proc.kill()
            srv.proc.wait(timeout=5)

    def stop_all(self) -> None:
        for key in list(self._servers):
            self._stop_one(key)


# ---------------------------------------------------------------------------
# Benchmark context — collects results when --bench is active.
# ---------------------------------------------------------------------------

class BenchContext:
    """Collects benchmark measurements during a pytest session.

    When --bench is not active, all methods are no-ops.
    Results are streamed to a live output file as they arrive, so you
    can `tail -f results/bench-*.md` to watch progress during a run.
    """

    def __init__(self, enabled: bool = False, runs: int = 1,
                 max_tokens: int = 128, prompt: str = "",
                 live_output_path: "Path | None" = None):
        self.enabled = enabled
        self.runs = runs
        self.max_tokens = max_tokens
        self.prompt = prompt
        self.results: list[BenchResult] = []
        self.live_output_path = live_output_path

    def record(self, model_name: str, family: str,
               gen_tps: float | None = None, ttft_ms: float | None = None,
               quality: str = "", scores: dict[str, float] | None = None) -> None:
        """Record a benchmark result.  No-op if --bench is not active.

        Each result is immediately appended to the live output file.
        """
        if not self.enabled:
            return
        result = BenchResult(
            model_name=model_name,
            family=family,
            gen_tps=gen_tps,
            ttft_ms=ttft_ms,
            quality=quality,
            scores=scores or {},
        )
        self.results.append(result)
        if self.live_output_path:
            append_result_line(result, self.live_output_path)


# ---------------------------------------------------------------------------
# pytest CLI options for --bench mode
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    group = parser.getgroup("bench", "rLLM benchmark options")
    group.addoption("--bench", action="store_true", default=True,
                    help="Benchmark mode (always on — measures tok/s + TTFT)")
    group.addoption("--bench-runs", type=int, default=1,
                    help="Number of runs per model (default: 1)")
    group.addoption("--bench-max-tokens", type=int, default=128,
                    help="Max tokens to generate per run (default: 128)")
    group.addoption("--bench-prompt", default="The meaning of life is",
                    help="Override benchmark prompt")
    group.addoption("--bench-output", default=None,
                    help="Write markdown results to this file path")
    group.addoption("--filter", default="",
                    help="Only run/bench models matching this substring (filters both tests and benchmarks)")
    group.addoption("--bench-filter", default="",
                    help="Alias for --filter (deprecated)")
    group.addoption("--bench-q4-only", action="store_true", default=False,
                    help="Skip bf16 variants, only bench Q4")
    group.addoption("--bench-bf16-only", action="store_true", default=False,
                    help="Skip Q4 variants")
    group.addoption("--bench-max-size", type=float, default=0,
                    help="Skip models exceeding this disk size in GB (0 = no limit)")


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rllm_binary():
    """Locate the rllm release binary; skip all tests if not found."""
    binary = _find_rllm_binary()
    if binary is None:
        pytest.skip("rllm binary not found (build with: cargo build --release)")
    return binary


@pytest.fixture(scope="session")
def gpu_available():
    """Skip all tests if no GPU is detected."""
    if not _has_gpu():
        pytest.skip("no GPU detected (need Metal on macOS or CUDA on Linux)")


@pytest.fixture(scope="session")
def models_dir():
    """Resolve the models directory."""
    d = Path(os.environ.get("RLLM_MODELS_DIR", "models"))
    if not d.is_dir():
        here = Path(__file__).resolve().parent
        for ancestor in [here] + list(here.parents):
            candidate = ancestor / "models"
            if candidate.is_dir():
                return candidate
        pytest.skip(f"models directory not found: {d}")
    return d


@pytest.fixture(scope="session")
def available_memory_gb():
    """Return available GPU/system memory in GB."""
    return _get_available_memory_gb()


@pytest.fixture(scope="session")
def gpu_count():
    """Detect number of available GPUs for tensor parallelism."""
    return _get_gpu_count()


@pytest.fixture(scope="session")
def server_manager(rllm_binary, gpu_available, gpu_count):
    """Session-scoped server manager; stops all servers at teardown."""
    mgr = ServerManager(rllm_binary, gpu_count=gpu_count)
    yield mgr
    mgr.stop_all()


@pytest.fixture(scope="session")
def bench_context(request):
    """Session-scoped benchmark context.  No-op when --bench is not active.

    Returns the same BenchContext instance created by pytest_sessionstart,
    so results recorded here are visible to pytest_terminal_summary.
    """
    session = request.config._tmp_session
    if hasattr(session, "_bench_context"):
        return session._bench_context
    # --bench not active — return a disabled context.
    return BenchContext(enabled=False)


# ---------------------------------------------------------------------------
# pytest hooks — terminal summary for --bench results
# ---------------------------------------------------------------------------

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print and save benchmark results."""
    # Access the bench_context fixture from the session.
    session = terminalreporter.config._tmp_session
    if not hasattr(session, "_bench_context"):
        return

    ctx = session._bench_context
    if not ctx.results:
        terminalreporter.write_line("\nNo benchmark results collected.", yellow=True)
        return

    gpu_name = detect_gpu_name()
    gpu_count = _get_gpu_count()

    table = format_markdown_table(
        ctx.results, gpu_name, gpu_count, ctx.max_tokens, ctx.runs,
    )
    terminalreporter.write_line("")
    terminalreporter.write_line(table)

    # Write results to file.
    output = config.getoption("--bench-output", default=None)
    output_path = Path(output) if output else None
    path = write_results(
        ctx.results, gpu_name, gpu_count, ctx.max_tokens, ctx.runs,
        output_path=output_path,
    )
    terminalreporter.write_line(f"Results saved to {path}")


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Create the shared BenchContext and stash it on the session.

    The bench_context fixture and pytest_terminal_summary both read from
    session._bench_context — one instance shared across the entire run.
    """
    # Always stash the session so the fixture can find it.
    # Bench is always on — every test run measures performance and writes results.
    session.config._tmp_session = session
    output = session.config.getoption("--bench-output", default=None)
    live_path = _resolve_output_path(Path(output) if output else None)
    session._bench_context = BenchContext(
        enabled=True,
        runs=session.config.getoption("--bench-runs", default=1),
        max_tokens=session.config.getoption("--bench-max-tokens", default=128),
        prompt=session.config.getoption("--bench-prompt",
                                        default="The meaning of life is"),
        live_output_path=live_path,
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "bench_only: only run in --bench mode")
    config.addinivalue_line("markers", "gpu: requires GPU (Metal on macOS, CUDA on Linux)")


collect_ignore_glob = ["test_nemotron_debug.py"]


def pytest_collection_modifyitems(config, items):
    """Apply --filter to skip non-matching model tests."""
    model_filter = (config.getoption("--filter", default="")
                    or config.getoption("--bench-filter", default=""))
    if model_filter:
        skip_filter = pytest.mark.skip(reason=f"--filter '{model_filter}' not matched")
        for item in items:
            # Only filter parametrized model tests (test_model_bf16[xxx], etc.)
            if "[" in item.nodeid:
                param_id = item.nodeid.split("[")[-1].rstrip("]")
                if model_filter not in param_id:
                    item.add_marker(skip_filter)
