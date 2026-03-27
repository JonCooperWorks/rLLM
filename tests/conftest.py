# ---------------------------------------------------------------------------
# conftest.py — pytest fixtures for GPU integration tests.
#
# Provides:
#   - GPU availability detection (Metal on macOS, CUDA on Linux)
#   - rLLM binary discovery
#   - Model directory resolution
#   - Server lifecycle management (start/stop rllm serve, health checks)
#   - Available memory detection for MoE streaming decisions
#
# Related: test_model_families.py (tests), coherence.py (validation)
# ---------------------------------------------------------------------------

import json
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_rllm_binary() -> Path | None:
    """Locate the rllm release binary relative to the repo root."""
    # Allow override via env var.
    if env := os.environ.get("RLLM_BIN"):
        p = Path(env)
        return p if p.is_file() else None

    # Walk up from this file to find the repo root (contains Cargo.toml).
    here = Path(__file__).resolve().parent
    for ancestor in [here] + list(here.parents):
        candidate = ancestor / "target" / "release" / "rllm"
        if candidate.is_file():
            return candidate
    return None


def _has_gpu() -> bool:
    """Return True if a Metal or CUDA GPU is available."""
    if platform.system() == "Darwin":
        # macOS — Metal is always available on Apple Silicon / recent Intel.
        return True
    # Linux — check for nvidia-smi (CUDA).
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

    For multi-GPU setups with tensor parallelism, the full aggregate memory
    is available since rllm shards the model across all GPUs.
    """
    if platform.system() == "Darwin":
        return psutil.virtual_memory().total / (1024 ** 3)

    # Linux: try nvidia-smi for GPU memory.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        # Sum all GPUs, nvidia-smi reports in MiB.  With tensor parallelism
        # the model is sharded across all GPUs, so aggregate memory applies.
        total_mib = sum(
            int(line.strip())
            for line in result.stdout.strip().splitlines()
            if line.strip().isdigit()
        )
        return total_mib / 1024
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # Fallback to system RAM.
        return psutil.virtual_memory().total / (1024 ** 3)


def _should_stream_experts(model_size_gb: float, is_q4: bool) -> bool:
    """Decide whether to use --stream-experts based on available memory.

    Q4 compression ratio varies widely: dense weights get ~3.5x compression,
    but MoE models have many non-expert weights (embeddings, norms, attention)
    that stay bf16.  We use a conservative 1.5x estimate for Q4 MoE models
    to avoid OOM from underestimating memory needs.
    """
    effective_size = model_size_gb / 1.5 if is_q4 else model_size_gb
    available = _get_available_memory_gb()
    # Stream if model exceeds 80% of available memory.
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
        memory_gb: float = 0,  # unused, kept for call-site compat
    ) -> str:
        """Return the base_url of a running server for this config.

        Reuses the server if the same config is already running.
        Otherwise stops ALL existing servers first, then starts fresh.
        """
        extra_args = extra_args or []
        key = self._config_key(model_dir, extra_args)

        if key in self._servers:
            srv = self._servers[key]
            # Quick health check — server may have died.
            try:
                r = requests.get(f"{srv.base_url}/health", timeout=5)
                if r.status_code == 200:
                    return srv.base_url
            except requests.ConnectionError:
                pass
            # Dead — remove and restart.
            self._stop_one(key)

        # Stop all other servers before starting a new one (prevent OOM).
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

        # Wait for /health to respond.
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            # Check if process died.
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
            # Timed out.
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
        # Graceful shutdown.
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
        # Try relative to repo root.
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
    """Session-scoped server manager; stops all servers at teardown.

    Automatically passes --tp <gpu_count> to use all available GPUs.
    """
    mgr = ServerManager(rllm_binary, gpu_count=gpu_count)
    yield mgr
    mgr.stop_all()
