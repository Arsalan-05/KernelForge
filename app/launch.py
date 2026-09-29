"""One-command demo launcher (Kaggle/Colab friendly).

Starts the FastAPI server (API + web UI on one port), waits until the model
has loaded, and with --share opens a Cloudflare quick tunnel that prints a
temporary public https://*.trycloudflare.com URL (no account needed).
Ctrl+C / stopping the notebook cell shuts everything down.

    python app/launch.py --share            # base model, public link
    python app/launch.py --share --adapter training/checkpoints/run_x/adapter
    python app/launch.py --mock             # no GPU/model: UI plumbing check only
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).parent.parent
CLOUDFLARED_URLS = {
    "x86_64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    "aarch64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
}
TUNNEL_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _wait_for_backend(url: str, proc: subprocess.Popen, timeout_s: float) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"Server exited with code {proc.returncode} before becoming healthy; see its log above.")
        try:
            resp = httpx.get(f"{url}/health", timeout=5.0)
            if resp.status_code == 200:
                return resp.json()
        except httpx.HTTPError:
            pass
        time.sleep(3)
    sys.exit(f"Server did not become healthy within {timeout_s:.0f}s.")


def _cloudflared_binary() -> str:
    found = shutil.which("cloudflared")
    if found:
        return found
    if platform.system() != "Linux" or platform.machine() not in CLOUDFLARED_URLS:
        sys.exit("--share needs cloudflared: install it (e.g. `brew install cloudflared`) and retry.")
    target = Path.home() / ".cache" / "kernelforge" / "cloudflared"
    if not target.exists():
        print("Downloading cloudflared for the public tunnel...", flush=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CLOUDFLARED_URLS[platform.machine()], target)
        target.chmod(target.stat().st_mode | stat.S_IEXEC)
    return str(target)


def _start_tunnel(port: int, timeout_s: float = 60.0) -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen(
        [_cloudflared_binary(), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    found: list[str] = []
    ready = threading.Event()

    def _read() -> None:
        for line in proc.stdout:
            match = TUNNEL_URL_RE.search(line)
            if match and not found:
                found.append(match.group(0))
                ready.set()

    threading.Thread(target=_read, daemon=True).start()
    if not ready.wait(timeout_s):
        proc.terminate()
        sys.exit("cloudflared didn't report a public URL in time. On Kaggle, check that Internet is enabled.")
    return proc, found[0]


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--share", action="store_true", help="Open a public Cloudflare quick-tunnel URL")
    parser.add_argument("--adapter", help="LoRA adapter path (omit to run the base model)")
    parser.add_argument("--mock", action="store_true", help="No model; echo template kernels (UI testing only)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Generation cap; the 4096 training default is slow on a T4")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=1200.0,
                        help="Seconds to wait for model download + load")
    args = parser.parse_args()

    local_url = f"http://127.0.0.1:{args.port}"
    env = os.environ.copy()
    env["KERNELFORGE_PRELOAD"] = "1"
    env["KERNELFORGE_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    if args.adapter:
        env["KERNELFORGE_ADAPTER_PATH"] = args.adapter
    if args.mock:
        env["KERNELFORGE_MOCK_MODEL"] = "1"

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    print(f"Starting KernelForge on {local_url} (loading the model can take a few minutes on first run)...",
          flush=True)
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", str(args.port)],
        cwd=REPO_ROOT,
        env=env,
    )
    tunnel = None
    try:
        health = _wait_for_backend(local_url, server, args.startup_timeout)
        print(f"\nReady: {health['banner']}", flush=True)
        print(f"GPU: {health.get('gpu_name') or 'none'} · Triton: {health.get('triton_version') or 'not installed'}",
              flush=True)
        print(f"Local:  {local_url}", flush=True)
        if args.share:
            tunnel, public_url = _start_tunnel(args.port)
            print(f"Public: {public_url}   (temporary; lives as long as this process)", flush=True)
        server.wait()
    finally:
        _stop(tunnel)
        _stop(server)


if __name__ == "__main__":
    main()
