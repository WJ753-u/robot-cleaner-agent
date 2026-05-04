from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def start_process(name: str, command: list[str], env: dict[str, str]) -> subprocess.Popen:
    print(f"[start] {name}: {' '.join(command)}", flush=True)
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
    )


def stop_process(name: str, process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    print(f"[stop] {name}", flush=True)
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start FastAPI and Streamlit together.")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8501)
    parser.add_argument("--api-reload", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["AGENT_API_BASE_URL"] = f"http://{args.api_host}:{args.api_port}/api/v1"

    api_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        args.api_host,
        "--port",
        str(args.api_port),
    ]
    if args.api_reload:
        api_command.append("--reload")

    web_command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app.py"),
        "--server.address",
        args.web_host,
        "--server.port",
        str(args.web_port),
    ]

    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        processes.append(("FastAPI", start_process("FastAPI", api_command, env)))
        time.sleep(1)
        processes.append(("Streamlit", start_process("Streamlit", web_command, env)))

        print(
            f"[ready] API: http://{args.api_host}:{args.api_port}/docs",
            flush=True,
        )
        print(
            f"[ready] Web: http://{args.web_host}:{args.web_port}",
            flush=True,
        )
        print("[info] Press Ctrl+C to stop both services.", flush=True)

        while True:
            for name, process in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    raise RuntimeError(f"{name} exited with code {exit_code}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[info] stopping services...", flush=True)
    finally:
        for name, process in reversed(processes):
            stop_process(name, process)


if __name__ == "__main__":
    main()
