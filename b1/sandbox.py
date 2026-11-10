"""Sandbox construction and execution.

The candidate program is executed once, in a throwaway directory that contains
only its input files, with sockets disabled and a hard timeout. This is a
research harness, not a security boundary: for untrusted code use a container.
"""
import ast
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

FORBIDDEN_IMPORTS = {
    "os", "subprocess", "socket", "urllib", "http", "requests", "shutil",
    "glob", "ctypes", "multiprocessing", "threading", "webbrowser", "ftplib",
    "telnetlib", "smtplib", "pickle", "marshal",
}
FORBIDDEN_TOKENS = ("ground_truth", "http://", "https://", "C:\\", "/Users/", "/home/")

SITECUSTOMIZE = '''\
import socket
def _blocked(*a, **k):
    raise OSError("network access is disabled in the B1 sandbox")
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
'''


def check_source(code: str) -> list:
    """Best-effort static check. Returns warnings; it does not block execution."""
    problems = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    problems.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                problems.append(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Name) and node.id in ("eval", "exec", "compile", "__import__"):
            problems.append(f"forbidden call: {node.id}")
    for token in FORBIDDEN_TOKENS:
        if token in code:
            problems.append(f"suspicious literal: {token!r}")
    return problems


def _new_box(prefix: str) -> Path:
    RUNS.mkdir(exist_ok=True)
    box = RUNS / f"{prefix}_{uuid.uuid4().hex[:8]}"
    box.mkdir()
    (box / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    return box


def make_json_sandbox(batch: Path, code: str, ref=None, test=None) -> Path:
    import json
    box = _new_box("json")
    if ref is None:
        shutil.copy(batch / "reference_samples.json", box / "reference_samples.json")
    else:
        (box / "reference_samples.json").write_text(
            json.dumps(ref, ensure_ascii=False), encoding="utf-8")
    if test is None:
        shutil.copy(batch / "samples.json", box / "samples.json")
    else:
        (box / "samples.json").write_text(
            json.dumps(test, ensure_ascii=False), encoding="utf-8")
    (box / "candidate.py").write_text(code, encoding="utf-8")
    return box


def make_csv_sandbox(batch: Path, code: str, train=None, test=None) -> Path:
    box = _new_box("csv")
    if train is None:
        shutil.copy(batch / "train.csv", box / "train.csv")
    else:
        train.to_csv(box / "train.csv", index=False, float_format="%.4f")
    if test is None:
        shutil.copy(batch / "test.csv", box / "test.csv")
    else:
        test.to_csv(box / "test.csv", index=False, float_format="%.4f")
    (box / "candidate.py").write_text(code, encoding="utf-8")
    return box


def run_sandbox(box: Path, timeout: int):
    """Run candidate.py once. Returns (CompletedProcess|None, elapsed, timed_out)."""
    wrapper = (
        "import sys; sys.argv=['candidate.py', '.']; sys.path.insert(0, '.'); "
        "import sitecustomize; import runpy; "
        "runpy.run_path('candidate.py', run_name='__main__')"
    )
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8",
           "PYTHONDONTWRITEBYTECODE": "1"}
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-E", "-s", "-B", "-c", wrapper],
            cwd=box, capture_output=True, text=True, timeout=timeout, env=env,
        )
        timed_out = False
    except subprocess.TimeoutExpired:
        proc, timed_out = None, True
    return proc, time.time() - start, timed_out


def read_predictions(box: Path):
    path = box / "predictions.json"
    if not path.exists():
        return None
    try:
        return __import__("json").loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
