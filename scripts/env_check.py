"""记录远程环境。只读检查，不修改全局 Python。"""

from __future__ import annotations

import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

PACKAGES = ("numpy", "scipy", "pandas", "matplotlib", "pyyaml", "pytest", "joblib", "tqdm")


def _memory_gb() -> str:
    if sys.platform != "win32":
        return "not-windows"
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return f"{status.ullTotalPhys / 1024**3:.1f} GB"


def main() -> None:
    lines = [
        "# ENV_CHECK",
        "",
        f"- hostname: {platform.node()}",
        f"- platform: {platform.platform()}",
        f"- python: {sys.version}",
        f"- executable: {sys.executable}",
        f"- cpu_count: {os.cpu_count()}",
        f"- memory: {_memory_gb()}",
        "",
        "## packages",
        "",
    ]
    for name in PACKAGES:
        lines.append(f"- {name}=={version(name)}")
    root = Path(__file__).resolve().parents[1]
    path = root / "reports" / "ENV_CHECK.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
