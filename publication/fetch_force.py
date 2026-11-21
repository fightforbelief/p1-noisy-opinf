"""Rebuild the archived time and total-force columns from 12 PhysioNet records."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

BASE = "https://physionet.org/files/gaitpdb/1.0.0/"
SUBJECTS = [f"GaCo{i:02d}" for i in range(1, 13)]


def download(name):
    return subprocess.check_output(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "12",
                                    "--max-time", "75", BASE+name])


def fetch(output):
    checks = {name.lstrip("*"): value for value, name in
              (line.split() for line in download("SHA256SUMS.txt").decode().splitlines() if line.strip())}
    def columns(subject):
        name = subject+"_01.txt"
        raw = download(name)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != checks[name]:
            raise ValueError(f"Checksum mismatch for {name}")
        return subject, np.loadtxt(io.BytesIO(raw), usecols=(0,17,18)), digest
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(columns, SUBJECTS))
    output.mkdir(parents=True, exist_ok=True)
    with (output/"observations.npz").open("xb") as f:
        np.savez_compressed(f, **{s:a for s,a,_ in records})
    sources = {s:{"url":BASE+s+"_01.txt", "sha256":digest} for s,_,digest in records}
    (output/"sources.json").write_text(json.dumps(sources, indent=2)+"\n", encoding="utf8")
    print(f"Verified {len(records)} source records", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    fetch(p.parse_args().output)
