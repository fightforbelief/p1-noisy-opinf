"""Fetch the fixed NOAA CO-OPS station-month requests and preserve each raw response."""
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
from urllib.parse import urlencode

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "tide_protocol.json").read_text())
API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


def fetch(pair):
    station, month = pair
    last = calendar.monthrange(int(month[:4]), int(month[4:]))[1]
    query = dict(product="water_level", application="p1research", begin_date=month+"01",
                 end_date=month+str(last), datum="MLLW", station=station, time_zone="gmt",
                 units="metric", format="json")
    url = API+"?"+urlencode(query)
    for attempt in range(4):
        try:
            raw = subprocess.check_output(["curl", "-fsSL", "--connect-timeout", "12",
                                           "--max-time", "65", url])
            records = json.loads(raw)
            if records.get("metadata", {}).get("id") != station or not records.get("data"):
                raise ValueError(f"Missing NOAA observations for {station} {month}: {str(records)[:120]}")
            break
        except (OSError, ValueError, subprocess.CalledProcessError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    return station+"_"+month+".json", url, raw


def run(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs = [(s, m) for s in PLAN["stations"] for m in PLAN["months"]]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, pairs))
    manifest = {}
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:xz") as archive:
        for name, url, raw in results:
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
            manifest[name] = {"url": url, "sha256": hashlib.sha256(raw).hexdigest()}
        contents = json.dumps(manifest, indent=2).encode()
        member = tarfile.TarInfo("manifest.json")
        member.size = len(contents)
        archive.addfile(member, io.BytesIO(contents))
    print(f"{len(results)} station-months; {output.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
