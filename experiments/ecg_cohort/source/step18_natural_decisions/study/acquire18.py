"""Transport-only acquisition for Step 18; never opens waveform members."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

WORKSPACE = Path(__file__).resolve().parents[2]
STEP_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_URL = "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
RESOLVED_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip"
EXPECTED_BYTES = 1_839_504_686
DEFAULT_ARCHIVE_PATH = WORKSPACE / "tmp" / "ptbxl18_source.zip"
DEFAULT_RECEIPT_PATH = STEP_ROOT / "references" / "ptbxl_v103" / "archive_acquisition18.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_archive(archive_path=DEFAULT_ARCHIVE_PATH, receipt_path=DEFAULT_RECEIPT_PATH):
    """Download opaque ZIP bytes, hash them, and seal a write-once receipt.

    This is deliberately separate from scientific extraction. No ZIP member,
    waveform header, or sample value is opened here. The full ZIP hash is a
    local transport pin; independently published member hashes are checked
    during extraction after the experimental freeze.
    """
    archive_path, receipt_path = Path(archive_path), Path(receipt_path)
    if archive_path.exists() or receipt_path.exists():
        raise FileExistsError("Archive or receipt already exists; acquisition is write-once")
    partial = archive_path.with_suffix(archive_path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"Incomplete acquisition exists and must be audited: {partial}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "Step18-Provenance/1.0"})
    digest, total = hashlib.sha256(), 0
    started = time.time()
    last_progress = started
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"Unexpected HTTP status {response.status}")
        if response.url != RESOLVED_URL:
            raise RuntimeError(f"Unexpected archive redirect {response.url}")
        if int(response.headers.get("Content-Length", "-1")) != EXPECTED_BYTES:
            raise RuntimeError("Official archive byte count changed")
        headers = dict(response.headers.items())
        with partial.open("xb") as target:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                total += len(chunk)
                if total > EXPECTED_BYTES:
                    raise RuntimeError("Archive exceeded the frozen byte count")
                now = time.time()
                if now - last_progress >= 15:
                    print(json.dumps({"event": "archive_transport_progress", "bytes": total,
                                      "expected_bytes": EXPECTED_BYTES,
                                      "elapsed_seconds": round(now - started, 1)}), flush=True)
                    last_progress = now
            target.flush()
            os.fsync(target.fileno())
    if total != EXPECTED_BYTES:
        raise RuntimeError(f"Truncated archive: {total} != {EXPECTED_BYTES}")
    # os.link fails if a second writer has already installed the destination.
    os.link(partial, archive_path)
    partial.unlink()
    receipt = {"schema": "step18-archive-acquisition-v1", "source_url": ARCHIVE_URL,
               "resolved_url": RESOLVED_URL, "archive_path": str(archive_path.resolve()),
               "bytes": total, "sha256": digest.hexdigest(), "http_headers": headers,
               "started_unix": started, "completed_unix": time.time(),
               "content_scope": "Opaque transport only; no ZIP members or signal/header values opened",
               "checksum_scope": "Local full-archive SHA256 pin; official per-member SHA256 checks deferred to frozen extraction"}
    with receipt_path.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"event": "archive_acquisition_complete", "bytes": total,
                      "sha256": receipt["sha256"], "receipt": str(receipt_path)}), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT_PATH)
    args = parser.parse_args()
    acquire_archive(args.archive, args.receipt)
