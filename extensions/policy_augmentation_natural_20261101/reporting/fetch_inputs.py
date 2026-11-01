#!/usr/bin/env python3
"""Recover pinned public Train inputs without accessing official Test data."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "6c1067e5a57161249b17289a565178dc7a3fb3ca"
BASE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/" + COMMIT + "/Dataset/"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    frozen = json.loads((ROOT / "freeze.json").read_text())["input_files"]
    expected_names = {"Hourly-train.csv", "Weekly-train.csv", "M4-info.csv"}
    assert {Path(name).name for name in frozen} == expected_names
    for relative, expected in sorted(frozen.items()):
        target = ROOT / relative
        assert target.parent == ROOT / "input_sources"
        if target.exists():
            if digest(target) != expected:
                raise ValueError("Existing input has a different frozen identity: " + target.name)
            print(target.name + ": verified")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.name if target.name == "M4-info.csv" else "Train/" + target.name
        temporary = target.with_suffix(target.suffix + ".download")
        if temporary.exists():
            raise FileExistsError(temporary)
        with urllib.request.urlopen(BASE + suffix) as response, temporary.open("xb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)
        if digest(temporary) != expected:
            raise ValueError("Downloaded input does not match freeze: " + target.name)
        temporary.replace(target)
        print(target.name + ": recovered and verified")


if __name__ == "__main__":
    main()
