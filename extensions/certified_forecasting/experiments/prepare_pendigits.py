#!/usr/bin/env python3
"""Fetch and parse original UCI PenDigits trajectories without resampling.

Citation: E. Alpaydin and F. Alimoglu (1996), Pen-Based Recognition of
Handwritten Digits, UCI ML Repository, doi:10.24432/C5MG6K (CC BY 4.0).

The source files use UNIX compress (.Z). Install ``unlzw3`` or provide ``gzip``
on PATH. The latter handles this format despite its name. Only the two named
archive members are read, so no archive paths are extracted to the filesystem.

The arrays retain separate strokes within each digit. No pen-up gap is treated
as a one-step transition. The COMMENT writer field interpretation is explicitly
recorded as an inference, supported by the source's documented 30/14 writer
split and the observed field counts. This loader makes no independence claim.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/81/"
    "pen+based+recognition+of+handwritten+digits.zip"
)
SOURCE_PAGE = (
    "https://archive.ics.uci.edu/dataset/81/"
    "pen+based+recognition+of+handwritten+digits"
)
ARCHIVE_SHA256 = "1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64"
SCHEMA_VERSION = "pendigits-original-strokes-v1"
EXPECTED_DIGITS = {"tra": 7494, "tes": 3498}
EXPECTED_WRITERS = {"tra": 30, "tes": 14}
SEGMENT_RE = re.compile(r'^\.SEGMENT\s+DIGIT\s+(\d+(?:-\d+)?)\s+\?\s+"([0-9])"\s*$')
COMMENT_RE = re.compile(r"^\.COMMENT\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
POINT_RE = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s*$")


@dataclass
class Digit:
    source_id: str
    label: int
    writer: int = -1
    writer_sample: int = -1
    strokes: list[np.ndarray] = field(default_factory=list)


def decompress_unix_compress(payload: bytes) -> str:
    """Decode .Z bytes; do not attempt to interpret them as gzip streams."""
    try:
        from unlzw3 import unlzw
    except ImportError:
        executable = shutil.which("gzip")
        if executable is None:
            raise RuntimeError(
                "UNIX-compress input requires gzip on PATH or pip install unlzw3"
            ) from None
        result = subprocess.run(
            [executable, "-dc"], input=payload, capture_output=True, check=False
        )
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        raw = result.stdout
    else:
        raw = unlzw(payload)
    return raw.decode("ascii")


def parse_unipen(text: str, split: str) -> list[Digit]:
    """Parse original digit records, preserving individual pen-down strokes."""
    if split not in EXPECTED_DIGITS:
        raise ValueError("split must be 'tra' or 'tes'")
    digits: list[Digit] = []
    current: Digit | None = None
    points: list[tuple[int, int]] | None = None

    def finish_digit() -> None:
        if current is None:
            return
        if points is not None:
            raise ValueError("A digit ended with an open PEN_DOWN stroke")
        if current.writer < 1 or not current.strokes:
            raise ValueError("Digit missing COMMENT metadata or strokes")
        digits.append(current)

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        segment = SEGMENT_RE.fullmatch(line)
        if segment:
            finish_digit()
            current = Digit(segment.group(1), int(segment.group(2)))
            continue
        comment = COMMENT_RE.fullmatch(line)
        if comment:
            if current is None:
                raise ValueError(f"COMMENT outside digit at line {line_number}")
            label, writer, sample = map(int, comment.groups())
            if label != current.label:
                raise ValueError(f"COMMENT/SEGMENT label disagreement at line {line_number}")
            if not 1 <= writer <= EXPECTED_WRITERS[split] or not 1 <= sample <= 250:
                raise ValueError(f"Unexpected COMMENT metadata at line {line_number}")
            current.writer, current.writer_sample = writer, sample
            continue
        if line == ".PEN_DOWN":
            if current is None or points is not None:
                raise ValueError(f"Unexpected PEN_DOWN at line {line_number}")
            points = []
            continue
        if line == ".PEN_UP":
            if current is None or points is None or not points:
                raise ValueError(f"Unexpected or empty PEN_UP at line {line_number}")
            current.strokes.append(np.asarray(points, dtype=np.int16))
            points = None
            continue
        point = POINT_RE.fullmatch(line)
        if point:
            if points is None:
                raise ValueError(f"Point outside a pen-down stroke at line {line_number}")
            xy = tuple(map(int, point.groups()))
            if not all(0 <= value <= 500 for value in xy):
                raise ValueError(
                    f"Coordinate outside declared acquisition range [0,500]: {xy}"
                )
            points.append(xy)
            continue
        if line.startswith((".INCLUDE", ".LEXICON", ".HIERARCHY", ".DT")):
            if line.startswith(".DT") and line.split()[1:] != ["100"]:
                raise ValueError(f"Unexpected sampling interval at line {line_number}")
            continue
        raise ValueError(f"Unrecognized UNIPEN line {line_number}: {line[:80]}")
    finish_digit()
    if len(digits) != EXPECTED_DIGITS[split]:
        raise ValueError(f"Expected {EXPECTED_DIGITS[split]} digits, got {len(digits)}")
    if len({digit.source_id for digit in digits}) != len(digits):
        raise ValueError("Duplicate source digit IDs")
    if set(digit.writer for digit in digits) != set(range(1, EXPECTED_WRITERS[split] + 1)):
        raise ValueError("Writer-like metadata does not match the documented split")
    return digits


def pack_digits(digits: list[Digit], prefix: str) -> dict[str, np.ndarray]:
    """Use flat points plus offsets; loading requires no object pickle."""
    strokes = [stroke for digit in digits for stroke in digit.strokes]
    lengths = np.asarray([len(stroke) for stroke in strokes], dtype=np.int64)
    digit_stroke_counts = np.asarray([len(digit.strokes) for digit in digits], dtype=np.int64)
    arrays = {
        "points": np.concatenate(strokes, axis=0),
        "stroke_offsets": np.r_[0, np.cumsum(lengths)],
        "digit_stroke_offsets": np.r_[0, np.cumsum(digit_stroke_counts)],
        "digit_label": np.asarray([digit.label for digit in digits], dtype=np.int8),
        "digit_writer": np.asarray([digit.writer for digit in digits], dtype=np.int16),
        "digit_writer_sample": np.asarray([digit.writer_sample for digit in digits], dtype=np.int16),
        "digit_source_id": np.asarray([digit.source_id for digit in digits]),
    }
    return {f"{prefix}_{key}": value for key, value in arrays.items()}


def summarise(digits: list[Digit]) -> dict:
    lengths = np.asarray([len(stroke) for digit in digits for stroke in digit.strokes])
    first = np.asarray([len(digit.strokes[0]) for digit in digits])
    all_points = np.concatenate([stroke for digit in digits for stroke in digit.strokes])
    quantiles = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
    return {
        "digits": len(digits),
        "strokes": len(lengths),
        "writer_like_groups": len({digit.writer for digit in digits}),
        "writer_counts": {
            str(writer): sum(digit.writer == writer for digit in digits)
            for writer in sorted({digit.writer for digit in digits})
        },
        "multi_stroke_digits": sum(len(digit.strokes) > 1 for digit in digits),
        "coordinate_min": int(all_points.min()),
        "coordinate_max": int(all_points.max()),
        "length_quantile_probabilities": quantiles,
        "all_stroke_length_quantiles": np.quantile(lengths, quantiles).tolist(),
        "first_stroke_length_quantiles": np.quantile(first, quantiles).tolist(),
        "first_stroke_count_at_least": {
            str(min_length): int(np.sum(first >= min_length))
            for min_length in [8, 12, 16, 20, 24, 32, 40, 64]
        },
    }


def load_prepared_strokes(path: Path, split: str, first_only: bool = True) -> list[dict]:
    """Read strokes as records with source-local writer IDs and raw points.

    The caller chooses development and evaluation writers, then applies the
    predeclared map ``raw_points / 250.0 - 1.0``. Writer IDs must be qualified by
    ``split``: tra writer 1 and tes writer 1 are different documented people.
    A record is a stroke, not an assertion that this is an independent unit.
    """
    if split not in EXPECTED_DIGITS:
        raise ValueError("split must be 'tra' or 'tes'")
    with np.load(path, allow_pickle=False) as data:
        points = data[f"{split}_points"]
        offsets = data[f"{split}_stroke_offsets"]
        digit_offsets = data[f"{split}_digit_stroke_offsets"]
        records = []
        for digit in range(len(digit_offsets) - 1):
            begin, end = map(int, digit_offsets[digit : digit + 2])
            if first_only:
                end = begin + 1
            for stroke_index in range(begin, end):
                records.append(
                    {
                        "split": split,
                        "writer": int(data[f"{split}_digit_writer"][digit]),
                        "writer_sample": int(data[f"{split}_digit_writer_sample"][digit]),
                        "digit_source_id": str(data[f"{split}_digit_source_id"][digit]),
                        "label": int(data[f"{split}_digit_label"][digit]),
                        "stroke_in_digit": stroke_index - begin,
                        "points": points[offsets[stroke_index] : offsets[stroke_index + 1]].copy(),
                    }
                )
        return records


def prepare(output_dir: Path, archive_path: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if archive_path is None:
        archive_path = output_dir / "pendigits.zip"
        if not archive_path.exists():
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
                archive_path.write_bytes(response.read())
    payload = archive_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError(
            "Archive SHA-256 mismatch; inspect an upstream change rather than "
            f"silently using different data. Expected {ARCHIVE_SHA256}, got {digest}."
        )
    packed: dict[str, np.ndarray] = {}
    summaries = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for split in EXPECTED_DIGITS:
            name = f"pendigits-orig.{split}.Z"
            digits = parse_unipen(decompress_unix_compress(archive.read(name)), split)
            packed.update(pack_digits(digits, split))
            summaries[split] = summarise(digits)
        (output_dir / "pendigits-orig.names").write_bytes(archive.read("pendigits-orig.names"))
    array_path = output_dir / "pendigits_original.npz"
    np.savez_compressed(array_path, **packed)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "doi": "10.24432/C5MG6K",
        "citation": "E. Alpaydin and F. Alimoglu (1996), Pen-Based Recognition of Handwritten Digits, UCI Machine Learning Repository",
        "license": "CC BY 4.0",
        "archive_sha256": digest,
        "output_file": array_path.name,
        "array_file_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
        "declared_coordinate_range": [0, 500],
        "declared_sampling_interval_ms": 100,
        "predeclared_coordinate_transform": "x / 250.0 - 1.0",
        "preprocessing": "Original pen-down coordinates; no resampling, alignment, or per-stroke normalization",
        "writer_field_caveat": (
            "The second COMMENT integer is inferred to be the within-file writer ID. "
            "Its counts match the documented 30 training and 14 disjoint test writers. "
            "The companion names file does not explicitly define the COMMENT fields."
        ),
        "independent_unit_caveat": (
            "These arrays do not establish independence. Do not count overlapping "
            "windows, strokes, or multiple digits from the same writer as independent "
            "writers. Use writer-level aggregates for any writer-population certificate "
            "and state the iid-writer assumption explicitly."
        ),
        "summaries": summaries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a locally downloaded pinned source ZIP")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data_cache" / "pendigits",
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, args.archive), indent=2))


if __name__ == "__main__":
    main()
