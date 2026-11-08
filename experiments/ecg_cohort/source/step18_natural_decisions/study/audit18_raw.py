"""Independent raw-byte checks of already staged Step 18 observations.

This audit uses struct.unpack_from and Decimal arithmetic. It deliberately
does not call the production header parser, decoder, extractor, or stage
loader. It may only inspect records from a stage that already has a receipt.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import re
import struct
import zipfile

import numpy as np

import data18 as D


def audit_staged_records(stage, indices=(0, 1, -1), *, output_root=D.DEFAULT_OUTPUT_ROOT,
                         archive_path=D.DEFAULT_ARCHIVE_PATH):
    stage = {"val": "validation", "cal": "calibration", "test": "terminal"}.get(stage, stage)
    if stage not in ("support", "train", "validation", "calibration", "terminal"):
        raise ValueError("Audit is restricted to already staged, non-reserve records")
    stage_dir = Path(output_root) / stage
    receipt = json.loads((stage_dir / "receipt.json").read_text())
    if (stage_dir / "FAILED.json").exists():
        raise AssertionError("Cannot audit a failed extraction as a successful stage")
    if D.verify_freeze(receipt["freeze"]["path"]) != receipt["freeze"]:
        raise AssertionError("Frozen source closure changed")
    acquisition = json.loads(D.DEFAULT_RECEIPT_PATH.read_text())
    if D.sha256_file(Path(archive_path)) != acquisition["sha256"]:
        raise AssertionError("Raw archive hash mismatch")
    artifact = stage_dir / "arrays.npz"
    if D.sha256_file(artifact) != receipt["artifact_sha256"]:
        raise AssertionError("Staged array hash mismatch")
    with np.load(artifact, allow_pickle=False) as saved:
        staged = {key: saved[key].copy() for key in saved.files}
    if ("Y" in staged) != (stage != "support"):
        raise AssertionError("Incorrect support target policy")
    with D.ROSTER_PATH.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["stage"] == stage]
    if any(not -len(rows) <= int(index) < len(rows) for index in indices):
        raise ValueError("An audit row index is out of range")
    selected_indices = sorted({int(index) % len(rows) for index in indices})
    if not selected_indices:
        raise ValueError("At least one audit row is required")
    checksum_lines = (D.REFERENCE_DIR / "SHA256SUMS.txt").read_text().splitlines()
    checksums = {line.split()[1].lstrip("*"): line.split()[0] for line in checksum_lines}
    wanted_patients = {int(rows[index]["patient_id"]) for index in selected_indices}
    minima = {}
    with (D.REFERENCE_DIR / "ptbxl_database.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            patient, ecg = int(Decimal(row["patient_id"])), int(Decimal(row["ecg_id"]))
            if patient in wanted_patients and (patient not in minima or ecg < minima[patient][0]):
                minima[patient] = (ecg, row["filename_hr"])
    outcomes = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        metadata = [name for name in names if name == "ptbxl_database.csv" or name.endswith("/ptbxl_database.csv")]
        if len(metadata) != 1:
            raise AssertionError("Ambiguous ZIP prefix")
        prefix = metadata[0][:-len("ptbxl_database.csv")]
        for index in selected_indices:
            row = rows[index]
            patient, ecg, origin = int(row["patient_id"]), int(row["ecg_id"]), int(row["origin"])
            expected_origin = 155 + int(hashlib.sha256(("step18-origin-v1:" + str(patient)).encode()).hexdigest(), 16) % 4825
            if origin != expected_origin or minima[patient] != (ecg, row["filename_hr"]):
                raise AssertionError("Metadata-only record/origin selection differs")
            header_path, signal_path = row["filename_hr"] + ".hea", row["filename_hr"] + ".dat"
            header_raw, signal_raw = archive.read(prefix + header_path), archive.read(prefix + signal_path)
            for name, blob in ((header_path, header_raw), (signal_path, signal_raw)):
                if hashlib.sha256(blob).hexdigest() != checksums[name]:
                    raise AssertionError("Independent official per-record hash mismatch")
            lines = [line.strip() for line in header_raw.decode("utf-8").splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
            first = lines[0].split()
            nsignals, frequency, nsamples = int(first[1]), Decimal(first[2].split("/")[0]), int(first[3])
            if (nsignals, frequency, nsamples) != (12, Decimal(500), 5000):
                raise AssertionError("Unexpected raw dimensions")
            signal_lines = [line.split(maxsplit=8) for line in lines[1:]]
            lead_candidates = [j for j, fields in enumerate(signal_lines) if fields[8].upper() == "II"]
            if len(lead_candidates) != 1:
                raise AssertionError("Lead II is not unique")
            lead = lead_candidates[0]
            fields = signal_lines[lead]
            # Independent field parsing uses string splits, not production regexes.
            format_before_offset, separator, offset_text = fields[1].partition("+")
            offset = int(offset_text) if separator else 0
            baseformat = format_before_offset.split(":")[0].split("x")[0]
            if baseformat != "16":
                raise AssertionError("Independent audit only supports format16")
            numeric_gain, units = fields[2].split("/", 1)
            if "(" in numeric_gain:
                gain_text, baseline_text = numeric_gain.split("(", 1)
                if not baseline_text.endswith(")"):
                    raise AssertionError("Malformed explicit baseline")
                baseline = Decimal(baseline_text[:-1])
            else:
                gain_text, baseline = numeric_gain, Decimal(fields[4])
            gain = Decimal(gain_text)
            unit_factor = {"mV": Decimal(1), "V": Decimal(1000), "uV": Decimal("0.001"),
                           "µV": Decimal("0.001"), "μV": Decimal("0.001")}[units]
            if len(signal_raw) != offset + 2 * nsignals * nsamples:
                raise AssertionError("Independent raw byte count mismatch")

            def scalar_mv(sample):
                digital = struct.unpack_from("<h", signal_raw, offset + 2 * (sample * nsignals + lead))[0]
                if digital == -32768:
                    raise AssertionError("Missing required raw sample")
                with localcontext() as context:
                    context.prec = 50
                    return (Decimal(digital) - baseline) / gain * unit_factor

            # This reconstructs exactly the 32 declared positions using scalar indexing.
            independently_decoded = np.asarray([float(scalar_mv(origin - 155 + 5 * j)) for j in range(32)])
            error_x = float(np.max(np.abs(independently_decoded - staged["X"][index])))
            scale_x = max(1.0, float(np.max(np.abs(independently_decoded))))
            if error_x > 4 * np.finfo(np.float64).eps * scale_x:
                raise AssertionError("Independent scalar history decode differs")
            for key, value in (("patient_id", patient), ("ecg_id", ecg), ("origin", origin)):
                if int(staged[key][index]) != value:
                    raise AssertionError("Saved identities differ from roster")
            result = {"stage_index": index, "patient_id": patient, "ecg_id": ecg,
                      "origin": origin, "history_max_absolute_difference": error_x,
                      "record_is_metadata_minimum": True, "origin_hash_matches": True,
                      "official_member_hashes_match": True, "target_extracted": stage != "support"}
            if stage != "support":
                physical_target = scalar_mv(origin + 20)
                expected_y = float(max(Decimal(-2), min(Decimal(2), physical_target)) / Decimal(2))
                error_y = abs(expected_y - float(staged["Y"][index]))
                if error_y > 4 * np.finfo(np.float64).eps:
                    raise AssertionError("Independent scalar target decode differs")
                result["target_absolute_difference"] = error_y
            outcomes.append(result)
    return {"schema": "step18-independent-raw-audit-v1", "stage": stage,
            "method": "struct little-endian scalar indexing plus 50-digit Decimal conversion; no production parser/decoder/extractor/loader",
            "support_target_absent": "Y" not in staged if stage == "support" else None,
            "archive_sha256": acquisition["sha256"], "artifact_sha256": receipt["artifact_sha256"],
            "checked_records": outcomes, "passed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 1, -1])
    parser.add_argument("--output-root", type=Path, default=D.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    result = audit_staged_records(args.stage, args.indices, output_root=args.output_root)
    if args.save:
        with args.save.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(result, indent=2))
