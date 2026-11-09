#!/usr/bin/env python3
"""Check the retained study source files and their frozen dependencies.

This read-only check validates source identities, not model training, forecast
accuracy, or external raw-data archives. Run with --root to select a checkout.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


EXTENSIONS = Path("extensions")
DIMENSION = EXTENSIONS / "dimension_forecasting"
PREPROCESSING = EXTENSIONS / "preprocessing_dimension"
POLICY = EXTENSIONS / "policy_augmentation_20261101"
PREDICTIVE = EXTENSIONS / "predictive_geometry"

# These are the published source identities, not newly calculated expectations.
ANCHORS = {
    "extensions/predictive_geometry/study/SNAPSHOT.json": "1cd1abde423659b79b4e4fd9373a3c80677917ada3490df08136c415e7c1d530",
    "extensions/predictive_geometry/study/protocol_freeze.json": "e5469b58a4ebd680ace41a5aa9ff9f18d7a5a177900ef0d3ec6f8b4b9f6ddd81",
    "extensions/predictive_geometry/study/followup_freeze.json": "3a8f90ef8797c4450e67275f5a5915637f7776869060fa8ba4f353d212896357",
    "extensions/predictive_geometry/study/numerical_sensitivity_freeze.json": "9b943cc140fec76e0005105416f33b13ed289087e98c7e6dd2057665a26395dc",
    "extensions/dimension_forecasting/freeze.json": "1b80460fe2d33ede50bd085c026f7cdb6219e4f54057ff01d67c0e026fbcd60e",
    "extensions/preprocessing_dimension/freeze.json": "6102fb7b5bb4726be149287c1645b7085feaec29a635483483d2e8a9f888c172",
    "extensions/crossfit_residual_reproduction_20261101/freeze.json": "fad1172d60165631afe79be2c8e57faabbe10324d18bd661ebeb9fe68687e5bb",
    "extensions/policy_augmentation_20261101/freeze.json": "8883a946000c6ffce1869436882366fe1465816b16ddb247919742017d132751",
    "extensions/policy_augmentation_natural_20261101/freeze.json": "d9a2386a2a8869463f87839be3f945c078b7c3826de091773b546f1c933c702e",
    "extensions/soft_state_ablation_20261102/freeze.json": "2745d10525a8fbc104141b7f76f6d5ff04e7cd421acb5d35b5573af2b7adcbc2",
    "extensions/crossfit_residual_v2/freeze.json": "b118e88a1c788fac3a268682736d01c5a9219146f0db7bda1929159c059d3210",
    "extensions/score_enrichment_20261102/PROSPECTIVE_FREEZE.json": "f2d7d31103acef84c5df3fa7db2d24670ea68d1b2121d56e7e1c8340eb960aa1",
    "extensions/reconstruction_mechanism/analyze.py": "069e2c1837080b15663f5bba02a5270ae87239b9b5c0653f53148929a6f2746d",
}

# Bases follow verify_freeze in each study's pipeline/study_contract module.
STUDIES = (
    ("dimension_forecasting", {}),
    ("preprocessing_dimension", {"inherited_files": DIMENSION}),
    ("crossfit_residual_reproduction_20261101", {
        "inherited_files": DIMENSION, "previous_extension_files": PREPROCESSING}),
    ("policy_augmentation_20261101", {
        "inherited_files": DIMENSION, "previous_extension_files": PREPROCESSING}),
    ("policy_augmentation_natural_20261101", {"dependencies": EXTENSIONS}),
    ("soft_state_ablation_20261102", {
        "inherited_files": DIMENSION, "numerical_solver_files": POLICY}),
    ("crossfit_residual_v2", {
        "inherited_files": DIMENSION, "previous_extension_files": PREPROCESSING}),
)


class SourceCheck:
    def __init__(self, root):
        self.root = root.resolve()
        self.digests = {}
        self.errors = []
        self.bindings = 0

    def path(self, relative):
        relative = PurePosixPath(str(relative))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Nonlocal source path: {relative}")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Source path escapes checkout: {relative}")
        return path

    def check(self, relative, expected, *, compressed=False):
        self.bindings += 1
        key = (str(relative), compressed)
        try:
            if key not in self.digests:
                opener = gzip.open if compressed else open
                value = hashlib.sha256()
                with opener(self.path(relative), "rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        value.update(chunk)
                self.digests[key] = value.hexdigest()
            if self.digests[key] != expected:
                raise ValueError("SHA256 differs")
            return True
        except (OSError, ValueError, EOFError) as error:
            self.errors.append({"path": str(relative), "error": str(error)})
            return False

    def read(self, relative):
        return json.loads(self.path(relative).read_text(encoding="utf-8"))

    def mapping(self, base, values):
        for relative, expected in values.items():
            self.check(base / relative, expected)


def verify(root):
    check = SourceCheck(root)
    anchored = {path: check.check(path, expected) for path, expected in ANCHORS.items()}
    manifests = 0
    for study, inherited in STUDIES:
        base = EXTENSIONS / study
        name = str(base / "freeze.json")
        if not anchored[name]:
            continue
        frozen = check.read(name)
        check.mapping(base, frozen["files"])
        for field, source in inherited.items():
            check.mapping(source, frozen[field])
        if "config_sha256" in frozen:
            check.check(base / "config.json", frozen["config_sha256"])
        manifests += 1

    snapshot_name = str(PREDICTIVE / "study/SNAPSHOT.json")
    if anchored[snapshot_name]:
        snapshot = check.read(snapshot_name)["files"]
        for entry in snapshot.values():
            check.check(PREDICTIVE / entry["path"], entry["sha256"],
                        compressed=entry.get("compression") == "gzip")
        # The existing materializer supplies stage-one records at this logical
        # path because the capacity study uses them to set its fixed margins.
        for filename in ("protocol_freeze.json", "followup_freeze.json",
                         "numerical_sensitivity_freeze.json"):
            name = str(PREDICTIVE / "study" / filename)
            if not anchored[name]:
                continue
            for logical, expected in check.read(name)["sha256"].items():
                if logical == "records.json":
                    check.check(PREDICTIVE / "archive/stage1/records.json.gz",
                                expected, compressed=True)
                else:
                    entry = snapshot[logical]
                    check.check(PREDICTIVE / entry["path"], expected,
                                compressed=entry.get("compression") == "gzip")
            manifests += 1

    errors = list({(e["path"], e["error"]): e for e in check.errors}.values())
    return {
        "status": "FAIL" if errors else "PASS",
        "scope": "Frozen source files and current cross-directory dependencies only",
        "root": str(check.root),
        "source_freezes_checked": manifests,
        "fixed_identity_checks": len(ANCHORS),
        "hash_bindings_checked": check.bindings,
        "distinct_file_encodings_read": len(check.digests),
        "error_count": len(errors),
        "errors": errors,
        "not_checked": [
            "External M4 raw CSVs listed in natural input_files",
            "Per-array numerical-sensitivity input_sha256 records",
            "The 50 original score-enrichment sources: only their available source-identity record is checked",
            "Model fitting, forecast accuracy, output audits and timing measurements",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository checkout containing the retained extensions")
    args = parser.parse_args()
    try:
        report = verify(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {"status": "FAIL", "error": str(error)}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
