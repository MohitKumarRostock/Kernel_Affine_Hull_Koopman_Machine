#!/usr/bin/env python3
"""Verify the coordinated document revision without running scientific code.

Uses only the standard library and read-only Git commands.  The receipt is
deterministic for the same input bytes and resolved baseline commit; it omits
wall-clock timestamps.  Re-running with --overwrite replaces only the chosen
receipt, never an input.  Run after manuscript and generated-asset edits finish.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
MANUSCRIPT = Path("extensions/tmlr_integration/manuscript")
REVISION = Path("extensions/tmlr_integration/revision_evidence/coordinated_20261101")
ORIGINAL_LABELS = (
    "prop:mechanism-krr", "prop:mechanism-redundancy",
    "prop:mechanism-distinct", "prop:mechanism-correction-bound",
)
THEORY_FILES = ("core_theory.tex", "core_decoder.tex", "focused_appendix_math.tex")
FOLLOWUPS = {
    "preprocessing_dimension": "preprocessing_dimension",
    "crossfit_residual": "crossfit_residual_reproduction_20261101",
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
INPUT = re.compile(
    r"\\(input|include)(?![A-Za-z])\s*"
    r"(?:\{([^{}]+)\}|([^\s%{}]+))"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def beneath(base, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts,
            f"Nonlocal path in input: {relative}")
    result = (base / path).resolve()
    require(result.is_relative_to(base.resolve()), f"Path escapes input root: {relative}")
    return result


def hash_map(base, expected):
    require(isinstance(expected, dict) and bool(expected), "Expected a nonempty hash map")
    checked, failures = {}, []
    for relative, digest in sorted(expected.items()):
        require(isinstance(digest, str) and SHA256.fullmatch(digest),
                f"Invalid SHA-256 for {relative}")
        path = beneath(base, relative)
        actual = sha(path) if path.is_file() else None
        checked[relative] = {"expected_sha256": digest, "actual_sha256": actual}
        if actual != digest:
            failures.append(relative)
    return {"status": "failed" if failures else "passed", "file_count": len(checked),
            "matched_count": len(checked) - len(failures),
            "mismatched_or_missing": failures, "files": checked}


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(result.returncode == 0,
            f"Read-only Git command failed: {result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout


def environments(data, name):
    token = name.encode("ascii")
    return re.findall(rb"\\begin\{" + token + rb"\}.*?\\end\{" + token + rb"\}",
                      data, flags=re.DOTALL)


def original_blocks(root, snapshot, baseline):
    original, current, source_records = {}, {}, {}
    for filename in THEORY_FILES:
        relative = (MANUSCRIPT / filename).as_posix()
        before = git(root, "show", f"{baseline}:{relative}")
        after = (root / relative).read_bytes()
        expected = snapshot["original_document_files"][relative]
        require(sha_bytes(before) == expected,
                f"Historical Git file does not match revision snapshot: {relative}")
        source_records[relative] = {"baseline_sha256": sha_bytes(before),
                                    "current_sha256": sha_bytes(after)}
        for target, data in ((original, before), (current, after)):
            for environment in ("proposition", "proof"):
                for block in environments(data, environment):
                    pattern = rb"\\label\{(prop:[^}]+)\}" if environment == "proposition" else rb"\\ref\{(prop:[^}]+)\}"
                    labels = re.findall(pattern, block)
                    if not labels:
                        continue
                    label = labels[0].decode("ascii")
                    if label not in ORIGINAL_LABELS:
                        continue
                    key = f"{environment}:{label}"
                    require(key not in target, f"Duplicate original environment: {key}")
                    target[key] = (relative, block)
    expected_keys = {f"{kind}:{label}" for kind in ("proposition", "proof")
                     for label in ORIGINAL_LABELS}
    require(set(original) == expected_keys, "Historical baseline does not contain exactly four original propositions and proofs")
    require(set(current) == expected_keys, "Current source is missing an original proposition or proof")
    records, failures = {}, []
    for key in sorted(expected_keys):
        before_path, before = original[key]
        after_path, after = current[key]
        unchanged = before == after and before_path == after_path
        records[key] = {"baseline_file": before_path, "current_file": after_path,
                        "baseline_sha256": sha_bytes(before), "current_sha256": sha_bytes(after),
                        "byte_identical": unchanged}
        if not unchanged:
            failures.append(key)
    return {"status": "failed" if failures else "passed", "propositions": 4, "proofs": 4,
            "comparison": "Exact bytes from begin through end of each original environment; new surrounding exposition is allowed",
            "changed_blocks": failures, "blocks": records, "source_files": source_records}


def csv_rows(path):
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def followup_assets(root, namespace, study_name, snapshot):
    assets = root / MANUSCRIPT / "generated" / namespace
    study = root / "extensions" / study_name
    manifest_path = assets / "asset_manifest.json"
    manifest = load_json(manifest_path)
    require(manifest["schema"] == "followup_manuscript_assets_v1", "Unexpected asset-manifest schema")
    require(manifest["study_directory"] == study_name, "Incorrect source-study directory")
    inputs = hash_map(study, manifest["inputs"])
    outputs = hash_map(assets, manifest["outputs"])
    require(inputs["status"] == "passed", f"Asset inputs changed: {inputs['mismatched_or_missing']}")
    require(outputs["status"] == "passed", f"Asset outputs changed: {outputs['mismatched_or_missing']}")
    actual_members = {p.relative_to(assets).as_posix() for p in assets.rglob("*") if p.is_file()}
    require(actual_members == set(manifest["outputs"]) | {"asset_manifest.json"},
            "Generated namespace has missing or unmanifested files")
    require(manifest["builder_sha256"] == sha(root / MANUSCRIPT / "build_followup_assets.py"),
            "Asset-builder hash does not match manifest")
    require(manifest["source_freeze_sha256"] == manifest["inputs"]["freeze.json"] == sha(study / "freeze.json"),
            "Source-freeze hash does not match manifest")
    bound_inputs = []
    for relative, digest in manifest["inputs"].items():
        key = (Path("extensions") / study_name / relative).as_posix()
        if key in snapshot["scientific_evidence_files"]:
            require(digest == snapshot["scientific_evidence_files"][key],
                    f"Asset input disagrees with pre-revision snapshot: {key}")
            bound_inputs.append(key)
    for cell in manifest["numeric_cells"]:
        value, places = float(cell["value"]), cell["decimal_places"]
        require(math.isfinite(value) and type(places) is int and 0 <= places <= 15,
                "Invalid numeric-cell format")
        require(cell["display"] == f"{value:.{places}f}", f"Incorrect decimal formatting: {cell['key']}")
    metrics = load_json(assets / "metrics.json")
    require(metrics["schema"] == "followup_manuscript_metrics_v1" and metrics["study_directory"] == study_name,
            "Incorrect metrics identity")
    definitions = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}",
                             (assets / "numbers.tex").read_text(encoding="utf-8"))
    require(len(definitions) == len(dict(definitions)), "Duplicate generated macro")
    require(set(dict(definitions)) == set(metrics["macros"]), "Macro definitions and metrics differ")
    for name, meta in metrics["macros"].items():
        require(dict(definitions)[name] == meta["display"], f"Macro display mismatch: {name}")
        if meta["rounding"] == "integer":
            require(float(meta["value"]) == int(meta["value"]), f"Noninteger macro: {name}")
            expected = f"{int(meta['value']):,}"
        else:
            match = re.fullmatch(r"(\d+) decimal places", meta["rounding"])
            require(match is not None, f"Unsupported macro rounding: {name}")
            expected = f"{float(meta['value']):.{int(match[1])}f}"
        require(meta["display"] == expected, f"Incorrect macro formatting: {name}")
    primary = csv_rows(study / "analysis/primary_contrasts.csv")
    require(metrics["primary_contrasts"] == primary, "Metrics do not retain exact original primary CSV rows")
    summary = load_json(study / "analysis/analysis_summary.json")
    for key in ("primary_family_size", "primary_comparisons_established"):
        require(metrics[key] == summary[key], f"Metrics differ from analysis summary: {key}")
    require(len(primary) == metrics["primary_family_size"] == 4, "Follow-up primary family is not four")
    if namespace == "crossfit_residual":
        provenance = metrics["computational_reproduction_provenance"]
        audit = load_json(study / "verification/reproduction_audit.json")
        require(provenance == audit["computational_reproduction_provenance"], "Reproduction disclosures differ from retained audit")
        require(provenance["additional_independent_repetitions"] == 0
                and provenance["historical_outputs_known_before_reproduction"] is True
                and provenance["original_completed_run_authenticated"] is False,
                "Missing same-cohort or historical-record limitation")
        require(metrics["macros"]["CFRAdditionalRepetitions"]["value"] == 0,
                "Additional-independent-repetitions macro is not zero")
    return {"manifest_sha256": sha(manifest_path), "inputs": inputs, "outputs": outputs,
            "snapshot_bound_input_count": len(bound_inputs), "builder_sha256": manifest["builder_sha256"],
            "numeric_format_records_checked": len(manifest["numeric_cells"]),
            "macro_definitions_checked": len(definitions), "primary_csv_rows_retained": len(primary),
            "limitation": "Checks hash bindings, exact copied primary rows and numeric/macro formatting; not an independent recomputation of every displayed estimate"}


def uncomment(text):
    lines = []
    for line in text.splitlines():
        for index, char in enumerate(line):
            if char == "%":
                preceding = len(line[:index]) - len(line[:index].rstrip("\\"))
                if preceding % 2 == 0:
                    line = line[:index]
                    break
        lines.append(line)
    return "\n".join(lines)


def latex_inputs(root, snapshot):
    manuscript = root / MANUSCRIPT
    queue, visited, edges, failures = ["main.tex", "supplement.tex"], {}, [], []
    while queue:
        relative = queue.pop(0)
        if relative in visited:
            continue
        path = beneath(manuscript, relative)
        if not path.is_file():
            failures.append({"source": relative, "problem": "missing input file"})
            continue
        body = path.read_bytes()
        visited[relative] = sha_bytes(body)
        for match in INPUT.finditer(uncomment(body.decode("utf-8"))):
            literal = (match[2] or match[3]).strip()
            if any(char in literal for char in ("\\", "#", "$")):
                failures.append({"source": relative, "target": literal, "problem": "dynamic input cannot be checked statically"})
                continue
            target = literal if Path(literal).suffix else literal + ".tex"
            try:
                resolved = beneath(manuscript, target)
            except ValueError as error:
                failures.append({"source": relative, "target": literal, "problem": str(error)})
                continue
            exists = resolved.is_file()
            edges.append({"source": relative, "command": match[1], "literal": literal,
                          "resolved": target, "exists": exists,
                          "new_file_relative_to_snapshot": (MANUSCRIPT / target).as_posix() not in snapshot["original_document_files"]})
            if exists:
                queue.append(target)
            else:
                failures.append({"source": relative, "target": target, "problem": "missing input file"})
    require("core_followup_experiments.tex" in visited, "Follow-up main-paper section is not included")
    for namespace in FOLLOWUPS:
        require(f"generated/{namespace}/numbers.tex" in visited, f"Missing follow-up macro input: {namespace}")
    return {"status": "failed" if failures else "passed", "entry_points": ["main.tex", "supplement.tex"],
            "file_count": len(visited), "input_occurrences": len(edges),
            "new_file_input_occurrences": sum(e["new_file_relative_to_snapshot"] for e in edges),
            "failures": failures, "source_sha256": dict(sorted(visited.items())), "inputs": edges,
            "scope": "Literal input/include paths reachable from both entry points, resolving from the manuscript working directory after comment removal; no TeX execution or macro/reference/graphics validation"}


def comparison_families(root):
    records = {}
    for name, expected_family, expected_established in (
        ("dimension_forecasting", 6, 3), ("preprocessing_dimension", 4, 1),
        ("crossfit_residual_reproduction_20261101", 4, 2),
    ):
        relative = Path("extensions") / name / "analysis/primary_contrasts.csv"
        rows = csv_rows(root / relative)
        decisions = [r["criterion_established"].lower() for r in rows]
        require(all(x in ("true", "false") for x in decisions), f"Unexpected primary decision encoding: {name}")
        established = decisions.count("true")
        require((len(rows), established) == (expected_family, expected_established),
                f"Original primary family or retained decisions differ: {name}")
        records[name] = {"file": relative.as_posix(), "sha256": sha(root / relative),
                         "primary_family_size": len(rows), "established": established}
    return {"studies": records, "scope": "Counts unchanged recorded primary decisions; no hypothesis tests recomputed or families pooled"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=HERE.parents[3])
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--baseline-ref", default="f36e7c797f37d7d073dae42603b3770543d9fb43",
                        help="Pinned historical Git reference used only for original proposition/proof bytes")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing verification receipt only")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    snapshot_path = (args.snapshot or root / REVISION / "input_snapshot.json").resolve()
    output = (args.output or root / REVISION / "revision_verification.json").resolve()
    snapshot = load_json(snapshot_path)
    require(snapshot["schema"] == "kahm-coordinated-manuscript-inputs-v1", "Unexpected input-snapshot schema")
    protected = {snapshot_path, Path(__file__).resolve()}
    protected.update((root / p).resolve() for key in ("scientific_evidence_files", "original_document_files") for p in snapshot[key])
    require(output not in protected and output.suffix == ".json", "Receipt destination would replace an input or is not JSON")
    require(args.overwrite or not output.exists(), "Receipt exists; choose a new --output or explicitly use --overwrite")
    baseline = git(root, "rev-parse", "--verify", args.baseline_ref + "^{commit}").decode("ascii").strip()
    checks = {}

    def run(name, operation):
        try:
            result = operation()
            result.setdefault("status", "passed")
            checks[name] = result
        except (ValueError, KeyError, TypeError, OSError, UnicodeError) as error:
            checks[name] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}

    def preserved():
        require(len(snapshot["scientific_evidence_files"]) == 144, "Expected exactly 144 scientific evidence files")
        return hash_map(root, snapshot["scientific_evidence_files"])

    run("scientific_evidence_preservation", preserved)
    run("original_propositions_and_proofs", lambda: original_blocks(root, snapshot, baseline))
    for namespace, study in FOLLOWUPS.items():
        run("assets_" + namespace, lambda n=namespace, s=study: followup_assets(root, n, s, snapshot))
    run("latex_input_paths", lambda: latex_inputs(root, snapshot))
    run("preserved_primary_families", lambda: comparison_families(root))
    passed = all(check["status"] == "passed" for check in checks.values())
    receipt = {
        "schema": "kahm-coordinated-revision-verification-v1", "status": "passed" if passed else "failed",
        "scope": "Document/source preservation, display provenance and static LaTeX input verification only",
        "snapshot_sha256": sha(snapshot_path), "snapshot_base_commit": snapshot["base_commit"],
        "original_block_baseline_commit": baseline, "baseline_requested": args.baseline_ref,
        "verifier_sha256": sha(Path(__file__).resolve()), "checks": checks,
        "not_checked": [
            "Scientific fitting, data generation, forecast scoring or production-audit reruns",
            "Recomputation of statistical inference or complete independent numerical-display arithmetic",
            "Mathematical validity of new exposition or semantic correctness of every narrative claim",
            "LaTeX compilation, undefined macros/references, graphic resolution or PDF visual quality",
            "Archive delivery, remote publication or authentication of unavailable original cross-fit outputs",
        ],
        "determinism": "No timestamp or transient absolute workspace root is included; identical inputs and baseline yield identical receipt bytes",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    require(not temporary.exists(), f"Staged receipt already exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(output)
    print(json.dumps({"status": receipt["status"], "output": str(output),
                      "sections": {name: result["status"] for name, result in checks.items()}}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError, UnicodeError) as error:
        print(f"verify_revision: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
