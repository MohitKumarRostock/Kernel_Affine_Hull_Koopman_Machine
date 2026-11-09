"""Verify the computational distribution against its historical source freeze.

Manuscript prose is outside this distribution. Its original identifiers remain
recorded, while every distributed frozen file is checked against either its
original digest or an explicitly declared documentation or verifier digest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HISTORICAL_COMMIT = "a8d193446d3cb9bb547ee508e11fe3a1c18d2338"
HISTORICAL_FREEZE_SHA256 = "1b80460fe2d33ede50bd085c026f7cdb6219e4f54057ff01d67c0e026fbcd60e"
OMITTED_MANUSCRIPTS = {
    "runtime/step1/forecasting_corrected/provenance/core_decoder.tex":
        "49ebd37b76867f53dcf0bd7c206b9027458f5a228fa456cf56001e779bc7e1c4",
    "runtime/step1/forecasting_corrected/provenance/focused_appendix_experiment.tex":
        "201a393434a425378455fde6e34c663d43bffccb069f99a3a4ab09eaeb8edf93",
}
REPLACED_FILES = {
    "analysis_code/analyze_forecasting.py":
        "6f15dcd726750346e48383a18ba07ab2514cd76d84b7f868f15d7fecba1f4205",
    "code/study_contract.py":
        "9cc974642784052cc5b51e27eadb80150f7d1f04c63ba9feeab594d9c9803858",
    "code/study_seals.py":
        "67d4467e809d5a9df6d55eec8a4827979082a3bb5f5ada1ecc3117040b9ef9a9",
    "runtime/step1/forecasting_corrected/README.md":
        "e4d580adfdd686175cfb7df966a03c113074e65266f112af18c5e32b733694fd",
}
DISTRIBUTED_REPLACEMENT_SHA256 = {
    "analysis_code/analyze_forecasting.py":
        "8bea541033f9fa82c0b3fbd1452934cbfe9134f174dd751295228e3e97a179aa",
    "code/study_contract.py":
        "6fc4bd9cefa4c69137f9a4d6731c9fd5cf6d81433fa49c065e4e6219a4e42962",
    "code/study_seals.py":
        "28a158fe05367d6442cc7d1d2b72d30373a7a219f46e574d22c28a10df9a9665",
    "runtime/step1/forecasting_corrected/README.md":
        "b998ae058af85f13c8da197c84896d56c598dd999e29234bf54827b90c914ae3",
}
ADDED_VERIFIER = "code/source_distribution.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confined_file(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(),
            "Missing or unconfined distributed source: " + relative)
    return path


def verify_source_distribution(root, record):
    """Verify all recorded computational files and report manuscript omissions."""
    root = Path(root).resolve()
    freeze_path = confined_file(root, "freeze.json")
    require(sha(freeze_path) == HISTORICAL_FREEZE_SHA256,
            "The historical source freeze has a different identity")
    require(record == json.loads(freeze_path.read_text()),
            "The source record differs from the identified historical freeze")
    distribution = json.loads(confined_file(root, "source_distribution.json").read_text())
    require(distribution.get("schema") == "kahm-forecasting-source-distribution-v1",
            "Unsupported source distribution schema")
    require(distribution.get("historical_commit") == HISTORICAL_COMMIT
            and distribution.get("freeze_sha256") == HISTORICAL_FREEZE_SHA256,
            "The distribution does not identify the historical source freeze")
    require(distribution.get("omitted_manuscripts") == OMITTED_MANUSCRIPTS,
            "The manuscript omission list differs from the declared distribution")
    replacements = distribution.get("replacement_files", {})
    expected_replacements = {
        path: {"original_sha256": original,
               "distributed_sha256": DISTRIBUTED_REPLACEMENT_SHA256[path]}
        for path, original in REPLACED_FILES.items()
    }
    require(replacements == expected_replacements,
            "Documentation or verification identities differ from the declared distribution")
    additions = distribution.get("additional_files", {})
    require(set(additions) == {ADDED_VERIFIER},
            "The additional source inventory must identify the distribution verifier")
    files = record.get("files", {})
    require(isinstance(files, dict) and files, "Empty historical source inventory")
    for relative, expected in {**OMITTED_MANUSCRIPTS, **REPLACED_FILES}.items():
        require(files.get(relative) == expected,
                "Original source identity differs: " + relative)
    for relative in OMITTED_MANUSCRIPTS:
        path = root / relative
        require(not path.exists() and not path.is_symlink(),
                "Manuscript source is outside this distribution: " + relative)
    verified_unchanged = 0
    for relative, expected in files.items():
        if relative in OMITTED_MANUSCRIPTS:
            continue
        if relative in REPLACED_FILES:
            replacement = replacements[relative]
            require(replacement.get("original_sha256") == expected,
                    "Replacement has an incorrect original identity: " + relative)
            expected = replacement.get("distributed_sha256")
        require(sha(confined_file(root, relative)) == expected,
                "Distributed source identity mismatch: " + relative)
        verified_unchanged += int(relative not in REPLACED_FILES)
    require(sha(confined_file(root, ADDED_VERIFIER)) == additions[ADDED_VERIFIER],
            "Distribution verifier identity mismatch")
    return {
        "status": "verified_available_source",
        "historical_freeze_sha256": HISTORICAL_FREEZE_SHA256,
        "verified_unchanged_file_count": verified_unchanged,
        "verified_replacement_files": replacements,
        "verified_additional_files": additions,
        "omitted_manuscripts": OMITTED_MANUSCRIPTS,
        "omitted_manuscript_contents_verified": False,
        "complete_historical_source_verified": False,
    }
