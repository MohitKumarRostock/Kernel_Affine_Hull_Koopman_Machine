#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
REPRO = ROOT / "REPRODUCIBILITY.md"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    readme = README.read_text(encoding="utf-8")
    repro = REPRO.read_text(encoding="utf-8")

    readme_anchor = "## Repository organization\n"

    readme_section = """## Reproducibility evidence

For a reviewer-facing summary of the current evidence state, see:

- `reproduction/REPRODUCIBILITY_EVIDENCE.md` — consolidated evidence summary and claim boundaries.
- `reproduction/reproducibility_evidence.json` — machine-readable form of the same summary.
- `reproduction/raw_data/DATASET_CARD.md` — canonical raw scientific input package.
- `reproduction/run_accounting/README.md` — historical run-accounting audit.
- `reproduction/prospective_campaigns/duffing_reference_27/README.md` — prospective 27-job Duffing reference rerun.

The historical audit and prospective gap-closing campaign are intentionally
kept separate. The historical run-accounting audit preserves the original
Duffing reference-search gap, while the prospective campaign records all 27
scheduled attempts and their exit statuses from a frozen clean commit.

The repository does not claim that every failed attempt from every historical
campaign was preserved when the available historical evidence does not support
that stronger statement.

"""

    if "## Reproducibility evidence\n" not in readme:
        readme = replace_once(
            readme,
            readme_anchor,
            readme_section + readme_anchor,
            "README reproducibility evidence insertion",
        )

    old_principle_4 = (
        "4. Every scheduled run is recorded, including failed runs.\n"
    )
    new_principle_4 = (
        "4. Prospective reproducibility campaigns use failure-preserving run "
        "ledgers that record every scheduled attempt, including failures.\n"
    )
    repro = replace_once(
        repro,
        old_principle_4,
        new_principle_4,
        "REPRODUCIBILITY principle 4",
    )

    old_principle_5 = (
        "5. Raw unaggregated experimental outputs are retained.\n"
    )
    new_principle_5 = (
        "5. Raw unaggregated experimental outputs used as reproducibility "
        "evidence are retained where available, with historical gaps reported "
        "explicitly rather than inferred away.\n"
    )
    repro = replace_once(
        repro,
        old_principle_5,
        new_principle_5,
        "REPRODUCIBILITY principle 5",
    )

    old_principle_10 = (
        "10. Release artifacts are protected by cryptographic checksums and "
        "immutable archival releases.\n"
    )
    new_principle_10 = (
        "10. Release artifacts are protected by cryptographic checksums; the "
        "final immutable archival release is created only at the release stage.\n"
    )
    repro = replace_once(
        repro,
        old_principle_10,
        new_principle_10,
        "REPRODUCIBILITY principle 10",
    )

    experimental_anchor = (
        "## Historical environments\n"
    )

    evidence_section = """## Evidence map

The current reviewer-facing evidence summary is:

`reproduction/REPRODUCIBILITY_EVIDENCE.md`

Its machine-readable counterpart is:

`reproduction/reproducibility_evidence.json`

The principal evidence layers are:

- `reproduction/raw_data/DATASET_CARD.md` for canonical raw scientific inputs;
- `reproduction/run_accounting/README.md` for the historical run-accounting audit;
- `reproduction/prospective_campaigns/duffing_reference_27/README.md` for the prospective Duffing reference campaign;
- `reproduction/reported_results.tsv` for result-level provenance and retention status.

The historical and prospective evidence are deliberately distinguished. The
historical run-accounting audit preserves a documented gap in the original
Duffing reference-search archive. A separately labeled prospective campaign
then reruns the complete 27-job grid from a frozen clean commit and records every
attempt, exit status, log, and output checksum.

This supports a strong failure-preserving claim for that prospective campaign.
It does not establish that every failed execution attempt from every historical
campaign was preserved.

"""

    if "## Evidence map\n" not in repro:
        repro = replace_once(
            repro,
            experimental_anchor,
            evidence_section + experimental_anchor,
            "REPRODUCIBILITY evidence-map insertion",
        )

    README.write_text(readme, encoding="utf-8")
    REPRO.write_text(repro, encoding="utf-8")

    print("PASS: README reviewer navigation added")
    print("PASS: REPRODUCIBILITY evidence map added")
    print("PASS: historical no-cherry-picking overclaim removed")
    print("PASS: immutable-release wording aligned with current release state")


if __name__ == "__main__":
    main()
