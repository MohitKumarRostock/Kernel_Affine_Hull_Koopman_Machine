"""Read-only independent audit of four-state pilot-v1 numerical evidence.

Uses standard-library rational sums and 80-digit Decimal arithmetic; imports
none of the experiment's calculation or sampling modules. Recomputes from
saved counts without drawing samples. The configuration bytes and inventory
are checked first. Verifies all scheduled results, including any legitimate
zero certificates or coverage failures; neither is treated as a code failure.

Numerical closeness uses RTOL/ATOL below, distinct from comparison_atol in the
campaign. Flags are checked exactly against the saved binary64 quantities so
roundoff at comparison boundaries is not silently reclassified. This is an
implementation/evidence audit, not a proof of the theorem, empirical
independence, or an independent rerun of the random sampling process.

Does not modify evidence or its original campaign_summary.json. The CLI
requires this verifier to be committed in a clean source tree and prints a
JSON verification report; reporting/archiving is a separate operation.
"""

import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

VERIFIER_ID = "four_state_pilot_independent_numerics_v1"
CONFIG_SHA256 = "46a4f04448e06be57cfd2ed191856403eba119699dabee8c584eae4a7991977c"
RTOL, ATOL = 5e-12, 5e-14
PRECISION = 80


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON literal: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def document(folder, name):
    return read_json((folder / name).read_text(encoding="utf-8"))


def records(folder, name):
    with (folder / name).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"Blank record in {name}:{number}")
            yield read_json(line)


def index_records(folder, name):
    result = {}
    for row in records(folder, name):
        key = row["attempt_id"]
        require(key not in result, f"Duplicate attempt in {name}: {key}")
        result[key] = row
    return result


def check_inventory(folder):
    manifest = folder / "SHA256SUMS"
    require(manifest.is_file() and not manifest.is_symlink(), "Missing/unsafe SHA256SUMS")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2, "Malformed checksum entry")
        checksum, name = fields
        path = PurePosixPath(name)
        require(re.fullmatch(r"[0-9a-f]{64}", checksum) is not None, "Malformed checksum")
        require(name and not path.is_absolute() and ".." not in path.parts
                and path.as_posix() == name and name != "SHA256SUMS", "Unsafe inventory path")
        require(name not in expected, "Duplicate inventory entry")
        expected[name] = checksum
    actual = set()
    for path in folder.rglob("*"):
        require(not path.is_symlink(), f"Evidence symlink: {path}")
        if path.is_file() and path != manifest:
            actual.add(path.relative_to(folder).as_posix())
    require(actual == set(expected), "Missing or unlisted evidence files")
    for name, checksum in expected.items():
        require(digest(folder / name) == checksum, f"Checksum mismatch: {name}")
    return digest(manifest), len(expected)


class Comparison:
    def __init__(self):
        self.max_errors = {}
        self.numbers_checked = 0
        self.context = ""

    def close(self, actual, expected, group):
        if isinstance(expected, (tuple, list)):
            require(isinstance(actual, (tuple, list)) and len(actual) == len(expected),
                    f"Shape mismatch {self.context}: {group}")
            for a, e in zip(actual, expected):
                self.close(a, e, group)
            return
        require(type(actual) in (int, float) and math.isfinite(actual),
                f"Invalid numeric value {self.context}: {group}")
        expected = float(expected)
        error = abs(actual - expected)
        self.max_errors[group] = max(self.max_errors.get(group, 0.0), error)
        self.numbers_checked += 1
        require(math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL),
                f"Numerical mismatch {self.context}/{group}: "
                f"saved={actual!r}, recomputed={expected!r}")


def fraction(value):
    return Fraction.from_float(float(value))


def dec(value):
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal.from_float(float(value))


def weighted_statistics(p, dynamics, counts):
    """Exact weighted residual sums, not the compact n_a*n_b formula."""
    p = fraction(p)
    q = 1 - p
    current = ((p, q), (p, q), (q, p), (q, p))
    future = current if dynamics == "identity" else (
        current[0], current[2], current[2], current[2]
    )
    labels = (0, 0, 1, 1)
    total = sum(counts)
    means, sses, sizes = [], [], []
    for c in range(2):
        indices = [i for i in range(4) if labels[i] == c]
        size = sum(counts[i] for i in indices)
        mean = tuple(
            sum(counts[i] * future[i][j] for i in indices) / size
            if size else Fraction(1, 2)
            for j in range(2)
        )
        sse = sum(
            (counts[i] * (future[i][j] - mean[j])**2
             for i in indices for j in range(2)),
            Fraction(0),
        )
        sizes.append(size)
        means.append(mean)
        sses.append(sse)
    f = sum(counts[i] * (1 - current[i][labels[i]])**2 for i in range(4)) / total
    return dict(f=f, s=sum(sses) / total, sizes=sizes, means=means, sses=sses,
                current=current, future=future)


def independent_certificate(f, s, n, kappa, delta):
    with localcontext() as ctx:
        ctx.prec = PRECISION
        zero, one, two = Decimal(0), Decimal(1), Decimal(2)
        f, s, k, d = dec(f), dec(s), dec(kappa), dec(delta)
        r = (two / d).ln() / Decimal(n)
        # Rearranged positive root of F - f = sqrt(2*r*F).
        F = min(one, ((r / two).sqrt() + (f + r / two).sqrt())**2)
        V = max(zero, s - (two * r).sqrt())
        L = max(zero, V.sqrt() - k * (two * F).sqrt())
        return dict(zip(
            ("r_delta", "F_delta", "V_delta", "L_kappa_delta"),
            map(float, (r, F, V, L)),
        ))


def verify_reference(saved, case, compare):
    population = weighted_statistics(case["p"], case["dynamics"], (1, 1, 1, 1))
    for name in ("p", "kappa", "dynamics"):
        require(saved[name] == case[name], f"Reference/configuration mismatch: {name}")
    for name, expected in dict(
        reference_id="four_state_uniform_reference_v1",
        state_names=["a", "b", "d", "e"],
        states=[[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]],
        probabilities=[.25]*4,
        labels=[0, 0, 1, 1],
        successor_indices=[0, 1, 2, 3] if case["dynamics"] == "identity" else [0, 2, 2, 2],
    ).items():
        require(saved[name] == expected, f"Reference mismatch: {name}")
    compare.close(saved["phi"], population["current"], "reference_coordinates")
    compare.close(saved["successors"], population["future"], "reference_coordinates")

    B = saved["optimal_B"]
    require(isinstance(B, list) and len(B) == 2 and all(len(row) == 2 for row in B),
            "Prediction matrix must be 2 by 2")
    compare.close(B, B, "reference_matrix")
    Bq = [[fraction(x) for x in row] for row in B]
    for i, row in enumerate(population["current"]):
        prediction = [sum(row[h] * Bq[h][j] for h in range(2)) for j in range(2)]
        compare.close([float(x) for x in prediction], population["means"][i // 2],
                      "conditional_mean_attainment")

    with localcontext() as ctx:
        ctx.prec = PRECISION
        A = [
            [dec(sum(Bq[h][i] * Bq[h][j] for h in range(2))) for j in range(2)]
            for i in range(2)
        ]
        spectral = (
            (A[0][0] + A[1][1]
             + ((A[0][0]-A[1][1])**2 + 4*A[0][1]**2).sqrt()) / 2
        ).sqrt()
        require(float(spectral) <= case["kappa"] + ATOL,
                "Saved optimizer exceeds norm budget")
        sufficient = (
            Decimal(1) if case["dynamics"] == "identity"
            else (A[0][0]+A[1][1]).sqrt()
        )
        risk = population["s"]
        optimum = dec(risk).sqrt()
        limit = max(
            Decimal(0),
            optimum - dec(case["kappa"]) * (2*dec(population["f"])).sqrt(),
        )
    variance = sum(
        (x - Fraction(1, 2))**2 for row in population["current"] for x in row
    ) / 4
    for name, value in dict(
        assignment_loss=population["f"],
        sigma_res_sq=risk,
        exact_optimal_mse=risk,
        exact_optimal_rmse=optimum,
        population_certificate_limit=limit,
        current_coordinate_variance=variance,
        sufficient_kappa=sufficient,
    ).items():
        compare.close(saved[name], value, "reference_population")


def verify_evidence(folder):
    folder = Path(folder).resolve()
    manifest_before, file_count = check_inventory(folder)
    require(digest(folder / "campaign_spec.json") == CONFIG_SHA256,
            "Wrong pilot configuration")

    config = document(folder, "campaign_spec.json")
    metadata = document(folder, "campaign_metadata.json")
    summary = document(folder, "campaign_summary.json")
    references = document(folder, "references.json")
    require(metadata["config_sha256"] == CONFIG_SHA256, "Configuration metadata mismatch")
    require(metadata["campaign_id"] == config["campaign_id"]
            and metadata["campaign_role"] == config["campaign_role"] == "pilot",
            "Campaign mismatch")
    require(document(folder, "source_after.json") == metadata["source"],
            "Source records differ")
    require(metadata["source"]["git_status_porcelain"] == "",
            "Recorded source was not clean")

    compare = Comparison()
    cases = {case["case_id"]: case for case in config["cases"]}
    require(set(references) == set(cases), "Missing or extra references")
    for key, case in cases.items():
        compare.context = key
        verify_reference(references[key], case, compare)

    planned = {}
    for case in cases.values():
        for n in config["sampling"]["sample_sizes"]:
            for rep in range(config["sampling"]["replicates_per_cell"]):
                key = f"c{case['case_key']}_m{n}_r{rep}"
                planned[key] = dict(
                    attempt_id=key,
                    attempt_number=1,
                    case_id=case["case_id"],
                    seed=dict(
                        campaign_key=config["rng"]["campaign_key"],
                        case_key=case["case_key"],
                        n_pairs=n,
                        replicate_index=rep,
                        root_seed=config["rng"]["root_seed"],
                    ),
                )
    require(len(planned) == 10240
            and index_records(folder, "schedule.jsonl") == planned,
            "Schedule/configuration mismatch")
    samples = index_records(folder, "samples.jsonl")
    require(set(samples) == set(planned), "Missing or extra samples")

    event_counts = Counter()
    for event in records(folder, "attempts.jsonl"):
        key = event["attempt_id"]
        require(key in planned, "Unscheduled attempt event")
        expected = "started" if event_counts[key] == 0 else "completed"
        require(event_counts[key] < 2 and event["event"] == expected,
                "Unexpected attempt event")
        event_counts[key] += 1
    require(set(event_counts) == set(planned)
            and all(n == 2 for n in event_counts.values()),
            "Incomplete attempt accounting")

    seen = set()
    outcomes = Counter(
        zero_certificates=0,
        strict_coverage_failures=0,
        tolerance_aware_coverage_failures=0,
    )
    for result in records(folder, "results.jsonl"):
        key = result["attempt_id"]
        compare.context = key
        require(key in planned and key not in seen,
                f"Unscheduled/duplicate result: {key}")
        job = planned[key]
        case = cases[job["case_id"]]
        ref = references[job["case_id"]]
        sample = samples[key]["sample"]

        for name, value in job["seed"].items():
            require(type(sample[name]) is int and sample[name] == value,
                    f"Seed mismatch: {key}")
        for name, value in dict(
            sampler_id="four_state_multinomial_pcg64_v1",
            seed_material_order=config["rng"]["seed_material_order"],
            seed_material=[
                job["seed"][x] for x in config["rng"]["seed_material_order"]
            ],
            bit_generator="PCG64",
            seed_sequence="SeedSequence",
            seed_sequence_pool_size=4,
            draws_per_replicate=1,
            state_order=["a", "b", "d", "e"],
            probabilities=[.25]*4,
        ).items():
            require(sample[name] == value,
                    f"Sampling metadata mismatch {key}: {name}")

        counts = sample["state_counts"]
        n = job["seed"]["n_pairs"]
        require(isinstance(counts, list) and len(counts) == 4
                and all(type(x) is int and x >= 0 for x in counts)
                and sum(counts) == n, f"Invalid counts: {key}")
        require(
            result["case_id"] == job["case_id"]
            and result["evaluation_id"] == "four_state_original_iid_evaluation_v1"
            and result["certificate_id"] == config["certificate_id"]
            and result["count_statistics"]["statistics_id"] == config["count_statistics_id"]
            and result["count_statistics"]["state_counts"] == counts,
            f"Result identity mismatch: {key}",
        )

        stats = weighted_statistics(case["p"], case["dynamics"], counts)
        saved = result["count_statistics"]["statistics"]
        require(saved["n_pairs"] == n and saved["n_classes"] == 2
                and saved["class_counts"] == stats["sizes"]
                and saved["simplex_sum_atol"] == 1e-12,
                f"Statistics metadata mismatch: {key}")
        for name, value in dict(
            f_hat=stats["f"],
            s_hat=stats["s"],
            class_successor_means=stats["means"],
            class_successor_sse=stats["sses"],
            current_max_row_sum_error=0.,
            successor_max_row_sum_error=0.,
        ).items():
            compare.close(saved[name], value, "statistics_" + name)

        cert = result["certificate"]
        require(cert["n_pairs"] == n and cert["kappa"] == case["kappa"]
                and cert["delta"] == config["evaluation"]["delta"]
                and cert["f_hat"] == saved["f_hat"]
                and cert["s_hat"] == saved["s_hat"],
                f"Certificate input mismatch: {key}")
        independent = independent_certificate(
            stats["f"], stats["s"], n, case["kappa"], cert["delta"]
        )
        for name, value in independent.items():
            compare.close(cert[name], value, "certificate_" + name)
        for name in ("V_delta", "L_kappa_delta"):
            if independent[name] == 0.0:
                require(cert[name] == 0.0, f"Expected exact zero {key}: {name}")
        require(0. <= cert["F_delta"] <= 1. and 0. <= cert["V_delta"] <= 1.,
                f"Invalid certificate component range: {key}")

        L = cert["L_kappa_delta"]
        optimum = ref["exact_optimal_rmse"]
        limit = ref["population_certificate_limit"]
        require(0 <= L <= math.sqrt(max(0., cert["V_delta"])) + ATOL,
                f"Invalid certificate range: {key}")
        atol = config["evaluation"]["comparison_atol"]
        require(result["comparison_atol"] == atol,
                "Coverage-comparison allowance changed")
        for name, value in dict(
            strict_coverage=L <= optimum,
            tolerance_aware_coverage=L <= optimum + atol,
            zero_certificate=L == 0.,
        ).items():
            require(type(result[name]) is bool and result[name] is value,
                    f"Incorrect flag {key}: {name}")

        # Exact consistency of saved float arithmetic; operands were checked above.
        for name, value in dict(
            gap_to_optimum=optimum-L,
            population_bound_slack=optimum-limit,
            sampling_gap=limit-L,
        ).items():
            require(result[name] == value,
                    f"Signed gap changed/clipped {key}: {name}")

        tolerances = config["evaluation"]["exclusion_tolerances"]
        require(len(result["exclusions"]) == len(tolerances),
                "Missing exclusion decisions")
        for decision, tolerance in zip(result["exclusions"], tolerances):
            require(decision["tolerance"] == tolerance, "Exclusion tolerance mismatch")
            for name, value in dict(
                excluded=L > tolerance,
                exact_error_exceeds_tolerance=optimum > tolerance,
                erroneous_exclusion=L > tolerance and optimum <= tolerance,
            ).items():
                require(type(decision[name]) is bool and decision[name] is value,
                        f"Incorrect exclusion flag {key}: {name}")

        outcomes["zero_certificates"] += result["zero_certificate"]
        outcomes["strict_coverage_failures"] += not result["strict_coverage"]
        outcomes["tolerance_aware_coverage_failures"] += not result["tolerance_aware_coverage"]
        seen.add(key)

    require(seen == set(planned), "Missing evaluated replicates")
    require(summary["status"] == "completed" and summary["exit_code"] == 0
            and summary["source_unchanged"] is True
            and summary["active_attempt_id"] is None,
            "Campaign summary is not complete/clean")
    for name in ("planned", "started", "sampled", "succeeded"):
        require(summary[name] == len(seen), f"Summary mismatch: {name}")
    for name in ("failed", "not_started", "started_without_terminal_event"):
        require(summary[name] == 0, f"Summary mismatch: {name}")

    manifest_after, _ = check_inventory(folder)
    require(manifest_after == manifest_before,
            "Evidence inventory changed during verification")
    return dict(
        verifier_id=VERIFIER_ID,
        status="passed",
        verified_replicates=len(seen),
        verified_references=len(references),
        evidence_files_checked=file_count,
        evidence_manifest_sha256=manifest_before,
        execution_source_commit=metadata["source"]["execution_source_commit"],
        config_sha256=CONFIG_SHA256,
        decimal_precision=PRECISION,
        numerical_rtol=RTOL,
        numerical_atol=ATOL,
        numerical_values_checked=compare.numbers_checked,
        max_absolute_errors=compare.max_errors,
        recorded_outcome_counts=dict(outcomes),
        random_samples_regenerated=0,
        evidence_modified=False,
    )


def verifier_provenance():
    path = Path(__file__).resolve()
    root = path.parents[1]

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=root, stderr=subprocess.PIPE
        )

    require(not git("status", "--porcelain=v1", "--untracked-files=all").strip(),
            "Commit the verifier and use a clean source tree before recording verification")
    commit = git("rev-parse", "HEAD").decode().strip()
    relative = path.relative_to(root).as_posix()
    require(
        hashlib.sha256(git("show", f"{commit}:{relative}")).hexdigest() == digest(path),
        "Verifier bytes are not those committed at HEAD",
    )
    return dict(
        verifier_source_commit=commit,
        verifier_path=relative,
        verifier_sha256=digest(path),
        python_version=sys.version,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    try:
        provenance = verifier_provenance()
        report = verify_evidence(args.evidence)
        report.update(provenance)
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
