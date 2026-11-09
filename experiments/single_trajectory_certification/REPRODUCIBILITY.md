# Reproducing the single-trajectory certification experiment

## Computational identity

The computational record consists of the exact source implementation, the
fixed experimental protocol, all registered outcome records, saved full
certificates, analysis results, and scientific figures. The source archive
preserves the bytes of the implementation and protocol used for the experiment.
The public entry point verifies these identities before executing the archived
implementation.

| Object | SHA256 |
| --- | --- |
| `source_snapshot.zip` | `60b176a759b50c13ec20ec59e71ea886dfa7f63974b8e0ae6062baacb8ccedae` |
| Archived `protocol.json` | `9336d35b8594078152e965c09f5667137ad96aabdee07b6ee3c99e5f1735a0d7` |
| Uncompressed `results/runs.jsonl` | `8af10519b1f9d770f2af1cf454193d753f57af838dab4f1881eb23ff910a4e50` |
| `results/runs.jsonl.gz` | `3efd9c2f5f634263d08e5a97cb9efa7ed45601e418775d462eb10650d93e36a9` |

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records the complete list of
archive members and retained-file identities. The wrapper cross-checks the
archived protocol's engine hashes, the analysis manifest, and
[results/run_manifest.json](results/run_manifest.json). It also verifies every
retained certificate and figure file against the distribution manifest.
SHA256 checks establish consistency with the supplied record; they are not a
digital signature or external timestamp.

The outcome file is compressed losslessly. Its uncompressed size is 12,722,983
bytes and its distributed gzip size is 1,382,584 bytes. Decompression must
recover the logical SHA256 listed above. Compression changes no observation,
decision, numerical endpoint, timing record, or identifier.

## Environment

The scientific run used Python 3.12.14 on Linux and six worker processes.
All engine computations, population-oracle checks, tests, and statistical
analyses use the Python standard library. The verified reproduction environment
is Python 3.12.14. Other compatible Python versions can be tried, but their
success is determined by the full checks below, not by an assumed numerical
equivalence.

Figure regeneration uses matplotlib 3.10.8. To create an isolated environment
from this directory:

```sh
python3 -m venv /tmp/single_trajectory_env
/tmp/single_trajectory_env/bin/python -m pip install -r requirements-figures.txt
```

Use that environment's Python executable in place of `python3` in the commands
below when requesting figures. Without `--figures`, no third-party Python
package is required. No network access is used by the reproduction commands.
Installing the optional plotting dependency may require a package index or a
local wheel cache.

## Verify the retained experiment

From the directory containing `reproduce.py`, choose a new absolute output
directory outside the supplied experiment directory:

```sh
python3 reproduce.py --mode check --output /tmp/single_trajectory_check
```

For figure regeneration as well:

```sh
/tmp/single_trajectory_env/bin/python reproduce.py \
  --mode check --output /tmp/single_trajectory_check_with_figures --figures
```

The check executes the following operations in order:

1. Verify the source archive, every archive member, the protocol and source
   identities, all retained evidence, and the distributed scientific figures.
   Verify the decompressed outcome records against their logical identity.
2. Create the requested output directory and copy the retained evidence into
   its `results/` subdirectory, decompressing the complete outcome file there.
3. Extract the verified implementation into an isolated temporary directory
   and execute all 97 original unit tests.
4. Check complete retention of all 2,400 registered outcomes and reconstruct
   the twelve saved full certificates and population-risk proofs from their
   recorded seeds.
5. Recompute the aggregate summary and both explicitly post-hoc explanatory
   calculations.
6. Require exact byte equality of the regenerated `summary.json`,
   `saved_audit.json`, `posthoc_graph_diagnostic.json`, and
   `bottleneck_diagnostics.json` with their distributed counterparts.
7. If requested, regenerate both scientific figures in PNG and vector PDF.
8. Verify that the supplied archive and evidence are unchanged, and record the
   executed commands, log hashes, output hashes, and status in
   `reproduction.json`.

The aggregate and saved-certificate scripts write result files. For this
reason the wrapper copies the evidence before running them. Do not invoke
these scripts directly on the distributed result directory when preserving
the computational record.

The certificate audit has a defined scope. Every original scientific run was
checked during generation. The reproduction check verifies the complete
outcome roster and reconstructs the full evidence retained for replicate zero
in each setting, together with any exceptional certificates. The distributed
experiment has twelve such files and no exceptional outcomes. This check does
not claim to reconstruct 2,400 full certificates from the condensed records.
The separate independent-summary audit is retained and hash-verified; its
historical analysis is not silently relabeled as a newly executed check.

The post-hoc graph diagnostic reconstructs only the saved independent-bit
setting with $p=0.50,K=320$, replicate zero. The bottleneck description uses
only the existing outcome records. Neither analysis adds a new scientific
simulation or changes the primary protocol.

## Regenerate all registered outcomes

The following command executes the fixed twelve-setting, 200-repetition
protocol and then performs the saved-certificate audit, summary, and
explanatory analyses:

```sh
python3 reproduce.py --mode run --output /tmp/single_trajectory_run --workers 6
```

Append `--figures` to generate figures from the new outcome summary. The worker
count controls parallel execution, not the registered seed assignment. Each
setting and repetition has a fixed seed identity derived from the protocol.
The same probability and repetition use the same trajectory prefix across
matched settings.

Every outcome remains in the denominator, including unavailable executions.
The runner records a failed execution instead of replacing it with a favorable
sample. It refuses to overwrite an existing record set unless compatible
resumption is explicitly requested.

To continue an interrupted run in the same output directory:

```sh
python3 reproduce.py --mode run --output /tmp/single_trajectory_run \
  --workers 6 --resume
```

Resumption requires an output created by this wrapper in run mode and matching
archive and protocol identities. The archived runner also checks its scientific
source identities and existing outcome identifiers. Completed outcomes are
retained. A corrupted or incomplete serialized record is an error; the wrapper
does not silently repair or discard it. A failed output remains available for
inspection, with its status and logs recorded.

A regenerated run is expected to preserve the registered scientific design and
deterministic numerical computations in the recorded environment. Whole-file
byte equality with the original run is not required: parallel completion order,
process timing, timestamps, and runtime metadata can differ. The wrapper's
strict comparison to the distributed summary applies to check mode, which
uses the original ordered outcome records. Run mode computes its own complete
summary and audit from its newly generated records.

The original study took approximately 1,013 seconds of wall time with six
workers in the recorded shared environment. This is a historical reproduction
cost, not an execution-time guarantee. Per-process producer, checker, and
oracle CPU times in the records are not deployment-latency measurements.

## Output structure

| Output | Meaning |
| --- | --- |
| `reproduction.json` | Reproduction identity, attempts, executed commands, status, and evidence hashes |
| `logs/attempt_*/` | Complete output from each executed command |
| `provenance/protocol.json` | Exact protocol used for this execution |
| `provenance/protocol.sha256` | Original protocol identity |
| `provenance/analysis_manifest.json` | Original analysis source identities |
| `results/runs.jsonl` | Complete uncompressed repetition records |
| `results/run_manifest.json` | Scientific source identities and runtime metadata |
| `results/summary.json` | Complete setting-wise summaries and matched comparisons |
| `results/certificates/` | Full certificates retained under the protocol |
| `results/saved_audit.json` | Complete-roster and saved-certificate reconstruction results |
| `results/posthoc_graph_diagnostic.json` | Exact diagnostic for the designated saved graph |
| `results/bottleneck_diagnostics.json` | Descriptive class-risk, fitted-risk, and calibration gaps |
| `figures/` | Regenerated figures when `--figures` is requested |

A successful check reports 97 passing tests, 2,400 retained outcomes, twelve
reconstructed full certificates, matching regenerated evidence, and unchanged
supplied files. Optional PDF regeneration need not be byte-identical to the
distributed PDFs because graphics metadata can include creation times. The
distributed figures themselves are always hash-verified and remain unchanged.

## Inspect the exact implementation

The archive is an ordinary ZIP file. Its 29 members include the implementation,
nine test modules, the protocol, source manifests, the plotting dependency
specification, and computational validation records. The main source roles are:

| Archived source | Role |
| --- | --- |
| `certify_series.py` | Temporal layout, candidate freezing, endpoint allocation, decisions, and independent report verification |
| `certified_models.py` | Exact rational encoders and verified fitted decoder membership |
| `certified_lower.py` | Sparse-graph and disjoint-pair empirical lower certificates |
| `certified_kl.py` | Directed rational logarithm bounds and verified KL endpoint inversion |
| `population_oracle.py` | Exact stationary fitted risks and verified class-optimum enclosures |
| `run_experiment.py` | Fixed simulation protocol and complete outcome retention |
| `experiment_statistics.py` | Exact binomial-tail calculations and confidence enclosures |
| `summarize_experiment.py` | Registered aggregate and matched analysis |
| `audit_saved.py` | Reconstruction of saved certificates from raw seeds |
| `diagnose_saved_graph.py` | Post-hoc exact sparse/full empirical comparison |
| `describe_bottlenecks.py` | Post-hoc descriptive risk and witness analysis |
| `plot_results.py` | Scientific figure generation |

List or extract the archive with Python's standard ZIP utility:

```sh
python3 -m zipfile -l source_snapshot.zip
python3 -m zipfile -e source_snapshot.zip /tmp/single_trajectory_source_inspection
```

For verified execution, use `reproduce.py`; manual archive extraction is for
inspection or independently developed checks. The scientific definitions and
scope are described in [METHODS.md](METHODS.md), and the principal results are
summarized in [README.md](README.md).

## Interpreting a successful reproduction

Successful numerical reproduction supports the arithmetic, data flow,
registered simulation results, and reported conditional-on-premises claims.
It does not validate stationarity or a dependence bound for an application
dataset, establish a certificate for every arbitrary finite time series, or
prove informative minimum-cost selection in every run. The exact finite-state
population oracle is an evaluation instrument for the controlled experiment;
its answers are not supplied to the learner or calibrator.

The source snapshot preserves the scientific identity of the computation. The
reproduction entry point is an execution and evidence-verification layer; it
does not substitute a different encoder, optimizer, or statistical method.
