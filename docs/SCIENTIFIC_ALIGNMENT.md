# Scientific documentation alignment (maintainer checklist)

This file describes documentation maintenance for the experimental archive.
The [README](../README.md), [experiment guide](EXPERIMENT_GUIDE.md),
[reproducibility instructions](../REPRODUCIBILITY.md), and per-experiment
protocols contain the scientific definitions and procedures needed to
understand the results without the research manuscript.

## Terminology and scientific scope

The corresponding JMLR submission is titled *A Hierarchy of Risk Certificates
for Representation-Based Prediction*. The method is more general than
forecasting, although the datasets and experiment scripts here concern
forecasting and other prediction tasks.

Use **representation–decoder class** for the complete specification of a
representation, feature pseudometric, output interval, and decoder family.
Use **optimal population class risk** and **fitted-decoder risk** for distinct
quantities. An individual fitted-risk upper bound does not certify class
feasibility unless that fitted decoder is admissible.

The principal experiment groups have different statistical scopes:

- `controlled_calibration`: simultaneous confidence bounds on population MSE,
  conditional on development, with prespecified iid calibration.
- `ecg_cohort` and `class_diagnosis`: deterministic minimum empirical-risk
  bounds on fixed PTB-XL and Bike Sharing observations; these are not
  population confidence bounds. The fixed-dataset theorem permits targets
  outside the decoder output interval.
- `deployment_cost` and `public_screening`: observed accuracy, storage,
  inference-time, fitted-model counts, and their stated limitations.
- `decoder_diagnostics`: supplementary predictor comparisons and ablations,
  including unsuccessful outcomes.

## Editing and provenance

1. Review definitions in `README.md` and `docs/EXPERIMENT_GUIDE.md` together.
   Do not silently change experimental hypotheses, datasets, confidence
   allocations, class membership, or threshold decisions.
2. Preserve distinctions between recomputing numerical evidence from retained
   inputs, regenerating fitted predictions, retraining models, and making new
   timing measurements. A `PASS` status establishes only the documented
   verification scope.
3. Every tracked source or documentation file listed in `MANIFEST.json` is
   bound by a SHA-256 hash and byte count. When a documentation file changes,
   create a new source revision and update **only its corresponding manifest
   entry**, using the exact UTF-8 byte length and SHA-256 digest.
4. Never rewrite old `validation/` reports to make them appear to validate
   a new source manifest. An earlier validation record remains historical
   evidence for the old manifest digest; a new run produces separate output.
5. Run `python3 -c "import reproduce; print(reproduce.verify_manifest())"`
   from the repository root and execute the documentation check in CI. For
   numerical reproduction use `python3 reproduce.py --output <new-directory>`
   with the pinned dependencies.
6. Once reviewed and validated, record the immutable source commit in any
   manuscript data/code availability statement. Do not cite a moving branch
   as the archived experimental source.

## Document scope

This file is a maintainer checklist, not a substitute for per-experiment
protocols, full numerical replay, proof review, or a rendered-equation review.
The repository's scientific documentation must remain usable without access
to the associated manuscript.
