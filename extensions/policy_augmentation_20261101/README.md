# Fresh policy-augmentation experiment

This extension evaluates the incremental contribution of an optional
in-sample KAHM correction to the unchanged expanded non-KAHM policy, against
a matched-count extra RBF residual search. See PROTOCOL.md and config.json
for the frozen scientific design. Earlier experimental results are not reused
as new observations.

From the repository root, using the declared requirements:

```bash
python extensions/policy_augmentation_20261101/code/pipeline.py freeze
python extensions/policy_augmentation_20261101/code/pipeline.py abc
python extensions/policy_augmentation_20261101/code/pipeline.py fit --workers 3
python extensions/policy_augmentation_20261101/code/pipeline.py seal-fits
python extensions/policy_augmentation_20261101/code/pipeline.py d
python extensions/policy_augmentation_20261101/code/pipeline.py predict --workers 3
python extensions/policy_augmentation_20261101/code/pipeline.py seal-predictions
python extensions/policy_augmentation_20261101/code/pipeline.py release
python extensions/policy_augmentation_20261101/analysis_code/analyze_policy.py
python extensions/policy_augmentation_20261101/audit/verify_policy.py
```

Run outputs are append-only. Reproduction should use a fresh extension-output
directory containing the same source/configuration and declared dependencies;
do not overwrite a completed or failed task. The files freeze.json,
FIT_LOCK.json, PREDICTION_LOCK.json, and TARGET_RELEASE.json identify the
actual completed execution. Raw data, fitted states, and full numerical
records are distributed separately from the compact repository analysis.

The inherited KAHM dimension and preprocessing code is unchanged. A cap is
an upper bound on local rank, and cap20 equals cap12 for the fixed history
width12. The added non-KAHM control lives in code/extra_rbf.py. Both added
families receive75 B candidate scores per horizon and one C family slot;
their feature widths and runtimes need not match.
