# Natural policy augmentation research

This extension compares P0, P1=P0+in-sampleKAHM, and
P0plus=P0+an additional NyströmRBF residual family on48 newly selected M4
Train-series identities. See `PROTOCOL.md` and `config.json` for the complete
chronology, candidate menus, and limits on inference.

From the repository root:

```bash
python extensions/policy_augmentation_natural_20261101/code/natural_pipeline.py fixture
python extensions/policy_augmentation_natural_20261101/code/natural_pipeline.py freeze
python extensions/policy_augmentation_natural_20261101/code/natural_pipeline.py prepare-abc
python extensions/policy_augmentation_natural_20261101/code/natural_pipeline.py fit --workers 3
python extensions/policy_augmentation_natural_20261101/code/natural_pipeline.py prepare-query
python extensions/policy_augmentation_natural_20261101/code/natural_pipeline.py predict --workers 3
python extensions/policy_augmentation_natural_20261101/code/natural_pipeline.py release-targets
python extensions/policy_augmentation_natural_20261101/code/natural_pipeline.py analyze
python extensions/policy_augmentation_natural_20261101/code/natural_pipeline.py verify
```

Stages refuse to overwrite existing scientific records. A refit reproduction
requires a separate clean checkout or output root carrying the same source,
configuration, roster, and source inputs. Raw official Train CSVs and M4
metadata are pinned and hashed; no official Test file is required.

`analysis/task_methods.csv` and `horizon_methods.csv` retain the complete
series/method and horizon results. `aggregates.csv` uses equal weighting
within frequency and between frequencies. `policy_contrasts.csv` gives both
complete-policy comparisons. Policy selections, conditional subsets, and
resources are retained separately. `audit/verification.json` records the
post-release numerical audit and its precise scope.

The natural benchmark panel is descriptive external evidence. It is not a
fresh set of independent source domains or an official M4 competition run.
The one-series chronological adaptation omits the synthetic study's
out-of-fold RBF candidate; its menu and results remain separately reported.
