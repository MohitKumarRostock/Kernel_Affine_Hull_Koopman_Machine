# Exploratory training of local forecast gates

This pilot evaluates whether optimizing local gate temperatures for forecasting improves KAHM relative to fixed KAHM and equally trained geometric controls. It contains nine tasks: three independent realizations of each of nonlinear autoregression, Van der Pol, and Duffing. The findings are exploratory because each law has only three repetitions and the generators are a separate implementation of previously studied equations.

## Models and data partitions

The local reference bank, history standardizer, principal directions, and kernel regularizer remain fixed. For each horizon, every geometric procedure receives three bounded log inverse-temperatures initialized at zero. The optimizer has a limit of 160 objective evaluations per horizon and ridge penalty. Each objective evaluation refits the ridge residual predictor on partition A and measures forecasting mean squared error on B. Partition C selects the predictor penalty and correction multiplier separately for each horizon. Partition D supplies held-out evaluation.

The baseline is raw-history ridge selected within A and then held fixed. A zero correction multiplier reproduces that baseline. The fitting API does not accept D targets. B is training data for the gate parameters and is therefore not a confirmation set.

Each task has 64, 32, 32, and 64 independent episodes in A, B, C, and D, respectively. Episodes have 72 observations. Histories contain 12 observations, horizons are 1, 4, and 12, and there are 13 origins per episode. Fitting-row caps are 512 for A and 256 each for B and C; all 832 D rows are evaluated. The oscillator observation grid begins at time 10.0 and advances in steps of 0.1. These implementation details distinguish the pilot from a bitwise reproduction of the earlier experiments.

The raw RBF comparator concatenates standardized histories with Nyström features. Its landmarks come from A, and its bandwidth and penalty are selected on B. It is a comparator in this pilot's suite, not an identical implementation of every earlier raw-RBF procedure.

## Reproduction

From `extensions/forecast_gate_pilot`, use Python 3.12 and the NumPy, SciPy, and scikit-learn versions recorded in `pilot_results/protocol.json`:

```bash
python test_geometry.py
python test_online.py
python run_pilot.py --quick --output /tmp/kahm-gate-quick
python run_pilot.py --run --output /tmp/kahm-gate-pilot
python run_pilot.py --audit --output /tmp/kahm-gate-pilot
python run_pilot.py --audit --output pilot_results
```

New output directories must be absent or empty. The audit reconstructs windows from raw observations, recomputes C and D losses, verifies complete method and candidate rosters, checks first-minimum selection and zero-correction equality, and reconstructs charged costs. It does not independently refit every optimization candidate or establish generalization to a new population. No confidence interval is attached to the three-repetition means.

## Supporting online module

`BoundedWindowRidge` in `online.py` is a numerical component for a fixed feature map. It minimizes the sum of squared residuals plus a positive ridge penalty on every supplied coefficient, including an appended constant. This differs from the unpenalized-intercept convention in `geometry.py`.

The module maintains a capped window through rank-one additions and deletions, with a recorded replay of the retained rows when a numerical check fails. Operations carry a geometry version; changing the map requires resetting or replaying the retained observations under that map. It does not update principal directions, references, or metrics, and it does not implement a complete delayed-target forecasting stream.

## Records and interpretation

| File or directory | Contents |
| --- | --- |
| `geometry.py` | Fixed and forecast-trained geometric gates and forecasting controls |
| `run_pilot.py` | Generators, A/B/C/D partitions, parameter fitting, and saved forecasts |
| `audit_pilot.py` | Independent reconstruction of losses, selection, splits, hashes, and costs |
| `online.py` | Fixed-feature sliding-window ridge and geometry-version checks |
| `pilot_results/` | Inputs, selected predictions, candidate losses, optimizer records, source snapshot, and audit |
| `CONFIRMATORY_PROTOCOL_DRAFT.md` | An archived, unexecuted protocol proposal, separate from the pilot findings |

The original `README.md` is part of the pilot's immutable source checkpoint. This guide supplies the current scientific description without altering that historical record. The pilot code was implemented from equations and protocol descriptions, and its results are not a bitwise replay of the supplied manuscript's prior experiments.

Optimizing through a ridge solve and maintaining a bounded kernel or feature window are established numerical ideas. The scientific question is whether the complete KAHM forecasting procedure supplies a useful advantage over equally trained alternatives. The subsequent [direct reconstruction study](../direct_reconstruction/README.md) evaluates a separate, specified intervention on reconstruction bandwidths and gates; its evidence is reported independently.
