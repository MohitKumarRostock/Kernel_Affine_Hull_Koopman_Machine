# Target-adapted geometry: scientific scope and subsequent evidence

The target-geometry study examines two components of a geometric residual forecaster: a shared supervised diagonal metric and donor residuals computed without fitting the donor on the corresponding held-out group. The experiment asks whether either component improves KAHM and whether KAHM then exceeds matched affine, locality, and RBF-local procedures.

## Results of the specified comparisons

| Question | Evidence | Interpretation |
| --- | --- | --- |
| Does learned-metric KAHM improve raw ridge? | Favorable simultaneous intervals on all three synthetic laws, with mean error reductions of 11.18%, 53.26%, and 19.44% on nonlinear autoregression, Van der Pol, and Duffing | The complete correction uses predictive structure missed by this ridge baseline. |
| Does metric learning improve KAHM? | All three learned-versus-fixed metric intervals include zero | The incremental benefit of the shared metric is unresolved. |
| Do out-of-fold donor residuals improve KAHM? | All three donor-target increment intervals include zero | Excluding a row from donor fitting does not itself establish a better target for the final corrected baseline. |
| Does KAHM exceed equally trained local controls? | Five of nine intervals favor the control, four are unresolved, and none favors KAHM | The study does not establish a KAHM-specific predictive advantage. |
| Does the archive comparison favor the added procedure? | Learned KAHM reduces source-balanced scaled error by 2.18% relative to ridge, at 124.9 times the measured mean pipeline time; fixed in-sample KAHM and RBF-local reduce error by 2.76% and 3.27% | The modest gain relative to ridge must be assessed against simpler alternatives and their costs. |

The synthetic intervals form one family of 18 two-sided Bonferroni-adjusted paired Student intervals, with 12 independent repetitions per law. An interval containing zero does not establish equivalence. The natural-data results are descriptive, with equal weighting of four selected collections after weighting their three selected series equally.

The [complete findings](FINDINGS.md), [primary contrasts](reports/primary_contrasts.csv), and [collection results](reports/natural_sources.csv) provide the full numerical evidence. Timing ratios refer to the recorded implementation and host rather than an intrinsic algorithmic constant.

## What the metric and residual targets represent

The diagonal metric minimizes a training-only Gaussian residual-regression proxy. The same fitted metric is supplied to all four geometric procedures. Their local banks are then refitted and their ridge corrections selected using validation data. This tests transfer of a common supervised metric; it does not directly optimize the final KAHM forecasting function.

Out-of-fold residuals use donor baselines that exclude the relevant group or respect the chronological availability of its target. The final baseline can differ from those donors. The conditional-risk identity in [THEORY.md](THEORY.md) shows why that discrepancy matters under fixed functions and a common evaluation law. It does not guarantee that a donor target transfers well across chronological distributions or to a changed final baseline.

The finite experiments distinguish a useful complete correction from an identified benefit of one component. Improving ridge with a learned-metric correction is compatible with an unresolved metric increment and stronger simpler local controls.

## Relationship to subsequent studies

The [direct reconstruction experiment](../direct_reconstruction/README.md) subsequently optimized bounded local bandwidths and gate temperatures against actual horizon-specific forecasting error. It has a separate roster, budget, and statistical analysis. The [online adaptation study](../adaptive_development/REPRODUCING.md) evaluates a different chronological update procedure with bounded storage. The [reconstruction-departure experiment](../reconstruction_mechanism/README.md) isolates additional reconstruction functions and bounds the residual correction.

Those studies test different interventions and do not retroactively change this experiment's results. The [integrated manuscript](../tmlr_integration/README.md) compares their implications while retaining their separate populations, losses, and uncertainty statements.

## Generality

The present study uses finite tuning grids, capped univariate histories, three previously studied simulation laws, and four selected archive collections. Additional random realizations test the same simulation laws; related series and overlapping windows do not increase the number of independent application domains. The Bitcoin tasks concern activity indicators rather than prices.

The evidence leaves open which geometric and decoder conditions make a KAHM correction useful on a new forecasting population. Answering that question requires hypotheses and controls that identify the proposed mechanism, together with data and selection rules appropriate to the intended generalization claim. It cannot be inferred solely from a smaller reconstruction loss, a converged proxy optimizer, or a more complex representation.
