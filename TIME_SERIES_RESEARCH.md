# Forecasting evidence and scientific scope

The central study asks whether an affine ridge decoder benefits from local position and reconstruction departures when all representations predict the same observed future response. It evaluates G (memberships), P (memberships plus local principal coordinates), K (P plus KAHM departures), and R (P plus matched positive-kernel RBF departures).

## Supported conclusion

At a common bank selected by P on validation data, P reduces nonlinear mean loss by 69.201% relative to G, and K reduces it by 10.459% relative to P. Both prescribed 3% margins are established in the four-comparison family. K's 0.029% point improvement over R does not establish the KAHM-specific superiority margin. Its linear loss is 1.111% higher than P while satisfying the 5% noninferiority criterion.

These are conditional results for the complete fitted procedure. Separate block transformations preserve common coordinates, but feature nesting does not ensure that finite-sample test loss decreases. The common-bank choice is asymmetric: P selects the bank. Independently selected banks describe a different procedure, with slightly lower nonlinear mean loss for R than for K. Neither the small point difference nor an unsuccessful superiority test establishes equivalence.

The reconstruction benefit also varies across laws and horizons. K worsens P on nonlinear autoregression at both noise levels; the aggregate nonlinear gain is concentrated at horizon 12. The simulation laws were used in earlier research, so fresh repetitions are not new application domains. The natural-data policy study does not supply a direct real-data validation of this four-arm procedure.

## What the theory establishes

KAHM and positive-kernel RBF can have equal augmented raw spans under reference invariance. A specified irregular reference geometry separates those spans. The proof concerns exact reconstruction maps, not query fallback or nonlinear block transformations. An invertible coordinate change can change an isotropic ridge penalty, so equal spans do not force equal fitted predictors. Distinct spans do not establish lower risk.

The generic bounded-feature ridge result controls coefficient and correction magnitudes. It is not a forecast-accuracy theorem, and it does not establish stable recursive rollout. The experiments use direct observed-output readouts rather than autonomous coordinate propagation. No Markov sufficiency, Koopman invariance, or compression claim is made.

## Related findings

The [research index](docs/RESEARCH_INDEX.md) and focused supplement retain the dimension, preprocessing, residual-target, policy-augmentation, and natural-data results. In particular, the selected non-KAHM policy is stronger than KAHM in the dimension study; the fresh policy study fails both nonlinear superiority margins; and KAHM augmentation worsens aggregate loss by 2.273% in the descriptive natural-data panel. Those results constrain broader utility claims even though the matched representation comparison is positive.

Computational regeneration of the original residual-study units adds no independent observations. Primary and secondary comparison families remain separate within each study. No pooled significance statement is constructed from the collection of experiments.

## Verification and availability

The retained computational audit provides independent calculations for selected numerical steps, not external scientific replication. Saved-loss arithmetic can be checked from the compact manuscript evidence. Full model replay requires the separately indexed arrays; full simulation regeneration requires the complete frozen source and its dependencies. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) specifies these routes and their limits.
