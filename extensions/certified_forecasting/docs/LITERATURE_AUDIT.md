# Literature context and contribution boundaries

The targeted literature review began on 28 October 2026 and was updated on
29 October 2026 during the scientific revision. It identifies close prior work
and supports the manuscript's contribution boundaries; it is not an exhaustive
priority search.

## Established ingredients

Soft state aggregation and adaptive aggregation precede modern representation
learning (Singh et al., 1994). The local KAHM reconstruction models are inherited
from Kumar et al. (2024, 2025). Learned Koopman dictionaries, operator regression,
forecasting error bounds, and dictionary refinement also have established
literatures. Bounded-loss risk calibration, empirical Bernstein concentration,
multiple testing, residual fitting, soft gates, and group-norm regularization
are used here as existing tools.

The manuscript studies their connection to a specific forecasting problem:
a pairwise lower bound on population squared-error risk under a fixed
representation and Lipschitz budget; direct calibration of that lower-bound
statistic; necessary geometric change for successful refinement; and retained
blocks that preserve the feasible forecast class, including under exact
duplication. The finite-family decision distinguishes acceptance, exclusion,
and inconclusive outcomes for the same observed-output target.

The elementary pairwise inequality is not presented as a priority claim or a
new concentration theorem. Its consequences, the specified learning procedure,
and the empirical behavior of the resulting decisions form the contribution.

## Closely related primary sources

| Primary source | Relevant established result | Relationship to this manuscript |
|---|---|---|
| [Kumar and Kargaran (2026), v3](https://arxiv.org/abs/2609.32652v3) | Soft geometric coordinates, norm-bounded linear prediction limits, fitted-model upper risk bounds, and independent-trajectory extensions | The observed-output pair witness, retained refinement geometry, and implemented selection procedure extend these ingredients. |
| [Conradie et al. (2026)](https://arxiv.org/abs/2603.15091) | Invariance diagnostics, multistep projection-error bounds, principal-angle methods, and dictionary refinement | These concern Koopman approximation and projection errors; the present certificates target finite-sample expected forecasting loss under a specified episode law. |
| [Coote and Colbrook (2026)](https://arxiv.org/abs/2606.29083) | Dictionary learning using residual dynamic mode decomposition spectral errors and a conditioning penalty | Spectral residual control differs from statistical certification of population mean squared error. |
| [Kostic et al. (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/196c4e02b7464c554f0f5646af5d502e-Abstract-Conference.html) | RKHS Koopman regression and learning bounds for independent and mixing observations | The held-out episode procedure evaluates fixed models without replacing their operator-estimation analysis. |
| [Angelopoulos et al., Learn then Test](https://arxiv.org/abs/2110.01052) | Predictive risk control formulated as multiple testing | Selecting from a candidate family fixed before independent calibration is an application of this principle. |
| [Müller et al. (2025)](https://proceedings.mlr.press/v291/muller25a.html) | Distribution-free limitations on lower bounds for model-class risk | The specified metric and sensitivity restriction are essential; the manuscript does not claim unrestricted rich-class falsification. |
| [Wu and Yang (2026)](https://arxiv.org/abs/2606.23295) | Non-asymptotic minimal-risk bounds from empirical minimal risk, including dimension-independent lower confidence bounds under their assumptions | Dimension independence alone is not the distinguishing contribution here. |
| [Blum and Nguyen (2026), v2](https://arxiv.org/abs/2610.17381v2) | Pairwise Lipschitz violation graphs and prediction ranges under a bounded corruption budget | The manuscript instead averages squared pairwise violations to lower-bound population forecasting risk. |
| [Jin et al. (2020)](https://arxiv.org/abs/2003.11662) | Data-driven Lipschitz abstractions and trajectory infeasibility tests | Trajectory invalidation and expected-risk exclusion answer different questions. |
| [Haseli and Cortés (2023)](https://par.nsf.gov/servlets/purl/10466905) | A basis-invariant consistency index with a prediction-error interpretation | Representation comparisons here keep the observed forecasting target fixed. |
| [Bach (2008)](https://www.jmlr.org/papers/v9/bach08b.html) | Group Lasso and multiple-kernel learning with sums of block norms | The norm is established; its role here is a common sensitivity budget for retained and duplicated feature blocks. |

The original primary pages for Conradie, Coote and Colbrook, Wu and Yang,
Blum and Nguyen, Müller et al., Kostic et al., Learn then Test, Jin et al., and
Bach were checked during the 29 October revision. Their stated scopes support
the distinctions above. The bibliography preserves publication status: the
2026 arXiv works cited as preprints are not described as accepted journal papers.

## Statistical interpretation

The population pair witness dominates the stated root-variance obstruction
under the same conditional pairing law. This does not imply uniform domination
of every finite-sample centered confidence bound. A necessary amount of metric
expansion also does not guarantee a successful learned refinement; irreducible
future noise may prevent it.

A risk bound requires the target, evaluation law, metric, and sensitivity class
to be aligned. Population optima over nested classes need not translate into
monotone errors for separately fitted finite-sample models. The squared
Frobenius ridge penalty can change under duplication even when the feasible
prediction class is invariant.

Independent simulated trajectories support episode-level concentration;
overlapping windows from one series do not supply independent trajectories.
Calibration must also be independent of model-family construction. Reusing
calibration streams or a previously examined writer cohort after changing the
study design requires an additional adaptive-validity argument. The current
manuscript therefore distinguishes fresh synthetic calibration from nominal
follow-up calculations on reused data.

## Bibliographic and formatting provenance

The manuscript's [bibliography](../paper/references.bib) retains journal versions
where specified and explicit arXiv identifiers for preprints. Learn then Test
is cited as the 2025 *Annals of Applied Statistics* article, volume 19, issue 2,
pages 1641--1662, DOI `10.1214/24-AOAS1998`. Haseli and Cortés is cited in volume
7 (2023), pages 649--654; its DOI includes 2022 because of its earlier online
publication. The Singh et al. entry follows the official 1994 conference year.

The LaTeX style derives from the [official JMLR style repository](https://github.com/JmlrOrg/jmlr-style-file)
and uses preprint mode. [JMLR author information](https://www.jmlr.org/author-info.html)
and [formatting guidance](https://jmlr.org/format/format.html) are the authoritative
submission references. The manuscript does not assign journal publication
metadata or imply acceptance.
