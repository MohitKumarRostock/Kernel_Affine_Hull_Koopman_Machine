# Literature and novelty audit

Prepared 28 October 2026. This is a targeted primary-source review, not a proof of
literature priority or a guarantee that all overlapping work has been found.

## Contribution boundary

The following are established ideas and must be credited rather than presented as
new in isolation:

- Soft state aggregation, including adaptive aggregation (Singh et al., 1994).
- KAHM reconstruction geometry (Kumar et al., 2024, 2025).
- Learned Koopman dictionaries (Li et al., 2017).
- Finite-sample risk analysis for Koopman regression (Kostic et al., 2022).
- Certified Koopman forecasting together with dictionary refinement (Conradie et
  al., 2026), and residual-driven learned dictionaries (Coote and Colbrook, 2026).
- Fitted-model risk upper bounds from bounded losses, empirical Bernstein bounds,
  and simultaneous testing or selection (Hoeffding; Maurer and Pontil; Learn then
  Test).
- Lower confidence bounds on model-class risk (Müller et al., 2025; Wu and Yang,
  2026), including dimension-independent lower bounds in appropriate settings.
- Pairwise Lipschitz inconsistency as a witness against exact interpolation
  (Blum and Nguyen, 2026), and Lipschitz model invalidation (Jin et al., 2020).
- Residual fitting and soft hierarchical gates (Bühlmann and Yu; Jordan and
  Jacobs).
- Block l1 norms and multiple kernel learning (Bach, 2008). A block norm may be
  useful for duplication-invariant refinement but is not a new regularizer.

The proposed paper can distinguish its particular result and implementation:
a population MSE lower witness formed from squared pairwise Lipschitz violations;
a direct finite-sample certificate of that witness; a necessary metric expansion
condition for a successful refinement; a hierarchy that retains previous
forecasting capabilities under a precisely specified metric and norm budget; and
an experimentally evaluated accept/exclude/refine decision for raw forecasts.

No exact prior theorem identical to

    inf_{Lip(g) <= k} E ||Y - g(Phi(X))||^2
      >= 1/2 E [(||Y-Y'|| - k ||Phi(X)-Phi(X')||)_+^2]

was located in the targeted searches. This is not enough to claim it is the first
such inequality. The proof is elementary, so the strongest novelty case should
be the consequences, learning procedure, and useful empirical decisions rather
than the inequality's existence alone.

## Especially close work

| Primary source | What it already does | Distinction to maintain |
|---|---|---|
| [Kumar and Kargaran 2026, v3](https://arxiv.org/abs/2609.32652v3) | Geometry-induced soft coordinates; linear-class lower confidence bounds; fitted-model risk bounds; independent-trajectory extension; conditional Koopman closure | New manuscript needs actual new learning method/results. Do not relabel an existing direct UCB as new. |
| [Conradie et al. 2026](https://arxiv.org/html/2603.15091v1) | Invariance diagnostics, multistep L2 and RKHS projection-error bounds, principal-angle decomposition, dictionary and kernel learning | Our target is finite-sample population raw-output risk and statistical class exclusion, not first certified Koopman forecasting/refinement. |
| [Coote and Colbrook 2026](https://arxiv.org/html/2606.29083v1) | Neural dictionary learning using ResDMD spectral residuals and a conditioning penalty | Spectral reliability is distinct from statistical mean-risk certification. |
| [Kostic et al. 2022](https://papers.nips.cc/paper_files/paper/2022/hash/196c4e02b7464c554f0f5646af5d502e-Abstract-Conference.html) | RKHS Koopman regression, reduced-rank estimator, independent and mixing-data bounds | Our held-out episodic certificate is model-specific; it does not replace their operator-estimation analysis. |
| [Learn then Test](https://arxiv.org/abs/2110.01052) | General predictive risk control through multiple testing | Finite frozen candidate selection is an application of this principle. |
| [Müller et al. 2025](https://proceedings.mlr.press/v291/muller25a.html) | Distribution-free limits for model-class-risk lower bounds | Avoid arbitrary rich-class falsification claims; a fixed metric/norm class is essential. |
| [Wu and Yang 2026](https://arxiv.org/abs/2606.23295) | High-confidence minimal-risk bounds from minimum empirical risk; dimension-independent lower confidence | Dimension independence alone is insufficient novelty. |
| [Blum and Nguyen 2026 v2](https://arxiv.org/abs/2610.17381v2) | Pairwise violation graphs for robust Lipschitz interpolation and pointwise prediction ranges | Pairwise violations are established; our expectation of squared violations bounds population MSE. |
| [Jin et al. 2020](https://arxiv.org/abs/2003.11662) | Lipschitz data-driven abstractions and optimization-based trajectory invalidation | Feasible-trajectory invalidation differs from expected-risk exclusion. |
| [Haseli and Cortés 2023](https://par.nsf.gov/servlets/purl/10466905) | Basis-invariant consistency index and a relative RMSE bound for dictionary predictions | Raw coordinate residuals can mislead; compare the same physical target across representations. |

## JMLR files and policy

Official formatting page: https://jmlr.org/format/format.html

Official style source: https://github.com/JmlrOrg/jmlr-style-file

Downloaded unmodified into this directory: `jmlr2e.sty`, `sample.tex`, `sample.bib`.
The style's `preprint` option removes journal-specific publication metadata.
Use it for an unsubmitted draft rather than inventing a volume, editor, dates of
acceptance/publication, or a JMLR paper identifier.

Author guidance: https://www.jmlr.org/author-info.html

Relevant guidance checked: concise and complete papers; clear and supported
claims; reproducible experiments; substantive differences from prior work. The
policy warns that papers over 35 pages can be harder to handle, and papers over
50 pages require cover-letter justification; appendix pages count. These are
editorial handling guidelines, not an acceptance rubric. Existing arXiv
preprints are permitted. Do not infer acceptance from topic fit or the use of a
JMLR template.

## Bibliographic notes

- `references.bib` contains 28 entries. Where a journal version exists it is
  used; new 2026 arXiv works are explicitly marked as preprints.
- Learn then Test is the 2025 Annals of Applied Statistics article, volume 19,
  issue 2, pages 1641--1662, DOI 10.1214/24-AOAS1998. Its arXiv posting predates
  publication.
- Haseli and Cortés is volume 7 (2023), pages 649--654; the DOI contains 2022
  because it was first available in October 2022.
- Singh et al. is the NIPS 1994 meeting, volume 7. Some catalogs date the printed
  proceedings 1995. The official NeurIPS page uses 1994.
- Evans and Jones authorship follows the original Gamma-test paper's title page;
  the institutional catalog reverses the display order.
- The manuscript should omit unused bibliography entries automatically through
  BibTeX; the audit does not recommend citation padding.

## Remaining research risks

1. A direct pair-score concentration bound may improve on the particular
   centered certificate near zero input variance without uniformly dominating
   it. Include counterexamples or numerical comparisons.
2. A necessary metric expansion condition is not a sufficient learning theorem.
   It cannot ensure recoverable signal when the target variation is noise.
3. A refinement may preserve representational inclusion but change the scale of
   the input metric and hence the meaning of a Lipschitz budget. State the metric
   and inherited norm precisely, and report comparable predictive budgets.
4. A valid finite-sample UCB can be too loose to support practical decisions.
   Realized certification rates, certificate widths, abstentions, and sample
   costs matter as much as empirical forecasting accuracy.
5. Independent simulation trajectories support episodic guarantees. Windowing a
   single real series does not yield independent episodes. A real-data result
   must use suitable independent units or clearly limit its statistical claim.
6. Upper/lower bounds must refer to the same target, law, and prediction class
   to support a near-optimality or exclusion statement.
