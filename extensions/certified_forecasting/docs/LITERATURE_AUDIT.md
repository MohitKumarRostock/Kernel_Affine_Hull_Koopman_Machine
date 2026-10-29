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
direct standard-deviation concentration, multiple testing, residual fitting,
soft gates, and group-norm regularization are used here as existing tools.
Paired-residual arguments for Lipschitz representation error and finite
quadratic programs for scalar Lipschitz regression also have direct
antecedents.

The manuscript studies their connection to a specific forecasting problem:
a conditional pairwise lower bound on population squared-error risk under a
fixed representation and Lipschitz budget; its exact slack decomposition;
stratification with estimated probability weights and random within-stratum
sample counts; and the compatibility needed for conditional optima to attain
the global optimum. The refinement results identify necessary geometric
change and retain earlier prediction classes, including under exact block
duplication. Simultaneous class exclusions and fitted-risk upper bounds also
give an interval for minimum feasible cost within the specified candidate
family. These statements concern the same observed-output target and
marginal evaluation law.

The elementary pairwise inequality is not presented as a priority claim or a
new concentration theorem. Stratification uses established concentration and
union bounds; relative-entropy inversion and empirical-minimum calibration are
comparison tools rather than new statistical inequalities. The contribution
is the stated collection of geometric results, calibrated decisions, and
empirical evaluation, with the limits of each claim made explicit.

## Closely related primary sources

| Primary source | Relevant established result | Relationship to this manuscript |
|---|---|---|
| [Akash et al. (2021)](https://ojs.aaai.org/index.php/AAAI/article/view/16815) | Lemma 2 bounds scalar prediction MSE for Lipschitz decoders from inverse contrastive loss using a paired-residual argument | This is a close antecedent of the representation-error principle; the manuscript compares the population expressions explicitly before developing conditional and stratified calibration. |
| [Gottlieb, Kontorovich, and Krauthgamer (2017)](https://www.wisdom.weizmann.ac.il/~robi/papers/GKK-MetricRegression-ITIT17-AuthorVersion.pdf) | Section III-A formulates finite scalar Lipschitz regression with pairwise constraints, including a quadratic program for squared loss | The empirical-minimum comparator uses this established structure and a Lagrangian dual lower value; it is not introduced as a new regression formulation. |
| [Kumar and Kargaran (2026), v3](https://arxiv.org/abs/2609.32652v3) | Soft geometric coordinates, norm-bounded linear prediction limits, fitted-model upper risk bounds, independent-trajectory extensions, and empirical-minimum comparisons in Section 2.5 | The conditional observed-output witness, stratified calibration, retained refinement geometry, and cost decisions extend these ingredients. |
| [Maurer and Pontil (2009)](https://arxiv.org/pdf/0907.3740) | Theorem 4 gives the empirical Bernstein mean bound; Theorem 10 gives direct sample-standard-deviation concentration | Both are attributed tools. The stronger centered comparator uses Theorem 10 and known support information, so an indirect square-root correction is not treated as unavoidable. |
| [Hoeffding (1963)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830) | Exponential inequalities for sums of bounded random variables | The relative-entropy mean endpoints and empirical-minimum lower endpoint are obtained from the established exponential chord argument and Chernoff inversion. |
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

### The closest paired-residual antecedent

Akash et al.'s [Lemma 2 and Appendix 7.3](https://arxiv.org/pdf/2102.08343)
consider recovering a continuous scalar nuisance variable by a Lipschitz decoder. To
compare under the same iid pair law, let the scalar target $Y$ have finite
second moment, take $\kappa>0$, set
$F=\{|Y-Y'|>a\}$ for $a>0$, write $\rho=\Pr(F)$, and suppose
$M=\mathbb E[\mathbf1_Fd(U,U')^2]\leq\varepsilon$. Their
inverse-contrastive-loss hypothesis implies this bound on $M$. With
$v=(|Y-Y'|-\kappa d(U,U'))_+$, the restricted $L^2$ triangle inequality
gives

$$
W_\kappa\geq\frac12\|\mathbf1_Fv\|_{L^2}^2
\geq\frac12(a\sqrt\rho-\kappa\sqrt M)_+^2
\geq\frac12(a\sqrt\rho-\kappa\sqrt\varepsilon)_+^2.
$$

In the positive regime $\varepsilon<a^2\rho^2/\kappa^2$, this is at
least their displayed bound $(a\rho-\kappa\sqrt\varepsilon)^2/4$.
This comparison credits the earlier principle and states the relation between
the two population expressions. It does not compare confidence corrections,
sampling designs, or all possible finite-sample procedures. The present
conditional-pair construction and estimated-mass stratified certificate are
analyzed separately; no global priority or statistical-dominance claim follows
from this inequality.

## Statistical interpretation

The population pair witness dominates the stated root-variance obstruction
under the same conditional pairing law. This does not imply uniform domination
of every finite-sample centered confidence bound. The updated comparison uses
known support widths, exact zero variance for a structurally constant
representation, and allocated direct-standard-deviation bounds. The retained
two-atom results with the earlier comparator are reported as historical
measurements rather than evidence against this stronger procedure.

The pair witness can be arbitrarily small relative to optimal risk on a
bounded three-atom law. Estimated-mass stratification can remove that
population gap for a compatible partition, but need not improve every
distribution or every finite-sample endpoint. Increasing the number of strata
can discard informative cross-stratum pairs and leave too few independent
pairs within each group. The broader Lipschitz optimum also need not equal the
optimum of a fitted clipped-affine family. These are distinct limits on what
more calibration data can resolve.

A necessary amount of metric expansion does not guarantee a successful learned
refinement; irreducible future noise may prevent it. Minimum feasible cost is
certified only when the lower and upper cost endpoints coincide, or when all
cheaper classes are excluded and an accepted model supplies the corresponding
upper endpoint. Otherwise the selected model is only the least costly
certified candidate.

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
follow-up calculations on reused data. Each additional experiment records its
own frozen protocol and independent sampling unit; those records do not
retroactively validate an earlier adaptive reuse of calibration data.

## Bibliographic provenance

The manuscript's [bibliography](../paper/references.bib) retains journal versions
where specified and explicit arXiv identifiers for preprints. Learn then Test
is cited as the 2025 *Annals of Applied Statistics* article, volume 19, issue 2,
pages 1641--1662, DOI `10.1214/24-AOAS1998`. Haseli and Cortés is cited in volume
7 (2023), pages 649--654; its DOI includes 2022 because of its earlier online
publication. The Singh et al. entry follows the official 1994 conference year.
Akash et al. is the 2021 AAAI article, volume 35, issue 8, pages 6582--6591,
DOI `10.1609/aaai.v35i8.16815`. Gottlieb et al. is the 2017 journal version
in *IEEE Transactions on Information Theory*, volume 63, issue 8, pages
4838--4849, DOI `10.1109/TIT.2017.2713820`; its earlier arXiv posting does not
change the journal publication year. Their primary articles were checked
directly during the additional theory revision.
