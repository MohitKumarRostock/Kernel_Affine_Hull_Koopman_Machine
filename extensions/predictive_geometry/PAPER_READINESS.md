# TMLR paper positioning and submission readiness

## What this paper contributes

**When Kernel Affine Hull Geometry Helps Prediction: Local Coordinates,
Decoder Capacity, and Conditional Gains** is a complete research manuscript
using the official TMLR style in anonymous review mode. Preparation of a
submission package does not assert that it has been submitted or accepted.
Author-specific OpenReview declarations are separate from the review PDF.

The positive KAHM claim is deliberately concrete. On the declared Van der Pol
forecasting generator, KAHM-derived local-coordinate features improve mean
test MSE relative to both raw histories and matched affine local-coordinate
features at each of two finite nonlinear decoder budgets. The 256-landmark
relative reductions are 47.3% and 19.1%, respectively. These are descriptive
ratios of means; the paper bases its inferential claims on paired absolute
differences and their simultaneous intervals.

The explanatory contribution is a separate controlled branch experiment.
Holding inputs and the learned geometry fixed while changing the target
reveals the importance of signed positional discrimination in the predictive
code. Near-perfect affine reconstruction does not prevent its matched score
code from becoming nearly insensitive to a relevant reflection. Locality-aware
affine scoring and explicit local coordinates largely repair that example.
This identifies a useful property in a KAHM comparison without claiming it
is exclusive to KAHMs or that it explains the Van der Pol result completely.

## Distinction from prior work

| Prior work | What is retained | Distinct contribution here |
|---|---|---|
| Original KAHM classification and collaborative-learning papers | Local kernel reconstruction machinery and its geometric motivation | Controlled attribution in prediction, including unfavorable conditions |
| Local coordinate and local tangent coding | Gating weights multiplied by local displacement coordinates | Evaluation of KAHM-derived gates within that established construction |
| Prediction-limit and risk-certificate manuscripts | Some upstream code, data-law definitions, and the need to distinguish representation and decoder | Fixed-input target intervention, new fitted-model experiments, fresh capacity follow-up, and numerical sensitivity |

The local-coordinate map, restoration of a normalization constant, and
elementary supporting propositions are not presented as new general methods
or KAHM approximation theorems. The diagnostic ridge and finite RBF heads
do not inherit the companion paper's fixed-sensitivity risk certificates.

## Evidence that must remain visible

- All 29 first-stage primary contrasts and all 12 fresh-stage primary
  contrasts have separate, complete tables. The two interval families do
  not jointly constitute a 95% familywise coverage statement.
- The follow-up was motivated by favorable first-stage observations before
  its own new data were generated. Internal hash freezes are not public
  prospective preregistrations or external time attestations.
- KAHM local coordinates lose to raw histories on nonlinear AR and to raw
  and affine-local alternatives on Duffing in the declared follow-up.
- Airfoil is one fixed grouped static-regression cohort. Twenty-four fitting
  seeds are not twenty-four independent natural-data replications.
- Query-row stabilization addresses a Gaussian-underflow pathway. It does
  not establish that signed normalization is universally well conditioned.
- The numerical rerun of published seeds validates computation; it is not
  additional independent scientific confirmation.

## Assessment before journal submission

The manuscript has a distinct empirical question, a supported conditional
positive finding, complete comparison families, and an executable artifact.
Its main scientific limitation is **generality beyond the declared synthetic
laws**. The conditional positive result and explanatory scope should remain
explicit. TMLR assesses whether claims have accurate, convincing support and
whether the findings interest and are clear to some of its audience; it does
not require a new state of the art or novelty of the studied method. Formatting
and computational reproducibility alone do not establish those criteria.

The highest-value additions would be:

1. Independent natural forecasting datasets, split by series, subject, or
   acquisition unit as appropriate, with strong tuned raw-input and local
   geometric baselines.
2. A geometric diagnostic specified from training/validation data that
   predicts the sign or size of KAHM's benefit on held-out tasks. The present
   study does not establish such a transferable criterion.
3. Tests across noise, observation maps, sample sizes, and geometric-bank
   sizes, motivated by an explicit hypothesis rather than a search for more
   favorable benchmarks.

These are proposed future experiments, not results silently included in the
paper. A submission may retain the present narrower explanatory scope, but
its scientific interest must be argued at that scope. These additions are not
formal TMLR submission requirements.

Before actual submission, authors must verify the text/results and complete
their OpenReview authorship, funding, conflict and related-work declarations.
Those author-specific facts are not inferred by the packaging code. The paper
includes disclosure of AI assistance. Read the current
[TMLR author guide](https://jmlr.org/tmlr/author-guide.html) and
[acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html).

The reproducibility supplement preserves the exact scientific source and
historical records. Its review manifest identifies retained licenses and
published upstream provenance, while first-party overview author metadata,
current-study repository links, Git history and named PDFs are omitted.
This is an explicit review view, not a claim that every attribution can be
removed or that the historical evidence has been independently timestamped.
The concrete build and verification procedure is in
[TMLR_SUBMISSION.md](TMLR_SUBMISSION.md).

## Where to review

- [Editable LaTeX](manuscript/main.tex)
- [Scientific audit](SCIENTIFIC_AUDIT.txt)
- [Reproduction guide](REPRODUCING.md)
- [Generated quantitative claim sheet](manuscript/generated/claim_sheet.json)

The result tables and figures are generated from checked records. Headline
prose quantities use generated LaTeX macros; additional rounded narrative
quantities have been checked against the claim sheet and scientific audit.
Changing the scientific inputs still requires author review of the claims
and prose, rather than treating a newly compiled PDF as a new validated study.
