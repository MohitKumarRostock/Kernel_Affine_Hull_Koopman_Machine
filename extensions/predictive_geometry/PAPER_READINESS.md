# Paper positioning and author review

## What this paper contributes

**When Kernel Affine Hull Geometry Helps Prediction: Local Coordinates,
Decoder Capacity, and Conditional Gains** is a complete research manuscript
in JMLR format. It is a draft for author review, not a submitted or accepted
journal article. The author names, order, affiliations, and contact details
are carried forward from the supplied companion manuscript.

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
Its main remaining scientific vulnerability is **generality and significance
beyond the declared synthetic laws**, rather than insufficient code packaging.
JMLR formatting and reproducibility do not establish that the contribution
meets a particular venue's editorial threshold.

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
its significance must be argued at that scope.

Before actual submission, the authors should verify the text and results,
confirm authorship and affiliations, supply the appropriate funding and
conflict disclosures, and describe overlap with both companion manuscripts
to the venue. Those author-specific declarations have not been fabricated.
The paper includes disclosure of AI assistance.

## Where to review

- [Manuscript PDF](manuscript/main.pdf)
- [Editable LaTeX](manuscript/main.tex)
- [Scientific audit](SCIENTIFIC_AUDIT.txt)
- [Reproduction guide](REPRODUCING.md)
- [Generated quantitative claim sheet](manuscript/generated/claim_sheet.json)

The result tables and figures are generated from checked records. Headline
prose quantities use generated LaTeX macros; additional rounded narrative
quantities have been checked against the claim sheet and scientific audit.
Changing the scientific inputs still requires author review of the claims
and prose, rather than treating a newly compiled PDF as a new validated study.
