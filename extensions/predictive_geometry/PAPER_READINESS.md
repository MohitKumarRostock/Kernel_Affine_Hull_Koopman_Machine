# Controlled representation study: contribution and evidence

The controlled study examines how geometric features retain predictive position and how a finite decoder uses that information. Its results are included in the [integrated manuscript](../tmlr_integration/README.md). The local manuscript directory retains the earlier standalone document, **When Kernel Affine Hull Geometry Helps Prediction: Local Coordinates, Decoder Capacity, and Conditional Gains**.

## Main findings

On the specified Van der Pol generator, KAHM local-coordinate features reduce test mean squared error relative to both raw histories and matched affine local-coordinate features at two finite nonlinear decoder capacities. With 256 landmarks, the relative reductions are 47.3% and 19.1%, respectively. These are ratios of aggregate means. Inferential conclusions use paired absolute differences and their simultaneous intervals.

A separate target-intervention experiment keeps the inputs and learned geometry fixed while changing the target. It shows how a reconstruction score can discard signed position even when the underlying affine reconstruction is nearly exact. Locality-aware affine scoring and explicit local coordinates largely restore predictive access in this example. The intervention isolates a property of the supplied code; it does not prove that this mechanism fully explains the Van der Pol result or is exclusive to KAHM.

The capacity follow-up also contains unfavorable comparisons. Raw histories outperform KAHM local coordinates on nonlinear autoregression, while both raw histories and affine local coordinates outperform KAHM on Duffing. The contribution is a conditional account of representation and decoder behavior, not universal forecasting superiority.

## Relationship to existing constructions

| Existing construction | Use in this study |
| --- | --- |
| KAHM local kernel reconstruction | A specified geometric map whose forecasting features are compared with matched alternatives |
| Local coordinate and tangent coding | Gates multiplied by local displacement coordinates, used as an established representation design |
| Finite ridge and RBF decoders | Restricted fitted predictors that make decoder capacity and regularization explicit |
| Conditional expectation under squared loss | The information-theoretic distinction between a representation's information and a decoder's ability to use it |

The paper does not claim that local-coordinate coding or elementary conditional-risk identities are new general methods. Nor do the fitted ridge and finite RBF predictors automatically inherit guarantees derived for a separately specified decoder class or sensitivity constraint.

## Statistical and computational scope

The first stage has 29 primary contrasts, and the fresh capacity study has 12. Each has its own simultaneous-interval family; the two families do not jointly constitute a single 95% familywise statement. The capacity experiment was motivated by the first-stage observations before its own samples were generated. Local hash records document that sequence without establishing public preregistration.

Airfoil is one fixed grouped static-regression cohort. Twenty-four fitting seeds describe variation within that cohort, not 24 independent natural-data replications. The numerical query-row stabilization study reuses capacity samples and is a sensitivity analysis. It addresses a Gaussian-underflow pathway without proving that signed normalization is uniformly well conditioned.

Re-executing published seeds provides computational reproduction. It does not produce additional independent statistical evidence. The source snapshot, archived records, and current execution manifests identify those distinct roles.

## Evidence and reproduction

- [Reproduction commands](REPRODUCING.md)
- [Scientific audit](SCIENTIFIC_AUDIT.txt)
- [Generated quantitative claim sheet](manuscript/generated/claim_sheet.json)
- [Historical archive packaging](TMLR_SUBMISSION.md)
- [Integrated manuscript](../tmlr_integration/README.md)

Tables, figures, and numerical macros are generated from the archived records. The resulting computation supports the stated comparisons; the scope of each claim remains determined by the experimental units, population, and decoder used in that comparison.
