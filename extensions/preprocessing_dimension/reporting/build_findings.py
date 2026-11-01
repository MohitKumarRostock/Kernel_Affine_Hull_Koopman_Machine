#!/usr/bin/env python3
"""Assemble the scientific report from already frozen, independently audited results."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'analysis'


def main():
    panel = pd.read_csv(ANALYSIS / 'panel_summary.csv')
    primary = pd.read_csv(ANALYSIS / 'primary_contrasts.csv')
    summary = json.loads((ANALYSIS / 'analysis_summary.json').read_text())
    audit = json.loads((ROOT / 'audit/final_results_audit.json').read_text())
    assert audit['status'] == 'passed'
    def value(context, method='kahm_departure', *, family='primary', population='nonlinear', mode='calibrated'):
        x = panel[(panel.family == family) & (panel.context == context) & (panel.method == method)
                  & (panel.panel == population) & (panel['mode'] == mode)]
        assert len(x) == 1, (context, method, family)
        return float(x.iloc[0].smse)
    def delta(a, b): return 100 * (a / b - 1)
    std, mm, tanh, chosen = [value(k) for k in ('standard', 'minmax', 'tanh', 'selected')]
    policy, rbf = value('selected', 'nonkahm_policy'), value('selected', 'rbf_departure')
    methods = [('ridge', 'Ridge'), ('raw_rbf', 'Raw inputs + Nystrom RBF'),
        ('raw_hgb', 'Histogram gradient boosting'), ('persistence', 'Persistence'),
        ('common', 'Common geometry'), ('kahm_departure', 'Bounded KAHM departure'),
        ('rbf_departure', 'Bounded RBF departure'), ('kahm_unbounded', 'Unbounded KAHM departure'),
        ('rbf_unbounded', 'Unbounded RBF departure'), ('nonkahm_policy', 'Selected non-KAHM policy')]
    procedure_table = '| Procedure | Nonlinear SMSE | Linear SMSE |\n| --- | ---: | ---: |\n'
    for key, label in methods:
        procedure_table += f'| {label} | {value("selected", key):.6f} | {value("selected", key, population="linear"):.6f} |\n'
    labels = [('standard', 'Standard / standard'), ('minmax', 'Min-max / min-max'),
        ('tanh', 'Tanh / tanh'), ('hybrid_tanh_minmax', 'Tanh common / min-max departure'),
        ('hybrid_minmax_tanh', 'Min-max common / tanh departure'), ('selected', 'Joint preprocessing and cap selection')]
    preprocessing_table = '| Common / departure preprocessing | Nonlinear SMSE | Error change versus standard (%) |\n| --- | ---: | ---: |\n'
    for key, label in labels:
        score = value(key)
        preprocessing_table += f'| {label} | {score:.6f} | {delta(score, std):+.3f} |\n'
    comparison_names = ['Selected KAHM vs non-KAHM policy', 'Selected KAHM vs selected RBF',
                        'Tanh KAHM vs standard KAHM', 'Selected KAHM vs ridge, linear']
    comparison_table = '| Comparison | Required margin | Error change (%) | Simultaneous lower gap | Established |\n| --- | --- | ---: | ---: | --- |\n'
    for label, (_, row) in zip(comparison_names, primary.iterrows()):
        comparison_table += (f'| {label} | {row["margin"]} | {-row.point_relative_gain_percent:+.3f} | '
                             f'{row.simultaneous_one_sided_lower:.6f} | {"Yes" if row.criterion_established else "No"} |\n')
    fixed_table = '| Common / departure | Shared-bias baseline | KAHM | Selected RBF | KAHM gain over bias (%) | KAHM gain over RBF (%) |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    fixed_metrics = []
    for pair, label, common in [('ss', 'Standard / standard', 'standard'), ('mm', 'Min-max / min-max', 'minmax'),
        ('tt', 'Tanh / tanh', 'tanh_z_1'), ('tm', 'Tanh / min-max', 'tanh_z_1'), ('mt', 'Min-max / tanh', 'minmax')]:
        b = value(common, 'shared_bias', family='secondary')
        k = value(pair, 'kahm', family='secondary'); r = value(pair, 'rbf', family='secondary')
        fixed_table += f'| {label} | {b:.6f} | {k:.6f} | {r:.6f} | {-delta(k,b):.3f} | {-delta(k,r):.3f} |\n'
        fixed_metrics.append({'pair': pair, 'bias': b, 'kahm': k, 'rbf': r, 'gain_over_bias_percent': -delta(k,b), 'gain_over_rbf_percent': -delta(k,r)})
    fitting = json.loads((ROOT / 'execution/fit/stage_receipt.json').read_text())
    predicting = json.loads((ROOT / 'execution/predict/stage_receipt.json').read_text())
    freeze = json.loads((ROOT / 'freeze.json').read_text())
    metrics = {'analysis_summary_sha256': hashlib.sha256((ANALYSIS / 'analysis_summary.json').read_bytes()).hexdigest(),
        'audit_receipt_sha256': hashlib.sha256((ROOT / 'audit/final_results_audit.json').read_bytes()).hexdigest(),
        'freeze_sha256': hashlib.sha256((ROOT / 'freeze.json').read_bytes()).hexdigest(),
        'primary': primary.to_dict(orient='records'), 'fixed_common': fixed_metrics,
        'selected_KAHM_SMSE': chosen, 'standard_KAHM_SMSE': std, 'minmax_KAHM_SMSE': mm,
        'tanh_KAHM_SMSE': tanh, 'nonKAHM_policy_SMSE': policy, 'RBF_SMSE': rbf,
        'tanh_error_change_vs_standard_percent': delta(tanh,std),
        'tanh_error_change_vs_minmax_percent': delta(tanh,mm),
        'KAHM_error_change_vs_policy_percent': delta(chosen,policy),
        'KAHM_error_change_vs_RBF_percent': delta(chosen,rbf),
        'fit_stage_wall_seconds': fitting['wall_seconds'], 'prediction_stage_wall_seconds': predicting['wall_seconds'],
        'source_freeze_utc': freeze['created_utc'], 'audit_status': audit['status']}
    (ROOT / 'REPORT_METRICS.json').write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n')
    report = f'''# Preprocessing and validation-selected subspaces in KAHM forecasting

**Controlled simulation study - 1 November 2026**

## Abstract

Training-fitted tanh preprocessing was evaluated after correcting and selecting the local subspace dimension of kernel affine hull features. A prospective source and analysis freeze defined five common/departure preprocessing pairs, six native dimension caps, expanded RBF controls, and four simultaneous primary comparisons. The study comprised 24 independent repetitions and 192 task suites from four numerical observation laws. The tanh diagonal procedure had {delta(tanh,std):.2f}% higher aggregate nonlinear standardized mean squared error (SMSE) than the standard diagonal procedure, with an adverse difference in every repetition aggregate. Jointly selected KAHM had {delta(chosen,policy):.2f}% higher error than the selected non-KAHM policy and {delta(chosen,rbf):.2f}% higher error than independently selected bounded RBF. Only the linear-control noninferiority claim was established.

A separate experiment held the common forecast and shared bias fixed and fitted centered corrections to honest residuals from disjoint episodes. KAHM reduced error beyond shared bias by 10.78-14.93% and improved over independently selected RBF by 2.37-4.14% across its five settings. These secondary comparisons support a focused residual-correction mechanism under the declared allocation, while the primary comparisons do not establish a practical advantage for tanh or for the selected complete KAHM procedure.

## 1. Experimental design

The laws were linear autoregression, nonlinear autoregression, Van der Pol, and Duffing, each at added-noise levels 0 and 0.1. History length was 12 and horizons were 1, 4, and 12. Independent A/B/C/D episode counts were 64/32/32/128, with 512/256/256/1664 retained forecast rows. The 13 origins within an episode were dependent. Inference therefore used 24 independent repetition aggregates, rather than treating 192 tasks or forecast rows as independent observations.

For every horizon, SMSE was test MSE divided by the population variance of retained A targets, with a floor of 1e-12. Horizons were averaged equally, followed by equal averaging over the three nonlinear laws and two noise conditions within each repetition. Linear controls were analyzed separately. Lower SMSE is better. A percentage error change in the tables is positive when the candidate has higher error than its comparator.

All native caps 1, 2, 4, 8, 12, and 20 were fitted. Each local construction began at `min(cap, d, N_local - 1)` and applied the verified native numerical rank rules. No extra normal-direction reservation was imposed. Caps 12 and 20 were identical at this study's ambient dimension 12; both were independently fitted and checked. A cap is a construction limit, not a guarantee that every local model has that rank.

In the primary experiment, geometry histories used training-only standardization, an unclipped min-max map to [-1,1] on training values, or `tanh((x - mean_A) / std_A)`. Exactly constant min-max fitting coordinates mapped to zero. Transformed tanh histories were not restandardized. The finite design also included tanh common features with min-max departures and the reverse combination. Primary hybrid banks shared one requested cap. Existing feature-level standardization and radial bounds were retained after geometry construction.

In the primary architecture, complete heads refitted concatenated common and departure features against the raw-ridge residual. B selected ridge penalties by horizon, then one pair and cap across horizons, independently for each procedure. Raw RBF also received all three history scalers; HGB retained its native raw-input grid. B choices were saved before C was loaded. C selected finite blend multipliers and the non-KAHM family from the already B-selected procedures. The shared raw ridge and target units were unchanged. Full details are in the frozen protocol and analysis protocol.

<!-- pagebreak -->

## 2. Complete forecasting procedures

The primary selected context compares complete tuning and calibration procedures. Common geometry, bounded RBF, and unbounded RBF select their own applicable preprocessing and dimensions; they are not constrained to KAHM's selected bank.

{procedure_table}

The selected KAHM procedure improved over selected common geometry by 8.17%, but did not improve over the selected non-KAHM policy. Its difference from independently selected bounded RBF was small: +0.00007960 SMSE, with a descriptive unadjusted paired 95% interval of approximately [-0.000107, 0.000267]. This interval does not establish equivalence. At KAHM's selected complete bank, the KAHM point advantage over RBF was 0.77%, and its descriptive interval also included zero. The complete-procedure evidence therefore does not establish the prespecified 3% KAHM-specific margin.

![Figure 1. Ten complete procedures in the selected context. Error bars are descriptive intervals for repetition means; paired primary comparisons determine the formal claims.](reporting/figures/figure_02_complete_procedures.png)

## 3. Simultaneous primary comparisons

The four one-sided margin gaps used Bonferroni allocation of family alpha 0.05 and a t critical value of 2.397875 with 23 degrees of freedom. A claim was established only when its simultaneous lower margin gap was positive. The nonlinear requirement was at least 3% superiority; the linear requirement allowed at most 5% degradation relative to ridge. The lower-gap column is in SMSE units and already incorporates the required margin.

{comparison_table}

Only one of four claims was established: linear-control noninferiority. The selected KAHM error was 0.176% higher than ridge on that panel, within the declared 5% allowance. Passing this control and the numerical stability screens does not compensate for the failed nonlinear superiority claims. Failure of a superiority test is not a proof of equivalence or a universal claim about other data populations.

![Figure 2. Mean margin gaps and simultaneous lower bounds. Positive lower bounds establish the corresponding prespecified claim. Percentages are display rescalings of absolute SMSE gaps by observed mean comparator error, not risk-ratio confidence intervals.](reporting/figures/figure_04_primary_margin_bounds.png)

<!-- pagebreak -->

## 4. Preprocessing, hybrids, and dimension selection

Each fixed preprocessing row below selected its native cap on B. The joint row additionally selected the preprocessing pair on B. These rows were all specified before test evaluation.

{preprocessing_table}

Tanh had {delta(tanh,std):.3f}% higher error than standardization and {delta(tanh,mm):.3f}% higher error than min-max. Both adverse differences occurred in all 24 repetition aggregates. The descriptive unadjusted 95% interval for tanh minus standard was [0.001559, 0.002102] SMSE. With the C blend fixed at one, the tanh penalty relative to standard remained 6.487%. Tanh also had higher aggregate error at every prespecified fixed cap. Calibration therefore did not create the unfavorable result.

The tanh-common/min-max-departure hybrid improved on tanh/tanh by 5.151%, but remained above the standard and min-max diagonal errors. The reverse hybrid was 2.717% worse than min-max/min-max. These full-head comparisons establish that the two blocks' preprocessing choices are consequential under the finite design; they do not hold the fitted common forecast fixed.

Selecting the dimension reduced nonlinear KAHM error by 2.963% relative to fixed native cap 2 and by 7.136% relative to fixed native cap 20, when those fixed-cap contexts also selected preprocessing on B. The nonlinear joint selections used caps 1/2/4/8/12 on 12/22/79/20/11 of 144 tasks. Cap 20 tied exactly with 12 and was not selected. Those frequencies do not justify replacing validation selection with a universal cap of four.

The joint search had slightly higher observed error than the prespecified standard-only cap-selected procedure. The primary arm remains the frozen joint selection rule; the lowest test row was not retrospectively substituted for it. Increasing a validation search menu need not improve the resulting held-out procedure.

![Figure 3. Fixed-cap and cap-selected KAHM under the three diagonal scalers. The selected-cap column represents independent per-task validation choices, not a single common rank. Intervals are descriptive.](reporting/figures/figure_01_preprocessing_dimension.png)

## 5. Honest corrections around a fixed common forecast

The secondary architecture partitioned A by whole episode into P and R: the first 32 episode identities fitted each common predictor, and the remaining 32 supplied honest residual targets. All history scalers and banks were fitted on P. The common cap was fixed per row, while departure caps were selected separately. B selected common settings without access to R outcomes. Common selection used P target variance, so R targets could not enter through a normalization denominator. Departure feature statistics and ridge slopes were fitted on R. One B residual-mean bias was frozen per common row. The correction used centered bounded features times ridge slopes and omitted the response-mean intercept. B selected departure penalties and caps; C selected only the incremental multiplier.

Within each common row, KAHM and RBF therefore had the identical common-plus-bias baseline. Different common models generated their own residual targets. The complete main experiment and this smaller-common-fit mechanism experiment have different allocations and must not be pooled into a single ranking.

{fixed_table}

Across these five cells, KAHM improved over independently cap-selected RBF by 2.372-4.140%, and over RBF at the KAHM-selected departure cap by 3.431-4.543%. All corresponding descriptive paired intervals favored KAHM. These intervals are unadjusted secondary inference; they are not replacements for the primary simultaneous margin tests.

With the tanh common forecast fixed, replacing tanh departure preprocessing with min-max improved KAHM by 2.819%. With the min-max common forecast fixed, introducing tanh departures worsened KAHM by 2.608%. The KAHM two-by-two interaction interval in this fixed-common design included zero. The evidence supports a benefit of min-max departures in both tested common rows, without establishing a distinctive interaction synergy.

The tanh-common/min-max-departure cell achieved the largest percentage gain over its own baseline, 14.934%, but its absolute SMSE was 0.030555. Min-max/min-max had lower absolute error, 0.029497. The larger percentage gain partly reflects the weaker tanh common predictor.

![Figure 4. Incremental corrections relative to the identical common-plus-bias baseline within each row. Intervals are descriptive paired SMSE-difference intervals rescaled for display; the secondary architecture is separate from the primary ranking.](reporting/figures/figure_03_fixed_common_increment.png)

## 6. Heterogeneity and numerical behavior

Tanh's largest penalties relative to standardization appeared for noiseless Van der Pol (+25.89%), noiseless Duffing (+14.20%), and noisy Van der Pol (+9.55%). Nonlinear autoregression changed little; its noisy condition had a 0.039% point improvement. These condition summaries are descriptive and do not define a retrospectively chosen deployment rule.

The strongest-policy deficit was especially visible for noiseless Van der Pol. HGB and the selected policy achieved SMSE 0.003999, compared with selected KAHM at 0.008756. The policy chose HGB in all 24 repetitions of that condition. Conversely, selected KAHM had lower mean error than the policy on noiseless Duffing. The aggregate conclusion coexists with this heterogeneity.

All 192 selected-procedure maximum-error screens passed; the largest selected KAHM-to-ridge maximum-error ratio was {summary['maximum_selected_error_ratio']:.6f}, below the declared limit of five. No unexpected factor, coordinate, gate, or kernel failure was recorded. Normalized-reconstruction guard row counts and coefficient/correction-bound violations were zero. The documented zero-rank fallback in the native-policy wrapper was retained: 36 primary local-model query records per reconstruction kind, all under tanh, and 13 secondary records per kind. These counts include repeated caps or cached-bank evaluations and are not failed tasks. Their association with tanh does not by itself explain the predictive differences.

## 7. Integrity, resources, and reproducibility

The source/configuration/analysis freeze preceded development-data generation. Every fit and B selection was sealed before test episode generation. Every forecast was sealed before targets were released for scoring. No task, candidate, or scientific outcome was omitted, and no frozen numerical source was changed during production. This was a prospective local source freeze, subsequently preserved on GitHub, rather than a public preregistration.

The independent audit checked all 192 development tasks, 28,800 B scores, 7,680 B context choices, 1,536 C policy choices, and 3,456 A-only geometry banks. It independently reconstructed 48,384 actual local bases and verified all task-to-repetition aggregate arithmetic and the four primary comparisons. Full raw forecast-to-loss reconstruction covered the prespecified eight repetition-zero tasks: 4,016 arrays and 12,048 horizon-loss joins. It did not replay raw losses for every task. All 33 analysis-table checksums passed.

The complete fit stage took {fitting['wall_seconds']:.1f} seconds and prediction took {predicting['wall_seconds']:.1f} seconds with three workers and one numerical thread per worker. These totals include the finite search and diagnostic procedures. They are shared-host study timings, not isolated per-method deployment costs. This experiment consequently does not establish a method-level cost-effectiveness ranking.

The analysis package contains all exported tables, checksums, figure metadata, and the independent audit. Twelve disjoint numerical archives retain development and test arrays, fitted states, validation/calibration forecasts, all 96,384 terminal forecast arrays, local-basis records, diagnostics, and execution receipts. Each archived member is checked against its original SHA-256 digest. Source and reconstruction commands are supplied by the pinned research branch.

## 8. Interpretation and next research question

Under this design, tanh(z) does not improve the cap-selected complete KAHM forecast, and the joint preprocessing/dimension search does not establish a practical advantage over the selected non-KAHM policy. The results favor retaining dimension as a validation-selected construction parameter, while keeping preprocessing benefits conditional on the tested procedure and data population.

The positive mechanism result is more focused: a centered KAHM residual correction supplies incremental predictive information beyond a fixed common forecast and shared bias, with modest advantages over matched RBF. The next useful controlled question is whether those advantages persist when the common predictor is strengthened. A prospective follow-up should generate residual targets through honest episode-wise cross-fitting, refit the common predictor on the full designated training block while preserving separate tuning and calibration data, and compare the incremental KAHM and RBF corrections under matched selection. The non-KAHM policy should remain the complete-procedure comparator.

The present evidence concerns four numerical laws, history width 12, three horizons, a fixed construction family, one tanh strength, and a finite search. It does not address longer histories, true ambient dimensions above 20, other tanh strengths, new natural-data cohorts, general adoption suitability, or risk-certificate coverage. Natural-series adoption would require a separate prospective evaluation after the controlled mechanism question is resolved.

## Sources and exact evidence

- [Frozen protocol and executable numerical source](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/5cfd7a1745a52459b7b5045fe84803b9bbb2b56a/extensions/preprocessing_dimension).
- [Published corrected construction and preceding principal study](https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/3cd8e56d95ec1187deed6070541f26742ce4f1cc/extensions/dimension_forecasting).
- Primary evidence: `analysis/panel_summary.csv`, `primary_contrasts.csv`, and `repetition_losses.csv`.
- Scaling and mechanism evidence: `secondary_contrasts.csv`, `preprocessing_interactions.csv`, `fixed_common_contrasts.csv`, `fixed_common_interactions.csv`, and `condition_summary.csv`.
- Verification: `audit/final_results_audit.json`, stage locks, and `REPORT_METRICS.json`.
'''
    (ROOT / 'FINDINGS.md').write_text(report)
    print(json.dumps({'status': 'written', 'report_words': len(report.split()), 'primary_claims': summary['primary_comparisons_established']}))


if __name__ == '__main__': main()
