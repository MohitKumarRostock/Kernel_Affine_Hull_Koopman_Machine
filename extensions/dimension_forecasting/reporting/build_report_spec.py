#!/usr/bin/env python3
"""Prepare a complete research report from the two independently audited studies."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pandas as pd


def build(forecast, risk, output):
    forecast=Path(forecast).resolve();risk=Path(risk).resolve();output=Path(output).resolve()
    fs=json.loads((forecast/"analysis/analysis_summary.json").read_text())
    fa=json.loads((forecast/"audit/post_release/final_results_audit.json").read_text())
    run=risk/"runs/20261101"; rs=json.loads((run/"analysis/summary.json").read_text())
    ra=json.loads((risk/"audit/post_release/final_results_audit.json").read_text())
    assert fs["status"]=="completed" and fa["status"]==ra["status"]=="passed"
    assert fs["integrity"]["seal_sha256"]["freeze.json"]=="1b80460fe2d33ede50bd085c026f7cdb6219e4f54057ff01d67c0e026fbcd60e"
    assert rs["freeze_sha256"]=="0a92dfb3d0f086541d989d36277620b49c103975d8608f7dc4f956bfe9b3a9ff"
    def ftable(name):
        p=forecast/"analysis"/(name+".csv")
        assert hashlib.sha256(p.read_bytes()).hexdigest()==fs["tables"][name+".csv"]["sha256"]
        return pd.read_csv(p)
    risk_files=json.loads((run/"analysis/analysis_files.json").read_text())
    def rtable(name):
        p=run/"analysis"/(name+".csv")
        assert hashlib.sha256(p.read_bytes()).hexdigest()==risk_files[name+".csv"]
        return pd.read_csv(p)
    panel=ftable("panel_summary");primary=ftable("primary_contrasts");secondary=ftable("secondary_contrasts")
    conditions=ftable("condition_summary");decomp=ftable("decomposition_summary");caps=ftable("cap_selections")
    risk_policy=rtable("selected_policy_summary");risk_comparison=rtable("policy_comparison_summary");risk_selected=rtable("selected_forecasts")
    def one(frame,**keys):
        selected=frame
        for k,v in keys.items():selected=selected[selected[k]==v]
        assert len(selected)==1,(keys,len(selected))
        return selected.iloc[0]
    def loss(view,method="kahm_departure",scope="nonlinear"):
        return float(one(panel,panel=scope,view=view,method=method,mode="calibrated").smse)
    historical=loss("context_historical2");native=loss("context_native20");selected=loss("context_selected")
    n=lambda x:f"{float(x):.6f}"
    signed=lambda x:f"{float(x):+.6f}"
    percent=lambda x:f"{float(x):+.2f}%"
    gain=lambda old,new:100*(1-new/old)
    labels={"ridge":"Raw-history ridge","raw_rbf":"Raw-input RBF","raw_hgb":"Histogram gradient boosting","persistence":"Persistence (calibrated)",
        "common":"Common local features","kahm_departure":"Bounded KAHM departure","rbf_departure":"Bounded RBF departure", "kahm_unbounded":"Unbounded KAHM departure",
        "rbf_unbounded":"Unbounded RBF departure","nonkahm_policy":"Selected non-KAHM procedure"}
    laws={"nonlinear_ar":"Nonlinear AR","vanderpol":"Van der Pol","duffing":"Duffing","harmonic":"Harmonic","quasiperiodic":"Quasiperiodic","logistic":"Logistic"}
    pages=[]
    def page(title=None):
        value=[];pages.append(value)
        if title:value.append({"type":"heading","text":title})
        return value
    def text(page,body,kind="body"):page.append({"type":kind,"text":body})
    def table(page,headers,rows,widths):page.append({"type":"table","headers":headers,"rows":rows,"widths":widths})
    def image(page,path,height):page.append({"type":"image","path":os.path.relpath(path, forecast/"reporting"),"max_height":height})
    fig=output/"figures"

    p=page()
    text(p,"Subspace selection in KAHM forecasting","title")
    text(p,"Comparative forecasting and finite-sample risk certificates | 1 November 2026","subtitle")
    text(p,"Abstract","subheading")
    text(p,"Local subspace dimension is evaluated as a validation-selected component of KAHM forecasting. Two independent experiments examine complete forecasting procedures and the utility of risk certificates for retained representations. The principal comparison preserves ten procedures and the original six primary criteria across 24 fresh repetitions. A separate bounded-data study evaluates three dimension policies across 26 repetitions and six observation laws. Dimension selection improves the principal nonlinear aggregate and lowers test MSE on five of the six certificate-study laws, while a larger fixed cap has mixed or adverse effects. The gain in the principal study is driven by the shared local representation; the incremental advantage of KAHM over equally tuned RBF reconstruction remains small. Certificate decisions for the selected forecasts do not improve under dimension selection.")
    table(p,["KAHM dimension policy","Nonlinear standardized MSE","Change from cap two"],[
        ["Historical cap two",n(historical),"Reference"],["Native cap twenty",n(native),percent(100*(native/historical-1))],
        ["Cap selected on validation B",n(selected),percent(100*(selected/historical-1))]],[.40,.31,.29])
    text(p,"Positive changes indicate higher error. The nonlinear aggregate gives equal weight to nonlinear AR, Van der Pol and Duffing; noise levels 0 and 0.1; horizons 1, 4 and 12; and the 24 independent repetitions. These values use the forecasting study's A-variance normalization and are distinct from the certificate study's raw MSE.","caption")
    image(p,fig/"forecast_dimension_selection.png",185)
    text(p,"Thin lines show individual repetition aggregates; large markers show their means. The cap histogram covers the 144 nonlinear task suites. At ambient dimension 12, native caps 12 and 20 define identical admissible ranks; the ascending tie rule selects 12.","caption")
    text(p,f"The selected-cap procedure reduces aggregate error by {gain(historical,selected):.2f}% relative to cap two and by {gain(native,selected):.2f}% relative to cap twenty. Three of the six predeclared primary criteria are established. The complete criterion for advancing the method is not established.")

    p=page("1. Complete forecasting comparison")
    text(p,"All procedures use the same A/B/C/D episode roles. A fits representations and readouts; B selects regularization and, for geometric procedures, one cap across horizons; C selects blending coefficients and a complete non-KAHM procedure. Every fitted state precedes D generation, and every D forecast precedes target release. The selected context tunes the five geometric procedures independently.")
    order=["ridge","raw_rbf","raw_hgb","persistence","common","kahm_departure","rbf_departure","kahm_unbounded","rbf_unbounded","nonkahm_policy"]
    table(p,["Procedure","Cap two context","Cap twenty context","Selected-cap context"],
          [[labels[m],n(loss("context_historical2",m)),n(loss("context_native20",m)),n(loss("context_selected",m))] for m in order],[.43,.19,.19,.19])
    text(p,"Mean nonlinear standardized MSE; lower is better. Raw-input competitors and persistence do not depend on the geometric cap. The non-KAHM policy chooses among seven complete procedures on C and is evaluated on D without further selection. Unbounded variants omit the departure's radial transformation while retaining bounded common features.","caption")
    text(p,"Primary comparisons","subheading")
    rows=[]
    for _,r in primary.iterrows():
        name=labels[r.comparator] if r.panel=="nonlinear" else "Linear ridge control"
        criterion="3% improvement" if r.panel=="nonlinear" else "5% noninferiority"
        rows.append([name,criterion,percent(r.point_relative_gain_percent),signed(r.simultaneous_one_sided_lower),"Established" if r.criterion_established else "Not established"])
    table(p,["Comparator","Criterion","KAHM point gain","Lower margin bound","Conclusion"],rows,[.29,.18,.16,.18,.19])
    text(p,"Positive point gain means lower KAHM error. The five nonlinear gaps are 0.97 times comparator loss minus KAHM loss. The linear gap is 1.05 times ridge loss minus KAHM loss. Each lower bound uses t(1 - 0.05/6, 23), giving an approximate 95% simultaneous family under the paired-t assumptions. A bound must be strictly positive. The 3% and 5% margins are the study's research criteria, not universal requirements for scientific relevance.","caption")
    text(p,"KAHM improves on raw RBF and common features by more than the required margin. Its point gains of 0.72% over independently tuned RBF reconstruction and 2.39% over boosting do not establish that margin. KAHM's mean error is 4.08% higher than that of the C-selected non-KAHM policy. The linear control meets noninferiority while its point error is 0.20% higher than ridge.")

    p=page("2. Representation mechanism and heterogeneous effects")
    d=one(decomp,panel="nonlinear",view="matched",mode="calibrated")
    table(p,["Selected versus historical cap-two KAHM","Change in standardized MSE"],[
        ["Complete predictor",signed(d.complete_change)],["Change in common-feature predictor loss",signed(d.common_change)],
        ["Change in the additional KAHM benefit",signed(d.departure_increment_change)]],[.69,.31])
    text(p,"The decomposition is an exact loss identity evaluated with common features and KAHM departures from the same selected bank. The negative common-feature term improves the complete predictor; the positive incremental term means that the additional KAHM benefit becomes smaller. Thus improved complete-procedure accuracy does not establish a stronger reconstruction contribution.")
    text(p,"KAHM remains useful relative to its common-feature control: the same-bank error reduction is 11.59%. Its advantage over the same-bank RBF departure is 0.68%, compared with 0.72% when caps are selected independently. Those small RBF contrasts have secondary, unadjusted paired-t intervals and do not satisfy the primary 3% margin. Bounded KAHM also has 2.28% lower mean error than its unbounded counterpart on the same bank.")
    rows=[]
    for law in ("nonlinear_ar","vanderpol","duffing"):
        for noise in (0.,.1):
            k=one(conditions,law=law,noise_level=noise,view="context_selected",method="kahm_departure",mode="calibrated").smse
            old=one(conditions,law=law,noise_level=noise,view="context_historical2",method="kahm_departure",mode="calibrated").smse
            rbf=one(conditions,law=law,noise_level=noise,view="context_selected",method="rbf_departure",mode="calibrated").smse
            rows.append([laws[law],f"{noise:.1f}",n(k),percent(gain(old,k)),percent(gain(rbf,k))])
    table(p,["Law","Added noise","KAHM standardized MSE","Gain over cap two","Gain over tuned RBF"],rows,[.23,.13,.22,.21,.21])
    text(p,"Condition means are descriptive. Noise zero still includes the source oscillator sensor noise; it means no additional Gaussian corruption. KAHM's advantage over RBF is concentrated in the oscillator conditions without added noise and reverses with added noise. Boosting remains substantially better on Van der Pol without added noise, so the KAHM-RBF contrast is insufficient to establish broad competitiveness.","caption")
    image(p,fig/"primary_margin_bounds.png",158)
    text(p,"Dots show mean nonlinear margin gaps; left ticks show their simultaneous one-sided lower bounds. The vertical line is zero. The separate linear noninferiority result is given in the preceding table.","caption")

    p=page("3. Forecasting within the certificate workflow")
    text(p,"The certificate experiment uses bounded synthetic observations and clipped block-direct forecasts. Each repetition has 128 training, 512 validation, 4,096 calibration and 2,048 test episodes. It evaluates histories 4 and 12 at common origins, native caps 1, 2, 4, 8, 12 and 20, and retained prefixes with 2, 6, 14 and 30 coordinates. A single history/cap bank is selected using the validation loss of its best prefix; all four prefixes of that bank remain available for certification. This preserves nested representation classes.")
    rows=[]
    for law in ("harmonic","quasiperiodic","nonlinear_ar","logistic","vanderpol","duffing"):
        values={arm:one(risk_policy,system=law,arm=arm).mean_test_mse for arm in ("cap2","native20","selected_cap")}
        count=int(one(risk_policy,system=law,arm="cap2").repetitions)
        rows.append([laws[law],str(count),n(values["cap2"]),n(values["native20"]),n(values["selected_cap"]),percent(gain(values["cap2"],values["selected_cap"]))])
    table(p,["Law","Runs","Cap two","Cap twenty","Selected cap","Selected gain"],rows,[.24,.12,.16,.16,.16,.16])
    text(p,"Raw test MSE averaged over the three horizons and then repetitions within each law. Positive gain means lower selected-cap error. Laws are reported separately; their errors are not pooled into a common physical risk.","caption")
    text(p,"The selected policy improves in 18 of 26 repetitions and ties in eight, with no observed worsening. It chooses caps one, two and four in 8, 8 and 10 repetitions. Every chosen KAHM forecast retains the 30-coordinate prefix, and selected-cap and cap-two policies choose the same history and prefix in all 26 repetitions. Cap twenty is not an actual rank of twenty: its selected banks store ranks 3, 4 or 12. It worsens Duffing mean error by 20.12% relative to cap two.")
    text(p,"Comparison with other procedures","subheading")
    rows=[]
    control_names={"affine":"Affine hull","rbf":"RBF bank","ridge_ar":"Ridge AR","polynomial":"Quadratic regression"}
    for law in ("harmonic","quasiperiodic","nonlinear_ar","logistic","vanderpol","duffing"):
        subset=risk_policy[(risk_policy.system==law)&risk_policy.arm.isin(control_names)]
        best=subset.loc[subset.mean_test_mse.idxmin()]
        selected_row=one(risk_policy,system=law,arm="selected_cap")
        rows.append([laws[law],control_names[best.arm],n(best.mean_test_mse),n(selected_row.mean_test_mse)])
    table(p,["Law","Lowest-error listed control","Control MSE","Selected KAHM MSE"],rows,[.24,.34,.21,.21])
    text(p,"This descriptive table identifies the smallest observed mean among the distinct affine, RBF, ridge-AR and quadratic controls; it defines no new selection rule or superiority test. Quadratic regression remains stronger on four laws. KAHM has the smallest mean among these procedures on nonlinear AR and logistic.","caption")

    p=page("4. Certificate utility is distinct from point accuracy")
    text(p,"A single failure allocation of 0.05 covers 4,680 designated one-sided statements across the 26-repetition campaign: 3,744 fitted-loss upper and class-witness lower endpoints, and 936 improvement endpoints. Every designated tail receives 0.05/4,680. The allocation is more conservative than the earlier study's per-repetition event, so changes relative to historical widths cannot be attributed to dimension alone.")
    table(p,["Selected-forecast RMSE tolerance","Accepted per policy","Excluded per policy","Unresolved per policy"],[
        ["0.20","5 / 26","0 / 26","21 / 26"],["0.35","13 / 26","0 / 26","13 / 26"],["0.50","21 / 26","0 / 26","5 / 26"]],[.37,.21,.21,.21])
    text(p,"Every job-by-tolerance decision is identical for cap two, cap twenty and selected cap. Acceptance requires all horizon upper endpoints to be at or below the squared RMSE tolerance. Exclusion requires at least one horizon lower endpoint to exceed it. Decisions are based on the calibration event, not on test errors.","caption")
    table(p,["Certificate diagnostic","Observed result"],[
        ["Positive adjacent-prefix improvement lower endpoints","132 / 702"],
        ["Positive dimension-policy improvement lower endpoints","0 / 234"],
        ["Zero lower endpoints for selected KAHM forecasts","234 / 234"],
        ["Upper endpoints at the squared-loss bound of four","0 / 1,872"],
        ["Conflicting acceptance and exclusion decisions","0"]],[.70,.30])
    text(p,"The positive improvement certificates concern refinements within a fixed retained bank. They do not certify the observed improvements from switching dimension policy. The test-error gains and the unchanged acceptance decisions therefore support different statements: dimension selection improves observed accuracy, while improved selected-forecast certificate resolution has not been established.")
    text(p,"A zero class lower endpoint is uninformative about a positive obstruction; it does not establish that the class can attain zero population risk. Upper certificates remain useful for some predeclared tolerances. None reaches the trivial loss bound, and 21 of 26 selected forecasts per policy meet the tolerance 0.50.")
    text(p,"The population class-optimal risks are unknown for these six laws. Independent test means below upper endpoints and above lower endpoints are consistency checks, not measurements of true coverage. The confidence claim follows from the independent calibration sampling and fixed-family allocation. It is separate from the approximate paired-t inference in the principal forecasting experiment.")
    text(p,"All 1,820 development fits met the stated optimization criterion. The independent analysis checked 624 decoder budgets and reconstructed all 624 pair-witness score families. Full prediction-to-loss arrays were retained for the prespecified first repetition of each law; every repetition retained the score-level information needed to recompute its designated endpoints.")

    p=page("5. Experimental validity and scientific implications")
    table(p,["Design component","Principal forecasting","Certificate workflow"],[
        ["Independent repetitions","24 complete four-law repetitions","26 repetitions over six laws"],
        ["Data roles","A/B/C/D: 64/32/32/128 episodes","Train/validation/calibration/test: 128/512/4096/2048"],
        ["Dimension choice","One B-selected cap per geometric method","One validation-selected history/cap bank per policy"],
        ["Readout selection","B penalty; C blending and procedure policy","Best validation prefix; fixed feasible block budget"],
        ["Terminal uncertainty","Six approximate paired-t margin bounds","4,680 finite-sample one-sided statements"],
        ["Primary scope","Four existing numerical laws and two noise levels","Six bounded synthetic laws; block-direct decoder"],
        ["Independent numerical audit","192 tasks; 31,104 arrays; 27 tables","26 repetitions; 4,680 endpoints; nine tables"]],[.26,.37,.37])
    text(p,"Native dimension requests use min(cap, ambient dimension, local reference count minus one), followed by positive-eigenvalue and projected-range checks. Of 2,688 cap-twenty local models in the principal study, 2,685 retain rank 12 and three have initial admissible rank one. Every historical cap-two bank has a stacked basis of full input rank. Increasing local rank therefore does not simply restore information lost by a global rank-two projection.")
    text(p,f"No scheduled forecasting task failed, and no query recovery, reconstruction fallback, factor failure or deterministic bound violation occurred. The largest selected-KAHM-to-ridge maximum-absolute-error ratio is {fs['maximum_selected_error_ratio']:.3f}, below five. Gaussian observations in this study do not satisfy the bounded-observation assumptions of the separate certificate experiment.")
    text(p,"Interpretation","subheading")
    text(p,"Validation-based dimension selection improves aggregate accuracy; uniformly increasing the cap is unsupported. The primary evidence establishes benefits over common features and raw RBF regression, together with linear-control noninferiority. The remaining comparisons limit broader superiority claims. The matched-bank decomposition attributes the dimension-selection gain mainly to improved common features, while observed accuracy gains leave selected-forecast certificate decisions unchanged.")
    text(p,"The oscillator results motivate studying when KAHM departures retain a material advantage over equally tuned RBF departures under observation noise. This is a hypothesis for further investigation. The experiments use fresh realizations of existing synthetic laws and provide no new application cohort. The targeted certificate experiment also leaves other decoder settings and the original two-dimensional certificate mechanism outside its scope.")
    text(p,"Evidence and reproduction","subheading")
    text(p,"Forecasting tables: extensions/dimension_forecasting/analysis. Certificate tables: extensions/dimension_certificates/runs/20261101/analysis. Each extension includes an independent audit in audit/post_release. Raw numerical records accompany the source and compact tables. Source, fitted-state, prediction or calibration, and target-release identities were verified before numerical reanalysis. The protocols were frozen locally before production, without public preregistration.","caption")

    spec={"title":"Subspace selection in KAHM forecasting","running_title":"KAHM forecasting: subspace selection and risk certificates","pages":pages,
        "evidence":{"forecast_summary_sha256":hashlib.sha256((forecast/"analysis/analysis_summary.json").read_bytes()).hexdigest(),
                    "forecast_audit_sha256":hashlib.sha256((forecast/"audit/post_release/final_results_audit.json").read_bytes()).hexdigest(),
                    "risk_summary_sha256":hashlib.sha256((run/"analysis/summary.json").read_bytes()).hexdigest(),
                    "risk_audit_sha256":hashlib.sha256((risk/"audit/post_release/final_results_audit.json").read_bytes()).hexdigest()}}
    destination=forecast/"reporting/report_specification.json";destination.write_text(json.dumps(spec,indent=2,ensure_ascii=False)+"\n")
    return destination


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("forecast",type=Path);parser.add_argument("risk",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args();print(build(args.forecast,args.risk,args.output))
