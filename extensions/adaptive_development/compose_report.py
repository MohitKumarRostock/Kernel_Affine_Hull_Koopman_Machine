"""Compose post-run interpretation after the frozen analysis and audit finish.

No estimator, evaluation rule or development gate is modified here. Any extra
peak-error summaries are explicitly descriptive post-run diagnostics.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import numpy as np

HERE=Path(__file__).resolve().parent
METHODS=["ridge","fixed_kahm","adaptive_kahm","adaptive_affine","adaptive_rbf","kernel_rr"]
LABELS={"ridge":"Raw ridge","fixed_kahm":"Fixed KAHM","adaptive_kahm":"Adaptive KAHM",
        "adaptive_affine":"Adaptive affine","adaptive_rbf":"Adaptive RBF-local","kernel_rr":"Sliding-window KRR"}
COLLECTIONS=["synthetic_stationary","synthetic_recurring","dominick_weekly","sunspot_daily","natural_equal_collection"]
COLLECTION_LABELS=["Stationary","Recurring","Dominick","Sunspot","Natural mean"]


def load_csv(name):
    with (HERE/"analysis"/name).open(newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    return f"{x:.3e}" if abs(x)>=10000 else f"{x:.4f}"


def block(kind,text):
    return {"type":kind,"text":text}


def main():
    decision=json.loads((HERE/"analysis"/"decision.json").read_text())
    audit=json.loads((HERE/"analysis"/"audit.json").read_text())
    assert decision["decision"]=="DEFER" and decision["technical_validity"], "Interpretation requires review for this decision."
    assert audit["status"]=="passed" and audit["decision"]==decision["decision"]
    assert decision["completed_runs"]==144 and decision["scored_forecasts"]==387900
    assert decision["framework_errors"]==0 and decision["framework_fallback_forecasts"]==0 and not decision["cap_violations"]
    assert all(not r["mechanism_gate"] and not r["natural_gate"] for r in decision["capacity_tiers"]), "Revise categorical gate interpretation."
    cm=load_csv("collection_metrics.csv")
    tm=load_csv("task_metrics.csv")
    comparisons=load_csv("comparisons.csv")
    c={(r["budget"],r["collection"],r["method"]):r for r in cm}
    cmp={(r["budget"],r["collection"],r["comparator"]):r for r in comparisons}
    loss=lambda b,s,m:float(c[(b,s,m)]["mean_scaled_mse"])
    gain=lambda b,s,m:float(cmp[(b,s,m)]["kahm_gain_percent"])
    tier={r["budget"]:r for r in decision["capacity_tiers"]}
    active=lambda b,s:int(c[(b,s,"adaptive_kahm")]["active_horizons"])
    total=lambda b,s:int(c[(b,s,"adaptive_kahm")]["horizon_count"])
    peaks=[]
    for method in ("adaptive_kahm","adaptive_affine"):
        path=HERE/"results"/f"synthetic_recurring__20261101__large__{method}"/"forecasts.npz"
        with np.load(path) as a:
            use=a["evaluation"].astype(bool)&np.isfinite(a["truth"])
            index=np.flatnonzero(use)
            error=a["prediction"][use]-a["truth"][use]
            k=int(index[np.argmax(np.abs(error))])
            interval=use&(a["origin"]>=1547)&(a["origin"]<=1562)
            peaks.append({"method":method,"task_id":"synthetic_recurring__20261101","budget":"large",
                          "origin":int(a["origin"][k]),"horizon":int(a["horizon"][k]),
                          "prediction":float(a["prediction"][k]),"truth":float(a["truth"][k]),
                          "baseline":float(a["baseline"][k]),"correction":float(a["correction"][k]),
                          "gamma":float(a["gamma"][k]),"absolute_error":float(abs(a["prediction"][k]-a["truth"][k])),
                          "evaluation_truth_min":float(np.min(a["truth"][use])),
                          "evaluation_truth_max":float(np.max(a["truth"][use])),
                          "sse_fraction_in_posthoc_interval_1547_1562":float(np.sum((a["prediction"][interval]-a["truth"][interval])**2)/np.sum(error**2))})
    (HERE/"analysis"/"post_run_diagnostics.json").write_text(json.dumps({"scope":"Descriptive post-run diagnostics; no forecasts removed or advancement rules changed.","peaks":peaks},indent=2)+"\n")
    kp=peaks[0]
    gates=[["Capacity tier","Recurring gate","Natural gate","p99 screen","Decision"]]
    for b,label in [("small","64 ref cap / 128 replay"),("large","128 ref cap / 256 replay")]:
        g=tier[b]
        gates.append([label,"Pass" if g["mechanism_gate"] else "Fail","Pass" if g["natural_gate"] else "Fail",
                      ("Pass: " if g["p99_allowance_gate"] else "Fail: ")+f"{g['worst_kahm_stream_p99_ms']:.2f} ms",g["decision"]])
    pages=[]
    pages.append({"title":"KAHM: a focused<br/>development decision", "blocks":[
        block("small","Completed 31 October 2026 | Bounded online adaptation | Development evidence"),
        block("p","<b>Recommendation: defer the proposed adaptive-KAHM extension as evidence of forecasting effectiveness.</b> The repaired prototype fails the prespecified accuracy requirements at both capacity tiers. It does not support a manuscript claim that KAHM is an effective general solution for time-series forecasting."),
        {"type":"table","data":gates,"widths":[144,90,80,130,80]},
        block("p",f"The completed amended study contains <b>{decision['completed_runs']} runs and {decision['scored_forecasts']:,} scored forecasts</b>. All runs are technically valid: no unexpected model exceptions, no framework fallback forecasts and no observed capacity violations. The independent evidence audit passed {audit['checks']:,} checks."),
        block("p",f"The central obstacle is predictive value. On the recurring generator, large-tier adaptive KAHM has mean scaled MSE <b>{num(loss('large','synthetic_recurring','adaptive_kahm'))}</b>, versus {num(loss('large','synthetic_recurring','ridge'))} for raw ridge and {num(loss('large','synthetic_recurring','kernel_rr'))} for kernel ridge. The real-series results provide no consistent KAHM advantage over all controls."),
        {"type":"figure","file":"collection_errors.png","caption":"All six methods are retained. Lower scaled MSE is better. The recurring panel uses a logarithmic axis because of extreme adverse forecasts; the two natural panels use linear axes. Small/large labels give reference and replay quotas."},
        block("small","One complete earlier attempt required a numerical amendment. Its source, failures and results are preserved in the package. The repeat is transparent development work after observed outcomes, not independent confirmation. Evidence: analysis/decision.json, collection_metrics.csv and audit.json.")
    ]})
    pages.append({"title":"1. Question, design and controls", "blocks":[
        block("p","The development question was whether a bounded adaptive KAHM representation improves forecasting performance beyond the same online update policy with simpler geometry, and beyond a practical kernel comparator. A positive answer would justify committing newly reserved data to a larger manuscript extension."),
        {"type":"table","data":[["Development inputs","Streams","Length each","History / horizons"],["Stationary nonlinear AR","4 seeds","2,048","12 / 1, 4, 12"],["Recurring A/B nonlinear AR","4 paired seeds","2,048","12 / 1, 4, 12"],["Dominick weekly SKU profit","3 preserved series","393","52 / 1, 2, 4"],["Sunspot daily","1 preserved series","4,096","28 / 1, 3, 7"]],"widths":[195,60,90,179]},
        block("p","The stationary and recurring synthetic streams share innovations within each of four seeds. The recurring schedule alternates two nonlinear regimes every 256 observations; the stationary schedule stays in regime A. Both regimes occur in the initial synthetic fitting prefix. Regime labels are retained for analysis and never supplied to a forecaster. The exact equation, seeds and burn-in are in PROTOCOL.md."),
        block("h2","Six methods and two capacity tiers"),
        block("p","Raw ridge is the common sliding-window baseline. Fixed KAHM retains its initial geometry while updating its heads. Adaptive KAHM updates geometry and heads. Adaptive affine and adaptive RBF-local use the same reference selection and refresh policy with different geometric features. Sliding-window Gaussian kernel ridge uses recursive add/delete inverse updates and a free intercept; it is not a reproduction of published KRLS-T."),
        block("p","The small tier permits 64 shared geometric references and 128 replay rows per horizon; the large tier permits 128 and 256. There are four local models of rank at most two. Every 16 observations, an adaptive arm can refresh one local model, retaining half its initial references as anchors and replacing at most four recent references. It rebuilds the affected geometry and all residual heads coherently. Kernel ridge has up to 64 or 128 supports per horizon. These quotas are not equal byte allocations."),
        block("h2","Strict target chronology and limited calibration"),
        block("p","The first 35% of each series initializes the model; 35-50% is chronological calibration; the latter half is evaluation. Each observation first scores previously issued forecasts, then releases newly matured labels, updates the model and issues immutable future predictions. At evaluation start, each horizon selects a mixing weight from 0, 0.25, 0.5 and 1 using only calibration forecasts whose targets have already arrived. The weight stays fixed thereafter; the underlying model may continue learning from matured evaluation labels."),
        block("p","No bandwidth, penalty, history, reference-policy or refresh search was performed. Dominick and Sunspot were deliberately chosen from previously inspected manuscript inputs, including contrasting earlier KAHM outcomes. Their reuse supports development decisions, not new-source confirmation."),
        block("small","Evidence: PROTOCOL.md, config.json, DATA_PROVENANCE.md, prepared_inputs/manifest.json and run.py. The original and amended scientific source checkpoints are local, not public preregistration.")
    ]})
    pages.append({"title":"2. Forecasting results", "blocks":[
        block("p","Primary loss is MSE divided by the mean squared seasonal difference in the initial fitting prefix. Average origins within horizon, horizons within series, then series within collection. The natural mean weights Dominick and Sunspot equally; synthetic conditions remain separate. Smaller is better; bold marks the best value, including numerical ties. This online schedule and scale differ from the earlier manuscript tables, so their numerical scores are not directly comparable."),
        block("h2","Small tier: 64 references and 128 replay rows"),
        {"type":"accuracy","budget":"small"},
        block("h2","Large tier: 128 references and 256 replay rows"),
        {"type":"accuracy","budget":"large"},
        block("p",f"On the equal-collection natural mean, adaptive KAHM is {-gain('small','natural_equal_collection','kernel_rr'):.2f}% worse than kernel ridge in the small tier and {-gain('large','natural_equal_collection','kernel_rr'):.2f}% worse in the large tier. Its gains over adaptive RBF-local are only {gain('small','natural_equal_collection','adaptive_rbf'):.3f}% and {gain('large','natural_equal_collection','adaptive_rbf'):.3f}%, respectively, below the 1% natural-aggregate threshold."),
        block("p",f"The large-tier Dominick result is {gain('large','dominick_weekly','ridge'):.3f}% better than raw ridge and {gain('large','dominick_weekly','adaptive_affine'):.3f}% better than adaptive affine, but {-gain('large','dominick_weekly','kernel_rr'):.2f}% worse than kernel ridge. This is insufficient evidence of a useful KAHM-specific advantage."),
        block("p",f"Calibration activates the KAHM correction on only {active('small','natural_equal_collection')}/{total('small','natural_equal_collection')} natural-data horizon/series combinations in the small tier and {active('large','natural_equal_collection')}/{total('large','natural_equal_collection')} in the large tier. On Sunspot, every KAHM horizon selects zero correction at both tiers. Those forecasts reproduce the common baseline to numerical precision and do not demonstrate added value from KAHM."),
        block("small","All per-series and per-horizon outcomes, calibration weights, resources and comparisons are retained in analysis/task_metrics.csv, horizon_metrics.csv, collection_metrics.csv and comparisons.csv. A value below one compares loss with the training-prefix scale; it does not by itself establish superiority over an evaluated seasonal-naive forecast. No window-level significance tests or broad transfer confidence intervals are claimed.")
    ]})
    pages.append({"title":"3. Why this prototype failed the recurring screen", "blocks":[
        {"type":"figure","file":"recurring_seed_results.png","caption":"All four recurring seeds and all six methods, after the numerical repair. The shared vertical axis is logarithmic when needed. Forecast origins are dependent; the four seeds are the replication units. No unfavorable seed is omitted."},
        block("p",f"In the small tier, KAHM activates {active('small','synthetic_recurring')}/{total('small','synthetic_recurring')} recurring-condition corrections. A method that selects a zero contribution can recover the baseline but cannot establish the proposed geometric advantage. In the large tier, {active('large','synthetic_recurring')}/{total('large','synthetic_recurring')} corrections are active, but severe errors occur after regime changes."),
        block("h2","A concrete failure, retained in the evaluation"),
        block("p",f"For recurring seed 20261101, the largest amended KAHM forecast error occurs at origin {kp['origin']}, horizon {kp['horizon']}. The forecast is <b>{kp['prediction']:,.2f}</b> while the observed target is <b>{kp['truth']:.4f}</b>; the raw baseline component is {kp['baseline']:.4f}. Evaluation targets for this stream range from {kp['evaluation_truth_min']:.3f} to {kp['evaluation_truth_max']:.3f}. The burst is a severe forecasting failure relative to the scale of the observed series."),
        block("p","The preserved diagnostic replay state (large KAHM, seed 20261101, origin 1554, horizon 12) explains why a numerical repair alone is insufficient. Its query's largest standardized feature is about 3.20 million, compared with about 392 in the current supervised replay. A fresh, correctly solved ridge objective still produces a correction of about 175,556. The adaptive affine arm also exhibits severe extrapolation under a fresh solve."),
        block("p","These observations are consistent with extrapolation arising from the changing representation and delayed-label replay: a current query can lie far outside the feature range represented by its matured long-horizon training rows. Isolating the contribution of scaling, label delay and reference changes would require a dedicated intervention or ablation. The KAHM signed-denominator check is safe at the preserved diagnostic state, so that particular guard does not prevent the failure."),
        block("small","These mechanism descriptions are post-run diagnostics, not additional advancement criteria. No clipping, normalization change, gamma retuning or error deletion was used. Evidence: AMENDMENT_DIAGNOSTICS.py/.json/.md, original preserved forecasts and analysis/post_run_diagnostics.json.")
    ]})
    krows=[r for r in tm if r["method"]=="adaptive_kahm"]
    overruns=sum(int(r["allowance_overruns"]) for r in krows)
    observations=sum(int(r["evaluation_observations"]) for r in krows)
    pages.append({"title":"4. Runtime and bounded state", "blocks":[
        block("p","The table reports the worst observed value across the 12 streams for each method/tier. A row's timing and memory maxima can come from different streams. The p99 is computed per stream before taking the worst value; it is not a pooled percentile that can hide a slow stream."),
        {"type":"resources"},
        block("p",f"Adaptive KAHM's worst per-stream p99 is {tier['small']['worst_kahm_stream_p99_ms']:.2f} ms in the small tier and {tier['large']['worst_kahm_stream_p99_ms']:.2f} ms in the large tier. Across its {observations:,} evaluation observation steps, {overruns:,} service intervals exceed 20 ms. The criterion concerns p99 on every stream; it does not require every single step to be below the allowance."),
        block("p","Service time includes history/pair assembly, all matured-horizon updates, geometry refreshes, feature/head rebuilds, kernel inverse audits and prediction. Initialization and calibration costs are recorded separately. Scoring, evidence serialization and state inspection are excluded. Workers run serially in fresh processes with one numerical thread and the same CPU affinity; detailed runtime/library metadata are retained in every summary."),
        block("p","Owned NumPy memory is the largest sampled total of unique model-owned array buffers. It excludes Python object overhead, temporary work arrays and the external evidence queues. Whole-worker peak RSS includes the interpreter, numerical libraries, input arrays and evidence storage. These are complementary measurements, neither a precise peak incremental deployment footprint."),
        block("p","The two capacity tiers are quotas, not equal memory or equal computational work. Kernel supports are separate for each horizon; the geometric bank is shared. Adaptive geometry remains active even when calibration selects a zero correction, so these are costs of the measured eager prototype. A disabled module could cost less. The 20 ms allowance is a monitored feasibility screen on this host, not an enforced scheduling guarantee or an application-derived service-level requirement."),
        block("small","Evidence: analysis/task_metrics.csv and each results/*/summary.json and trace.json. Complete initialization, calibration, mean/median/p95/p99/maximum service, update and prediction timings are retained.")
    ]})
    pages.append({"title":"5. Numerical amendment and audit", "blocks":[
        block("p","A clean scientific interpretation required distinguishing numerical implementation failure from a poor fitted forecasting function. The first complete attempt was preserved before editing any frozen source. Only the correction-head solver was amended; the complete roster was then repeated."),
        {"type":"table","data":[["Evidence item","Original attempt","Amendment A1 repeat"],["Completed method/tier/stream runs","144","144"],["Scored forecasts","387,900","387,900"],["Unexpected model exceptions","4",str(decision['framework_errors'])],["Framework fallback forecasts","8,659",str(decision['framework_fallback_forecasts'])],["Observed cap violations","0",str(len(decision['cap_violations']))],["Independent consistency audit","Passed; 6,704,494 checks",f"Passed; {audit['checks']:,} checks"],["Prespecified decision","TECHNICAL_BLOCK",decision['decision']]],"widths":[220,152,152]},
        block("h2","Repairing arithmetic without improving the model by decree"),
        block("p","The original failures were in correction-head normal equations. In a reproduced failing state, accumulated and freshly reconstructed feature cross-products differed by about 2.70 billion. The accumulated ridge matrix had a negative eigenvalue, while the fresh matrix's smallest eigenvalue was approximately one. Large standardized rows entering and leaving a finite window had caused subtractive cancellation."),
        block("p","A1 recomputes current-baseline residuals from the same bounded replay, centers features and targets to eliminate the free intercept, and solves the augmented ridge objective with column equilibration and pivoted QR. The penalty rows receive the same variable transformation. The objective and alpha remain unchanged. The same implementation serves all four geometric arms; raw ridge, scalers, geometry, kernel settings, reference policy, gamma grid, chronology and decision rules are unchanged."),
        block("p","All 13 numerical and chronology tests passed before the repeat. They include agreement with independent batch objectives, stable eviction of huge transient features, unchanged raw baseline coefficients, coherent geometry rebuilds, matured-target checks and invariance of earlier forecasts to perturbations of later observations. Independent diff review found no blocking defect."),
        block("p",f"The saved-evidence audit reconstructs target values, forecast coverage, target timing, calibration selection, losses, aggregates, decision gates, timing summaries, capacity samples, shared baselines and matched adaptive reference updates. Its maximum shared-baseline discrepancy is {audit['maximum_shared_baseline_absolute_difference']:.3g}. It imports neither the forecasting model nor the original analyzer. This is independent computational consistency checking within the study, not external scientific replication."),
        block("small","All first-attempt records remain in archives/initial_attempt.zip. A1 was made after observing those outcomes; the repeat is not a new independent experiment. Counting two executions as 288 independent experimental units would be incorrect. Evidence: AMENDMENT_A1.md, AMENDMENT_VALIDATION.json, AMENDMENT_TEST_LOG.txt, AMENDMENT_DIAGNOSTICS.* and analysis/audit.json.")
    ]})
    pages.append({"title":"6. Decision for the manuscript", "blocks":[
        block("p","<b>Do not present this adaptive extension as support for a broad forecasting-effectiveness claim.</b> Both complete capacity tiers fail the recurring-mechanism and natural-data accuracy gates. A useful runtime result cannot compensate for the lack of comparative predictive value and the severe extrapolation bursts."),
        block("h2","The fixed advancement requirements"),
        block("p","The recurring condition required at least 3% mean loss reduction versus each of adaptive affine, adaptive RBF-local and kernel ridge, with wins on at least three of four paired seeds against each; it also required at least 1% gain versus fixed KAHM and raw ridge. The natural condition required at least 1% gain on the equal-collection aggregate against each main comparator, at least 3% against all three on one same collection, and no collection more than 5% worse against any of them. KAHM also needed p99 service at or below 20 ms on every stream, with no unexpected exception or cap violation in any method."),
        block("p","These are conservative development triage thresholds, not a calibrated hypothesis test or a universal definition of practical effectiveness. They were fixed before the original run and retained through A1. The current decision is DEFER because the synthetic mechanism gate fails; it is no longer a technical block."),
        {"type":"table","data":[["Manuscript choice","Recommendation"],["Lead claim: KAHM is generally effective for forecasting","Unsupported by this pilot. Do not add it on this evidence."],["Adaptive KAHM as a new main contribution","Defer until a revised method demonstrates stable, useful gains over the matched controls."],["Exploratory appendix or limitations section","Appropriate if clearly labeled as development evidence and both attempts are disclosed."],["Further work if pursued","First test a justified stability intervention; reserve independent sources only after the development protocol is settled."]],"widths":[220,304]},
        block("h2","Suggested evidence-matched wording"),
        block("p","<i>A focused development study evaluated bounded online adaptation of KAHM against a common ridge baseline, fixed KAHM, matched adaptive geometric controls and sliding-window kernel ridge. After a documented numerical correction that preserved the fitted objective, adaptive KAHM did not satisfy the prespecified accuracy criteria. Its correction was frequently suppressed on the selected natural series, while recurring-regime streams exposed severe extrapolation. These findings motivate further work on stability and task-specific benefit; they do not establish general forecasting effectiveness.</i>"),
        block("h2","What this result can and cannot rule out"),
        block("p","The conclusion concerns this fixed, simpler adaptive prototype. It does not disprove KAHM as a representation or invalidate earlier forecast-trained experiments. Only one generator, four paired seeds and two previously inspected natural collections were used. No broad hyperparameter optimization was attempted. The prototype does not repeatedly optimize the earlier manuscript's forecast-trained kernel/gate parameters. A different stable design may perform better, but that remains an untested hypothesis."),
        block("small","A proposed LaTeX appendix subsection is included as manuscript_addendum.tex. It is a draft for review, not a claim that the main manuscript has been revised or that an independent confirmation study has been completed.")
    ]})
    pages.append({"title":"7. Reproduction and provenance", "blocks":[
        block("h2","What the package contains"),
        block("p","The ZIP contains this report; the frozen implementation, protocol, configuration and tests; 12 prepared input archives; the four original natural-data NPZs and source manifest needed for optional regeneration; all amended issued forecasts, traces and summaries; analysis CSVs and JSON; independent audit; original-attempt archive; executable numerical diagnostics; the optional LaTeX subsection; report/figure generators; licenses; and a per-file size/SHA-256 inventory."),
        block("h2","Inspect without refitting"),
        block("p","Extract the package and use the directory containing its top-level README as the working directory. Install the pinned numerical dependencies from extensions/adaptive_development/requirements.txt in a suitable Python environment. To independently recompute the saved tables and decision, run:"),
        block("small","<font name='Courier' size='7.2'>python -m extensions.adaptive_development.analyze<br/>  --output /tmp/kahm_analysis<br/><br/>python -m extensions.adaptive_development.audit<br/>  --analysis /tmp/kahm_analysis<br/>  --output /tmp/kahm_analysis/audit.json</font><br/>The wrapped lines above show command arguments; use one shell command per invocation. Complete copyable commands are in the package README."),
        block("p","For a full forecast replay, use the extension README commands with a fresh output directory. Preserve the supplied results. Numerical libraries and CPU dispatch can cause small floating-point differences; timing depends on the host and scheduling. Replaying saved evidence is computational verification, not an additional independent scientific replication. Optional --prepare regeneration uses the included original natural archives and deterministic synthetic generator."),
        block("h2","Source identities and amendment trail"),
        block("small","Upstream parent commit: 88e9c568ee983856804a2b2da41f6290f259b8cd<br/>Original local freeze commit: 3d374de47d694f6ae3267e4b6ac82787755eeca5<br/>A1 local freeze commit: 6027eb995ca91adeab23711b71a3ae1790c9654e<br/>Original archive SHA-256 is retained in archives/initial_attempt_record.json.<br/>A1 freeze SHA-256: 140085b63b4af295cb6f631125e6a0c74aec84b7b2708c4fbed3fe21a98b35af<br/>RELEASE.json records the final local release identity. No public branch publication was performed."),
        block("h2","Data sources, transformations and licenses"),
        block("p","[1] Godahewa et al., <i>Dominick Dataset</i>, version 2, Monash Time Series Forecasting Repository / Zenodo. DOI: <link href='https://doi.org/10.5281/zenodo.4654802' color='#245b89'>10.5281/zenodo.4654802</link>. Original provider: University of Chicago Kilts Center. These are weekly retailer SKU profit series. This study reuses preserved series T7830, T25851 and T108533."),
        block("p","[2] Godahewa et al., <i>Sunspot Dataset (with Missing Values)</i>, version 4 (daily series), Monash / Zenodo. DOI: <link href='https://doi.org/10.5281/zenodo.4654773' color='#245b89'>10.5281/zenodo.4654773</link>. Original provider: SILSO. The preserved final 4,096-observation crop of T1 starts at original raw index 69,828. The selected crop is finite; no interpolation is performed."),
        block("p","The pinned archive records declare CC BY 4.0. Source checksums, creator metadata, prior selection and transformation notices are retained in DATA_PROVENANCE.md and the original source manifest/README. Code follows the repository MIT license (Mohit Kumar, 2026); DATA_LICENSE governs original research data and non-code outputs with its third-party exclusions. No dependency source code is bundled."),
        block("small","AI tools assisted implementation, checks, analysis and report preparation. The report documents observed evidence and limitations; it is not a submitted manuscript or an external independent replication.")
    ]})
    (HERE/"report_content.json").write_text(json.dumps({"pages":pages},indent=2)+"\n")

    def plain(text):
        text=re.sub(r"<b>(.*?)</b>",r"**\1**",str(text))
        text=text.replace("<br/>"," ")
        return re.sub(r"<[^>]+>","",text)
    def mdtable(data):
        data=[[plain(x).replace("|","/") for x in row] for row in data]
        return "\n".join(["| "+" | ".join(data[0])+" |","| "+" | ".join(["---"]*len(data[0]))+" |"]+["| "+" | ".join(row)+" |" for row in data[1:]])
    markdown=[]
    for i,page in enumerate(pages):
        markdown.append(("# " if i==0 else "## ")+plain(page["title"]))
        for item in page["blocks"]:
            kind=item["type"]
            if kind in ("p","small","h2"):
                markdown.append(("### " if kind=="h2" else "")+plain(item["text"]))
            elif kind=="table":
                markdown.append(mdtable(item["data"]))
            elif kind=="accuracy":
                b=item["budget"]
                markdown.append(mdtable([["Method"]+COLLECTION_LABELS]+[[LABELS[m]]+[num(loss(b,s,m)) for s in COLLECTIONS] for m in METHODS]))
            elif kind=="resources":
                data=[["Method / tier","Worst p99 ms","Maximum ms","Max sampled owned NumPy KiB","Worker RSS MiB"]]
                for m in METHODS:
                    for b in ("small","large"):
                        r=[x for x in tm if x["method"]==m and x["budget"]==b]
                        data.append([LABELS[m]+" / "+b,f"{max(float(x['p99_service_ms']) for x in r):.2f}",f"{max(float(x['max_service_ms']) for x in r):.2f}",f"{max(float(x['max_observed_owned_numpy_bytes']) for x in r)/1024:.1f}",f"{max(float(x['process_peak_rss_kib']) for x in r)/1024:.1f}"])
                markdown.append(mdtable(data))
            elif kind=="figure":
                markdown.append(f"![{item.get('caption','Study figure')}](analysis/figures/{item['file']})")
    (HERE/"FINDINGS.md").write_text("\n\n".join(markdown)+"\n")
    def texnum(x):
        if abs(x)<10000:
            return f"{x:.4f}"
        mantissa,exponent=f"{x:.3e}".split("e")
        return "$"+mantissa+r"\times 10^{"+str(int(exponent))+"}$"
    tex=[r"% Optional exploratory appendix subsection. Review before inclusion.",
         r"% This file uses standard LaTeX table commands; no main manuscript is overwritten.",
         r"\section{Focused development study of bounded adaptive KAHM}",
         r"\label{sec:adaptive-development}",
         "We evaluated a bounded online KAHM prototype against a common raw-history ridge baseline, fixed KAHM, matched adaptive affine and RBF-local controls, and sliding-window Gaussian kernel ridge. The study used four paired stationary/recurring synthetic seeds, three previously inspected Dominick series, and one previously inspected Sunspot series, at two capacity tiers. These are reference/replay quotas, not equal-byte allocations. This is development evidence rather than independent source confirmation. Kernel bandwidths and gate temperatures remain fixed; this simpler prototype is not a streaming reproduction of the earlier forecast-trained model.",
         "The original complete attempt encountered four correction-head numerical failures. A documented amendment replaced accumulated correction normal equations with a stable direct solve of the same bounded-replay, free-intercept ridge objective. All other modeling, calibration, chronology and advancement rules were retained, and the complete 144-run roster was repeated. Both attempts are preserved. The amended evaluation contains 387,900 scored forecasts, with no unexpected model exceptions or capacity violations.",
         "Loss is MSE scaled by the initial-prefix mean squared seasonal difference, averaged across horizons and then streams. The natural aggregate gives the two collections equal weight. All forecasts were issued causally; horizon mixing weights were selected from a fixed calibration grid and then held fixed."]
    for b,label in [("small","64 references; 128 replay rows"),("large","128 references; 256 replay rows")]:
        table_lines=[r"\begin{table}[t]",r"\centering",r"\small",r"\caption{Amended development results ("+label+r"). Lower scaled MSE is better.}",r"\label{tab:adaptive-"+b+r"}",r"\begin{tabular}{lrrrr}",r"\hline",r"Method & Recurring & Dominick & Sunspot & Natural mean \\",r"\hline"]
        for m in METHODS:
            table_lines.append(LABELS[m]+" & "+" & ".join(texnum(loss(b,s,m)) for s in COLLECTIONS[1:])+r" \\")
        table_lines += [r"\hline",r"\end{tabular}",r"\end{table}"]
        tex.append("\n".join(table_lines))
    tex.append("Adaptive KAHM did not satisfy the prespecified accuracy criteria at either tier. On the natural data, calibration frequently suppressed its correction; recurring regimes exposed severe extrapolation that persisted under a fresh correctly solved ridge objective. These findings motivate further work on stability and task-specific benefit, but do not establish general forecasting effectiveness. The conclusion concerns this fixed prototype and does not rule out other forecast-trained or adaptive KAHM formulations.")
    (HERE/"manuscript_addendum.tex").write_text("\n\n".join(tex)+"\n")
    (HERE/"manuscript_addendum_standalone.tex").write_text("\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n\\input{manuscript_addendum.tex}\n\\end{document}\n")
    final_validation={"decision":decision["decision"],"completed_runs_per_attempt":144,"complete_attempts":2,
                      "scored_forecasts_per_attempt":387900,"amended_technical_validity":decision["technical_validity"],
                      "amended_framework_errors":decision["framework_errors"],"amended_framework_fallback_forecasts":decision["framework_fallback_forecasts"],
                      "amended_cap_violations":decision["cap_violations"],"pre_repeat_tests":13,"audit_status":audit["status"],
                      "audit_checks":audit["checks"],"scientific_scope":"Post-observation development amendment; no independent confirmation.",
                      "presentation_scope":"Figures, report and peak-error summaries are post-run descriptive work; frozen decision and scores are unchanged."}
    (HERE/"FINAL_VALIDATION.json").write_text(json.dumps(final_validation,indent=2)+"\n")
    print(json.dumps({"decision":decision["decision"],"report_pages_planned":len(pages),"files":["FINDINGS.md","report_content.json","manuscript_addendum.tex","FINAL_VALIDATION.json"],"peaks":peaks},indent=2))


if __name__=="__main__":
    main()
