"""One frozen primary pass through all 54 decision-cohort decoder classes."""
import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_name]="1"
from datetime import datetime,timezone
from fractions import Fraction as F
import json
from pathlib import Path
import time
import traceback
import data37 as d
import full_graph37 as g

ROOT=Path(__file__).resolve().parents[1]

def binding(task,info,mapping):
    return {"input_manifest_sha256":d.sha_file(ROOT/d.MANIFEST_PATH),
        "cohort_path":info["cohort"]["path"],"cohort_sha256":info["cohort"]["sha256"],
        "feature_path":mapping["feature_path"],"feature_sha256":mapping["feature_sha256"],
        "map_payload_sha256":mapping["map_payload_sha256"],
        "prior_model_seal_sha256":d.sha_file(ROOT/"inputs/prior/model_seal36.json")}

def summarize(record,info,mapping,relative):
    return {"task":record["task"],"map":record["map"],"family":mapping["family"],
        "scaling":mapping["scaling"],"dimension":mapping["dimension"],"n":record["n"],
        "feature_id":record["feature_id"],"class_id":record["class_id"],
        "epsilon":record["epsilon"],"epsilon_physical_mse":str(info["epsilon_physical_mse"]),
        "physical_mse_multiplier":str(info["physical_mse_multiplier"]),
        "class_risk":record["class_risk"],"range_floor_mean":record["range_floor"]["mean"],
        "new_primal_mean":record["new_primal_mean"],"diagnosis":record["diagnosis"],
        "gap_resolved":record["gap_resolved"],"existing_eligible_head_count":record["existing_eligible_head_count"],
        "new_finite_cohort_adequacy":record["new_finite_cohort_adequacy"],
        "solver_status":record["numerical"]["status"],"solver_iterations":record["numerical"]["iterations"],
        "solver_fallback_reason":record["numerical"]["fallback_reason"],
        "pair_count":record["distance"]["pair_count"],"nonredundant_pair_count":len(record["distance"]["nonredundant_pair_indices"]),
        "zero_distance_pair_count":record["distance"]["zero_distance_pair_count"],
        "nonzero_dual_count":len(record["dual"]["signed_multipliers"]),
        "selected_primal_source":record["selected_primal_source"],"timing":record["timing"],
        "receipt_path":relative,"receipt_sha256":d.sha_file(ROOT/relative)}

def main():
    started=time.perf_counter();protocol=d.verify_protocol_seal()
    marker=ROOT/"output/primary_started37.json"
    if marker.exists() or (ROOT/"output/result37.json").exists():raise RuntimeError("Step37 primary cannot be rerun")
    marker.parent.mkdir(parents=True,exist_ok=True)
    with marker.open("xb") as handle:
        handle.write(d.canonical_bytes({"started_utc":datetime.now(timezone.utc).isoformat(),
            "protocol_seal_sha256":d.sha_file(ROOT/"protocol_seal37.json"),"single_primary_pass":True}))
    d.event("primary_started",primary_marker_sha256=d.sha_file(marker),
        protocol_seal_sha256=d.sha_file(ROOT/"protocol_seal37.json"),
        preexecution_gate_sha256=d.sha_file(ROOT/"audit/preexecution_gate37.json"))
    rows=[];head_rows=[];error=None;status="COMPLETE";task_meta={}
    ordered=[(task,name) for task in protocol["tasks"] for name in protocol["map_order"]]
    try:
        inputs=d.load_inputs()
        for task in protocol["tasks"]:
            info=inputs[task];targets=info["cohort"]["y"]
            task_meta[task]={"decision_n":len(targets),"epsilon":str(info["epsilon"]),
                "epsilon_physical_mse":str(info["epsilon_physical_mse"]),
                "physical_mse_multiplier":str(info["physical_mse_multiplier"]),
                "targets_outside_interval":sum(y<0 or y>1 for y in targets),
                "range_floor_mean":str(sum(((g.clipped(y)-y)**2 for y in targets),F(0))/len(targets))}
        for task,name in ordered:
            if time.perf_counter()-started>protocol["execution"]["global_wall_budget_seconds"]:
                status="BUDGET_STOP";break
            info=inputs[task];mapping=info["maps"][name];bound=binding(task,info,mapping)
            d.event("class_diagnostic_started",task=task,map=name,feature_id=mapping["feature_id"],class_id=mapping["class_id"])
            record=g.diagnose(mapping["z"],info["cohort"]["y"],ids=info["cohort"]["ids"],
                feature_id=mapping["feature_id"],class_id=mapping["class_id"],epsilon=info["epsilon"],
                heads=[mapping["heads"][recipe] for recipe in ("affine","extension32")],
                task=task,map_name=name,kappa=protocol["class"]["kappa"],
                settings=protocol["numerics"]["settings"],input_binding=bound)
            relative="output/classes/"+task+"/"+name+".json";d.write_json(ROOT/relative,record)
            row=summarize(record,info,mapping,relative);rows.append(row)
            for head in record["inherited_heads"]:
                head_rows.append({"task":task,"map":name,"family":mapping["family"],"scaling":mapping["scaling"],
                    "epsilon":record["epsilon"],"class_diagnosis":record["diagnosis"],
                    "class_risk":record["class_risk"],**head})
            d.event("class_diagnostic_completed",task=task,map=name,receipt_path=relative,
                    receipt_sha256=row["receipt_sha256"],diagnosis=record["diagnosis"])
            print(json.dumps({"completed":len(rows),"task":task,"map":name,"diagnosis":record["diagnosis"],
                "lower_over_epsilon":float(F(record["class_risk"]["lower"])/info["epsilon"]),
                "upper_over_epsilon":float(F(record["class_risk"]["upper"])/info["epsilon"]),
                "gap":float(F(record["class_risk"]["width"])),"solver":record["numerical"]["status"],
                "seconds":record["timing"]["total_seconds"]}),flush=True)
    except Exception as exc:
        status="STOPPED_ERROR";error={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        d.event("primary_stopped_error",error=error)
    completed={(row["task"],row["map"]) for row in rows}
    missing=[{"task":task,"map":name} for task,name in ordered if (task,name) not in completed]
    result={"schema":"step37-full-class-diagnostic-result-v1","step":37,"status":status,
        "finished_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":d.sha_file(ROOT/"protocol37.json"),
        "protocol_seal_sha256":d.sha_file(ROOT/"protocol_seal37.json"),
        "preexecution_gate_sha256":d.sha_file(ROOT/"audit/preexecution_gate37.json"),
        "input_manifest_sha256":d.sha_file(ROOT/d.MANIFEST_PATH),
        "input_access_sha256":d.sha_file(ROOT/"output/input_access37.json") if (ROOT/"output/input_access37.json").exists() else None,
        "prior":protocol["prior"],"task_metadata":task_meta,"class_rows":rows,"head_rows":head_rows,
        "completed_classes":len(rows),"expected_classes":54,"inherited_heads":len(head_rows),
        "missing_classes":missing,"error":error,"total_wall_seconds":time.perf_counter()-started,
        "scope":protocol["scientific_scope"],"new_prediction_model_fits":0,"new_forecast_predictions":0,
        "prior_test_results_unchanged":True,"reserved_patients_accessed":0}
    d.write_json(ROOT/"output/result37.json",result)
    d.event("primary_finished",status=status,completed_classes=len(rows),missing_classes=len(missing),
            result_sha256=d.sha_file(ROOT/"output/result37.json"))
    print(json.dumps({"status":status,"completed_classes":len(rows),"missing_classes":len(missing),
        "result_sha256":d.sha_file(ROOT/"output/result37.json"),"total_wall_seconds":result["total_wall_seconds"]}),flush=True)
    if status!="COMPLETE":raise SystemExit(2)

if __name__=="__main__":main()
