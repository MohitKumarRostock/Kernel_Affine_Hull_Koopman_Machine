"""Synthetic integration check: no public archive or model is opened."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from threadpoolctl import threadpool_limits
import run36 as run
import models36 as models
import protocol36
import triggered_policy36 as policy
import data36

ROOT=Path(__file__).resolve().parents[1]


def main():
    protocol=protocol36.draft()
    checks={}
    # Two maps share exact synthetic coordinates. They have deliberately different
    # encoder costs; after a verified eligible fit, ordinary cost pruning is tested.
    rng=np.random.default_rng(360001)
    ztrain=rng.normal(size=(32,3));zdecision=rng.normal(size=(6,3))
    train={"ids":[f"artificial-train:{i:03}" for i in range(32)],"y":[F(1,4)+F(i%3,20) for i in range(32)]}
    decision={"ids":[f"artificial-decision:{i:03}" for i in range(6)],"y":[F(1,4)+F(i%3,20) for i in range(6)]}
    anchors=np.arange(32)
    maps={"raw_artificial_a":{"feature_id":"artificial:a","payload_bytes":100,"numeric_bytes":24},
          "raw_artificial_b":{"feature_id":"artificial:b","payload_bytes":100000,"numeric_bytes":24}}
    features={name:{"train":ztrain,"decision":zdecision} for name in maps}
    bounds={}
    for name,info in maps.items():
        kw={"feature_id":info["feature_id"],"dimension":3,"map_payload_bytes":info["payload_bytes"]}
        bounds[name]={"affine":policy.head_cost_lower_bound("affine",**kw),
                      "extension32":policy.head_cost_lower_bound("extension32",**kw,anchor_features=ztrain.tolist(),anchor_ids=train["ids"])}
    order=policy.map_order({name:v["payload_bytes"] for name,v in maps.items()})
    # Exercise real fits, exact evaluators, digest retention, repeated-fit identity
    # and post-timing canonical comparison in an isolated temporary output tree.
    old_root,old_out=run.ROOT,run.OUT
    with tempfile.TemporaryDirectory(prefix="artificial-run36-") as td:
        run.ROOT=Path(td);run.OUT=run.ROOT/"output"
        try:
            args=(order,maps,features,bounds,train,decision,anchors,F(1),protocol)
            e,eb,ec=run.timed_policy("artificial","exhaustive",0,*args)
            run.retain_execution(e,eb,ec)
            canonical={row["candidate_id"]:row for row in e["candidate_rows"]}
            checks["exhaustive_real_fits_all_four"]=(e["counts"]["fit_calls"]==4)
            checks["exhaustive_trace_digest_unchanged_after_retention"]=(models.object_sha(e["trace"])==e["trace_sha256"])
            executions=[]
            for mode in ("ordinary","triggered"):
                record,bundles,certs=run.timed_policy("artificial",mode,0,*args)
                path=run.retain_execution(record,bundles,certs,canonical)
                saved=json.loads((run.ROOT/path).read_text())
                checks[mode+"_same_choice"]=(record["selection"]==e["selection"])
                checks[mode+"_cost_pruning"]=(record["counts"]["cost_pruned_heads"]>=2)
                checks[mode+"_trace_digest_unchanged"]=(models.object_sha(saved["trace"])==saved["trace_sha256"])
                checks[mode+"_post_timing_fresh_fit_identity"]=saved["canonical_identity_checked_after_timing"]
                executions.append(record)
            stats=run.task_statistics(e,executions)
            checks["no_utility_without_extra_skips"]=not stats["positive_incremental_certificate_utility"]
            # Force a valid common-range abstention with outside-interval labels.
            outside={"ids":decision["ids"],"y":[F(3)]*6}
            for mode in ("ordinary","triggered"):
                record,bundles,certs=run.timed_policy("artificial",mode,1,order,maps,features,bounds,train,outside,anchors,F(1),protocol)
                checks[mode+"_common_range_abstention"]=(record["selection"]["abstained"] and record["counts"]["fit_calls"]==0 and record["counts"]["range_pruned_heads"]==4)
            # At a zero tolerance, the nonzero affine losses cause certificate calls;
            # equal query coordinates force positive pair optima for unequal labels.
            query=np.zeros((6,3))
            triggerfeatures={name:{"train":ztrain,"decision":query} for name in maps}
            record,bundles,certs=run.timed_policy("artificial","triggered",2,order,maps,triggerfeatures,bounds,train,decision,anchors,F(0),protocol)
            checks["actual_pair_trigger_and_class_prune"]=(len(certs)==2 and record["counts"]["class_pruned_heads"]==2 and record["trace"]["exact_selected_pair_distance_evaluations"]>0)
            run.retain_execution(record,bundles,certs,canonical=None)
            checks["pair_receipts_retained"]=(len(list((run.OUT/"pairs").rglob("*.json")))==2)
        finally:
            run.ROOT,run.OUT=old_root,old_out
    if not all(checks.values()):
        raise AssertionError(checks)
    receipt={"schema":"step36-artificial-runner-preflight-v1","status":"PASS","artificial_only":True,
             "public_measurement_values_opened":False,"checks":checks,
             "source_sha256":{name:hashlib.sha256((ROOT/"source"/name).read_bytes()).hexdigest() for name in
                  ("run36.py","models36.py","triggered_policy36.py","protocol36.py","check_run36.py")}}
    data36.write_json(ROOT/"tests/artificial_run_preflight36.json",receipt)
    print(json.dumps(receipt,sort_keys=True))


if __name__=="__main__":
    with threadpool_limits(limits=1):main()
