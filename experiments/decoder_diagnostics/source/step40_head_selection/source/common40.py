"""Frozen identities, exact score arithmetic, and write-once Step40 records."""
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAMBDA_TOKENS = (("0", "l0"),)
WEIGHT_TOKENS = (("1/2", "w05"),)
SELECTORS = ("joint_rolling", "joint_last_fold", "qp_rolling", "affine_rolling",
             "ordinary_rolling", "hgb_rolling", "kahm_joint_rolling",
             "raw_joint_rolling", "nystrom_joint_rolling")

def canon(value):
    if isinstance(value, F): return str(value)
    if isinstance(value, dict): return {str(k):canon(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)): return [canon(v) for v in value]
    if hasattr(value, "item"): return value.item()
    return value

def canonical_bytes(value):
    return (json.dumps(canon(value),sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(canonical_bytes(value)).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def relative(path): return str(Path(path).resolve().relative_to(ROOT.resolve()))
def utc(): return datetime.now(timezone.utc).isoformat()

def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream: stream.write(canonical_bytes(value))
    return {"path":relative(path),"sha256":sha(path)}

def box(lo, hi):
    lo,hi=F(lo),F(hi)
    if lo>hi: raise ArithmeticError("Reversed exact interval")
    return {"lower":str(lo),"upper":str(hi),"width":str(hi-lo)}

def scale_box(record, scale):
    scale=F(scale)
    if scale<0: raise ValueError("Nonnegative risk scale required")
    return box(F(record["lower"])*scale,F(record["upper"])*scale)

def pooled(records):
    n=sum(item["n"] for item in records)
    if n<=0: raise ValueError("Cannot pool empty development cohorts")
    return box(sum((F(item["risk"]["lower"])*item["n"] for item in records),F(0))/n,
               sum((F(item["risk"]["upper"])*item["n"] for item in records),F(0))/n)

def head_configs(maps):
    rows=[]
    for name in maps:
        rows.append({"candidate_id":name+"::qp_l0_w05","map":name,"token":"qp_l0_w05",
                     "recipe":"qp_regularized","ridge_lambda":"0","extension_weight":"1/2"})
        rows.append({"candidate_id":name+"::affine","map":name,"token":"affine",
                     "recipe":"affine","ridge_lambda":None,"extension_weight":None})
    return rows

def selector_pool(rows,label):
    if label not in SELECTORS: raise ValueError("Unknown predeclared selector")
    if label=="ordinary_rolling": return [r for r in rows if r.get("recipe")=="baseline"]
    if label=="hgb_rolling": return [r for r in rows if r.get("baseline_kind")=="hgb"]
    if label=="affine_rolling": return [r for r in rows if r.get("recipe")=="affine"]
    if label=="qp_rolling": return [r for r in rows if r.get("recipe")=="qp_regularized"]
    out=[r for r in rows if r.get("recipe") in ("affine","qp_regularized")]
    if label in ("kahm_joint_rolling","raw_joint_rolling","nystrom_joint_rolling"):
        family=label.split("_")[0];out=[r for r in out if r["family"]==family]
    return out

def select(rows,label):
    pool=selector_pool(rows,label)
    if not pool: raise ValueError("Empty selector pool")
    key="last_fold_physical_risk" if label=="joint_last_fold" else "pooled_physical_risk"
    chosen=min(pool,key=lambda r:(F(r[key]["upper"]),r["selection_payload_bytes"],r["candidate_id"]))
    return {"candidate_id":chosen["candidate_id"],"criterion":key,"risk":chosen[key],
            "selection_payload_bytes":chosen["selection_payload_bytes"],"pool_size":len(pool),
            "selection_uses_development_only":True}

def same_final_payload(new,reference):
    fields=("candidate_id","head_id","head_sha256","test_predictions_digest")
    return all(isinstance(new.get(k),str) and new[k] and new[k]==reference.get(k) for k in fields)

def no_worse(new,reference):
    if same_final_payload(new,reference):
        return {"pass":True,"reason":"SAME_FINAL_CANDIDATE_PAYLOAD_EXACT_EQUALITY"}
    passed=F(new["physical_test_risk"]["upper"])<=F(reference["physical_test_risk"]["lower"])
    return {"pass":passed,"reason":"INTERVAL_NONINFERIORITY" if passed else "NOT_ESTABLISHED_BY_INTERVALS"}

def practical_gate(new,reference,ordinary,affine):
    a=new["physical_test_risk"];b=reference["physical_test_risk"]
    den=F(b["lower"])>0
    gain=den and 20*F(a["upper"])<=19*F(b["lower"])
    ordinary_check=no_worse(new,ordinary);affine_check=no_worse(new,affine)
    return {"reference_positive_risk":den,"at_least_five_percent_mse_improvement":gain,
            "no_worse_than_selected_ordinary":ordinary_check["pass"],
            "no_worse_than_selected_affine":affine_check["pass"],
            "ordinary_comparison_reason":ordinary_check["reason"],
            "affine_comparison_reason":affine_check["reason"],
            "joint_same_as_affine":same_final_payload(new,affine),
            "joint_superiority_over_affine_from_identity":False,
            "numerical_criterion_pass":gain and ordinary_check["pass"] and affine_check["pass"],
            "relative_gain_status":"DEFINED" if den else "UNDEFINED_RELATIVE_GAIN",
            "population_noninferiority_claim":False}
