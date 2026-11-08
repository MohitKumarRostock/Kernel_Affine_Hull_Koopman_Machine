"""Training-only finite-feature heads and label-free arithmetic predictions."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_key]="1"
from fractions import Fraction as F
import hashlib
import json
from math import isqrt
import time
import numpy as np
import full_graph37 as g
import finite_feature35 as ff

ASSURANCE=ff.ASSURANCE
RECIPES=("affine","extension32","qp32","envelope_all","qp_all")
HEAD_SCHEMA="step38-stored-feature-extension-head-v1"

def canonical(value):return g.canon(value)
def canonical_bytes(value):return g.canonical_bytes(value)
def digest(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()
def box(lo,hi):return {"lower":str(lo),"upper":str(hi),"width":str(hi-lo)}
def clip(value):return max(F(0),min(F(1),value))
def features_exact(z):return [[F.from_float(float(x)) for x in row] for row in np.asarray(z,dtype=np.float64)]

def fit_values(z,y,*,ids,feature_id,kappa="2",role="train",input_binding=None,settings=None):
    """A full-graph fit or a separately scoped post-assessment diagnostic."""
    if role not in ("train","assessment_diagnostic","artificial"):raise ValueError("Unknown fit role")
    start=time.perf_counter();z=np.asarray(z,dtype=np.float64);y=list(map(F,y));kap=F(kappa)
    if len(z)!=len(y) or len(ids)!=len(z) or len(set(ids))!=len(ids):raise ValueError("Training roster mismatch")
    settings=dict(g.SOLVER_SETTINGS if settings is None else settings)
    graph,pairs,floors,exact,weights,scale=g.distance_graph(z,kappa=kap);graph["kappa_numerator"]=kap.numerator
    graph_seconds=time.perf_counter()-start
    numerical=g.numerical_proposal(y,graph,pairs,floors,settings=settings)
    witness_start=time.perf_counter()
    candidates=[g.repair_proposal(y,y,weights,scale,pairs,"clipped_target_envelope")]
    if numerical["raw_primal_binary64"] is not None:
        candidates.append(g.repair_proposal(numerical["raw_primal_binary64"],y,weights,scale,pairs,"solver_envelope"))
    center=clip(sum(y,F(0))/len(y));values=[center]*len(y)
    sse=g.verify_values(values,y,pairs,weights,scale)
    candidates.append({"source":"clipped_mean_constant","value":center,"values":values,"sse":sse,"mean":sse/len(y)})
    chosen=min(candidates,key=lambda c:(c["sse"],c["source"]))
    env=min((c for c in candidates if c["source"]!="solver_envelope"),key=lambda c:(c["sse"],c["source"]))
    dual=g.weak_dual(y,graph,pairs,floors,exact,numerical,kap)
    range_terms=[(clip(v)-v)**2 for v in y];floor=sum(range_terms,F(0))/len(y)
    lower=max(floor,dual["mean"]);upper=chosen["mean"]
    if not 0<=lower<=upper:raise ArithmeticError("Training class-risk bracket conflicts")
    return canonical({"schema":"step38-training-class-fit-v1","role":role,"assurance":ASSURANCE,
        "feature_id":feature_id,"class_id":ff.class_identity(feature_id,kap,("0","1")),
        "n":len(y),"dimension":int(z.shape[1]),"ids":list(ids),"kappa":kap,"output_interval":["0","1"],
        "feature_digest":ff.digest(features_exact(z)),"target_digest":ff.digest(y),"input_binding":input_binding or {},
        "distance":graph,"numerical":numerical,"primal_candidates":candidates,
        "selected_qp_source":chosen["source"],"selected_envelope_source":env["source"],
        "dual":dual,"range_floor":{"singleton_sse":range_terms,"mean":floor,"sse":floor*len(y)},
        "class_risk":box(lower,upper),"gap_target_mean":str(g.GAP_MEAN),"gap_resolved":upper-lower<=g.GAP_MEAN,
        "training_qp_no_worse_than_envelope":chosen["sse"]<=env["sse"],
        "head_promotion_allowed":role=="train","targets_clipped_in_loss":False,
        "timing":{"total_seconds":time.perf_counter()-start,"exact_graph_seconds":graph_seconds,
            "numerical_proposal_seconds":numerical["proposal_wall_seconds"],"exact_witness_seconds":time.perf_counter()-witness_start},
        "native_transform_error_enclosed":False,"future_response_coverage_claim":False,"population_risk_claim":False})

def make_extension(fit,z,*,anchor_indices,recipe,fit_path,fit_sha256):
    if fit["role"]!="train" or not fit["head_promotion_allowed"]:raise ValueError("Assessment witness cannot become a head")
    if recipe not in ("qp32","qp_all","envelope_all"):raise ValueError("Unknown new head recipe")
    z=np.asarray(z,dtype=np.float64)
    if ff.digest(features_exact(z))!=fit["feature_digest"]:raise ValueError("Head geometry differs from fitted geometry")
    key="selected_envelope_source" if recipe=="envelope_all" else "selected_qp_source"
    candidate=next(c for c in fit["primal_candidates"] if c["source"]==fit[key])
    if len(anchor_indices)!=len(z) or len(set(anchor_indices))!=len(z):raise ValueError("Invalid anchor columns")
    head=canonical({"schema":HEAD_SCHEMA,"kind":"symmetric_lipschitz_extension","recipe":recipe,
        "feature_id":fit["feature_id"],"class_id":fit["class_id"],"dimension":fit["dimension"],
        "kappa":fit["kappa"],"output_interval":fit["output_interval"],"assurance":ASSURANCE,
        "anchor_ids":fit["ids"],"anchor_indices":list(map(int,anchor_indices)),
        "anchor_features":features_exact(z),"anchor_values":candidate["values"],
        "training_fit_path":fit_path,"training_fit_sha256":fit_sha256,"training_fit_role":"train",
        "selected_training_source":candidate["source"],"training_anchor_risk":candidate["mean"],
        "formula":"clip_I((max_i(v_i-kappa*d(z,a_i))+min_i(v_i+kappa*d(z,a_i)))/2)"})
    return head

def head_identity(head):
    return "head38:"+ff.digest(head) if head.get("schema")==HEAD_SCHEMA else ff.head_identity(head)

def fit_affine(z,y,*,feature_id,kappa="2",ridge_lambda="1/1000000",training_binding=None):
    z=np.asarray(z,dtype=np.float64);ys=list(map(F,y));yf=np.asarray([float(v) for v in ys])
    xc=z-z.mean(axis=0);yc=yf-yf.mean()
    beta=np.linalg.solve(xc.T@xc+len(z)*float(F(ridge_lambda))*np.eye(z.shape[1]),xc.T@yc)
    intercept=float(yf.mean()-z.mean(axis=0)@beta)
    center=[sum((F.from_float(float(v)) for v in z[:,j]),F(0))/len(z) for j in range(z.shape[1])]
    receipt=ff.make_affine_head(beta.tolist(),intercept,feature_id=feature_id,kappa=kappa,output_interval=("0","1"),
        project=True,center_features=center,center_target=sum(ys,F(0))/len(ys),bits=80)
    receipt["training_binding"]=training_binding or {};return receipt

def query_distances(anchors,queries,*,anchor_ids,query_ids,feature_id,bits=80):
    """Shared exact query-to-all-training-anchor distance enclosures; no labels."""
    a=features_exact(anchors);z=features_exact(queries)
    if not a or not z or len(a[0])!=len(z[0]) or len(anchor_ids)!=len(a) or len(query_ids)!=len(z):raise ValueError("Query geometry mismatch")
    if len(set(anchor_ids))!=len(a) or len(set(query_ids))!=len(z):raise ValueError("Repeated query/anchor ID")
    exponent=max(x.denominator.bit_length()-1 for row in a+z for x in row)
    def integers(rows):return [[x.numerator << (exponent-(x.denominator.bit_length()-1)) for x in row] for row in rows]
    ai=integers(a);zi=integers(z);floors=[];exact=[]
    for vector in zi:
        for anchor in ai:
            sq=sum((x-y)**2 for x,y in zip(vector,anchor))
            if exponent<=bits:
                scaled=sq << (2*(bits-exponent));floor=isqrt(scaled);equal=floor*floor==scaled
            else:
                floor=isqrt(sq) >> (exponent-bits);equal=(floor*floor << (2*(exponent-bits)))==sq
            if equal:exact.append(len(floors))
            floors.append(str(floor))
    return canonical({"schema":"step38-query-distances-v1","feature_id":feature_id,"anchor_ids":list(anchor_ids),
        "query_ids":list(query_ids),"n_anchor":len(a),"n_query":len(z),"dimension":len(a[0]),
        "anchor_feature_digest":ff.digest(a),"query_feature_digest":ff.digest(z),"features_exact":z,
        "bits":bits,"common_binary_exponent":exponent,"pair_order":"query_then_anchor_rowmajor",
        "floor_scaled":floors,"exact_pair_indices":exact,"target_values_accessed":False})

def evaluate_head(head,z,*,ids,distance=None):
    """Enclose the fixed formula; target loss is computed separately after access."""
    qz=features_exact(z);h_id=head_identity(head);kap=F(head["kappa"])
    if len(qz)!=len(ids) or any(len(v)!=head["dimension"] for v in qz):raise ValueError("Head/query roster mismatch")
    if head["class_id"]!=ff.class_identity(head["feature_id"],kap,("0","1")):raise ValueError("Class identity mismatch")
    rows=[]
    if head["kind"]=="clipped_affine":
        beta=list(map(F,head["coefficients"]));intercept=F(head["intercept"])
        if sum((v*v for v in beta),F(0))>kap*kap:raise ValueError("Affine sensitivity violation")
        for name,vector in zip(ids,qz):
            raw=intercept+sum((b*x for b,x in zip(beta,vector)),F(0));value=clip(raw)
            rows.append({"query_id":name,"raw_prediction":box(raw,raw),"prediction":box(value,value)})
        distance_digest=None;columns=None
    else:
        if head["kind"]!="symmetric_lipschitz_extension":raise ValueError("Unsupported head")
        if distance is None or distance["feature_id"]!=head["feature_id"] or distance["query_ids"]!=list(ids):raise ValueError("Unbound query distances")
        if distance["query_feature_digest"]!=ff.digest(qz):raise ValueError("Query distance feature mismatch")
        lookup={name:i for i,name in enumerate(distance["anchor_ids"])}
        columns=[lookup[name] for name in head["anchor_ids"]]
        if head.get("schema")==HEAD_SCHEMA and columns!=head["anchor_indices"]:raise ValueError("New anchor indices mismatch")
        if head.get("schema") not in (HEAD_SCHEMA,ff.HEAD_SCHEMA):raise ValueError("Unknown extension schema")
        if head.get("schema")==ff.HEAD_SCHEMA and len(columns)>32:raise ValueError("Legacy anchor limit exceeded")
        values=list(map(F,head["anchor_values"]));exact=set(distance["exact_pair_indices"]);den=1<<distance["bits"]
        if len(values)!=len(columns) or any(not 0<=v<=1 for v in values):raise ValueError("Invalid anchor values")
        for i,name in enumerate(ids):
            lower_lo=[];lower_hi=[];upper_lo=[];upper_hi=[]
            for j,value in zip(columns,values):
                index=i*distance["n_anchor"]+j;floor=int(distance["floor_scaled"][index])
                dlo=F(floor,den);dhi=F(floor+(index not in exact),den)
                lower_lo.append(value-kap*dhi);lower_hi.append(value-kap*dlo)
                upper_lo.append(value+kap*dlo);upper_hi.append(value+kap*dhi)
            elo,ehi=max(lower_lo),max(lower_hi);ulo,uhi=min(upper_lo),min(upper_hi)
            rlo,rhi=(elo+ulo)/2,(ehi+uhi)/2;plo,phi=clip(rlo),clip(rhi)
            if not 0<=plo<=phi<=1 or phi-plo>kap/F(den):raise ArithmeticError("Invalid arithmetic prediction enclosure")
            rows.append({"query_id":name,"lower_envelope":box(elo,ehi),"upper_envelope":box(ulo,uhi),
                "raw_prediction":box(rlo,rhi),"prediction":box(plo,phi)})
        distance_digest=digest(distance)
    return {"schema":"step38-head-prediction-v1","feature_id":head["feature_id"],"class_id":head["class_id"],
        "head_id":h_id,"query_ids":list(ids),"query_feature_digest":ff.digest(qz),"n":len(qz),
        "distance_digest":distance_digest,"anchor_columns":columns,"rows":rows,
        "predictions":[r["prediction"] for r in rows],"targets_accessed":False,"assurance":ASSURANCE,
        "future_response_coverage_claim":False,"native_transform_error_enclosed":False}

def baseline_prediction(name,values,ids):
    values=[v if isinstance(v,F) else F.from_float(float(v)) for v in values]
    return {"schema":"step38-baseline-prediction-v1","baseline":name,"query_ids":list(ids),
        "predictions":[box(v,v) for v in values],"targets_accessed":False,
        "class_membership_certified":False,"assurance":"EXACT_LOSS_OF_STORED_BASELINE_PREDICTIONS"}

def score(predictions,targets):return ff.loss_bounds(predictions,targets)

def select(rows):
    if not rows:raise ValueError("Cannot select from empty candidate panel")
    chosen=min(rows,key=lambda row:(F(row["validation_risk"]["upper"]),row["deployment_payload_bytes"],row["candidate_id"]))
    return {"candidate_id":chosen["candidate_id"],"validation_risk":chosen["validation_risk"],
        "deployment_payload_bytes":chosen["deployment_payload_bytes"],"eligible":chosen["validation_eligible"]}

def practical_gate(new,legacy,ordinary):
    nu=F(new["upper"]);ll=F(legacy["lower"]);bl=F(ordinary["lower"])
    denominator=ll>0
    improvement=denominator and 20*nu<=19*ll
    competitive=nu<=bl
    return {"legacy_positive_risk_denominator":denominator,"at_least_five_percent_mse_improvement":improvement,
        "no_worse_than_validation_selected_ordinary_baseline":competitive,
        "numerical_practical_transfer_criterion":improvement and competitive,
        "relative_gain_status":"DEFINED" if denominator else "UNDEFINED_RELATIVE_GAIN",
        "five_percent_refers_to":"MSE_not_RMSE","population_noninferiority_claim":False}
