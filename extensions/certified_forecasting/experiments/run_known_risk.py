"""Exact-risk two-atom benchmark; all trials retained in a small CSV file.

Unlike an empirical test-set proxy, both oracle and fitted population risks
are evaluated analytically. The centered comparator is the raw-target
Euclidean extension of Kumar and Kargaran (2026), with one reference class.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from certified_forecasting.certificates import mean_interval


def two_atom_pair_scores(states, deficit):
    """One score per disjoint pair of independent binary calibration states.

    The score is ``deficit**2 / 2`` for unequal states and zero otherwise.
    An odd final observation is excluded from pairing but remains available
    for the fitted-risk and centered bounds. At least two independent pairs
    are required by the empirical Bernstein radius.
    """
    z = np.asarray(states)
    if z.ndim != 1 or z.size < 4:
        raise ValueError("At least four independent calibration states are required")
    if not np.all((z == 0) | (z == 1)):
        raise ValueError("Calibration states must be binary")
    deficit = float(deficit)
    if not np.isfinite(deficit) or not 0.0 <= deficit <= 2.0:
        raise ValueError("The unexplained target separation must lie in [0, 2]")
    end = 2 * (z.size // 2)
    return (z[:end:2] != z[1:end:2]).astype(float) * deficit**2 / 2.0


def run(config,output):
    cfg=json.loads(config.read_text())
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    for n in cfg["calibration_sizes"]:
        if isinstance(n, bool) or not isinstance(n, int) or n < 4:
            raise ValueError("Calibration sizes must be integers of at least four")
    output.mkdir(parents=True)
    rng=np.random.default_rng(cfg["seed"])
    rows=[]
    start=time.perf_counter()
    # Every interval is a separate experiment. Delta is not claimed globally
    # over Monte Carlo repetitions or grid conditions.
    for p in cfg["probabilities"]:
        for a in cfg["softness"]:
            for kappa in cfg["kappa"]:
                distance=np.sqrt(2.0)*a
                deficit=max(0.,2.-kappa*distance)
                oracle=p*(1-p)*deficit**2
                for n in cfg["calibration_sizes"]:
                    for rep in range(cfg["replicates"]):
                        training=rng.binomial(1,p,cfg["n_training"])
                        phat=training.mean()
                        # Exact empirical constrained least squares on the two atoms.
                        pred0=-1.+phat*deficit
                        pred1=1.-(1.-phat)*deficit
                        truth=(1-p)*(pred0+1.)**2+p*(pred1-1.)**2
                        z=rng.binomial(1,p,n)
                        loss=np.where(z==0,(pred0+1.)**2,(pred1-1.)**2)
                        q=two_atom_pair_scores(z, deficit)
                        # Three reported directions: fitted UCB, pair LCB,
                        # centered LCB. Give each delta/3 for joint coverage.
                        tail=cfg["delta"]/3.
                        lo=mean_interval(q,2.,tail)
                        up=mean_interval(loss,4.,tail)
                        shat=4.*z.mean()*(1-z.mean())
                        that=2.*a*a*z.mean()*(1-z.mean())
                        r=np.log(2./tail)/n
                        v=max(0.,shat-4.*np.sqrt(r/2.))
                        t=min(1.,that+np.sqrt(2.*r)+1./n)
                        centered=max(0.,np.sqrt(v)-kappa*np.sqrt(t))**2
                        rows.append({
                            "p":p,"softness":a,"kappa":kappa,"n":n,"replicate":rep,
                            "oracle_mse":oracle,"fitted_population_mse":truth,
                            "pair_lower":lo["lower"],"centered_lower":centered,
                            "fitted_upper":up["upper"],
                            "pair_mean":lo["mean"],"empirical_mse":up["mean"],
                            "pair_valid":int(lo["lower"]<=oracle+1e-12),
                            "centered_valid":int(centered<=oracle+1e-12),
                            "upper_valid":int(truth<=up["upper"]+1e-12),
                            "joint_valid":int(lo["lower"]<=oracle+1e-12 and centered<=oracle+1e-12 and truth<=up["upper"]+1e-12),
                        })
    with (output/"trials.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary={"config":cfg,"configuration_sha256":hashlib.sha256(config.read_bytes()).hexdigest(),
             "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "n_trials":len(rows),"elapsed_seconds":time.perf_counter()-start,
             "failure_counts":{k:sum(1-r[k] for r in rows) for k in ["pair_valid","centered_valid","upper_valid","joint_valid"]},
             "confidence_scope":f"Within each trial, the three reported one-sided bounds have joint confidence at least {100.0 * (1.0 - cfg['delta']):.6g}%. No joint guarantee across trials is asserted.",
             "monte_carlo_scope":"Only these finite two-atom configurations; empirical coverage is not a proof of the theorem."}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=Path("configs/known_risk_v1.json"))
    p.add_argument("--output",type=Path,default=Path("results/known_risk_v1"))
    a=p.parse_args();run(a.config,a.output)
