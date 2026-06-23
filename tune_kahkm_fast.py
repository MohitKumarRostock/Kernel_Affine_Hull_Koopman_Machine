from __future__ import annotations
import argparse, csv, time, math, json
from pathlib import Path
import numpy as np
from experiment_16_ai_classic_control_closed_loop_pylance_clean import (
    collect_closed_loop_data, kahm_associations, fit_kahkm,
    evaluate_association_rollout, _episode_start_target_pairs, _relative_error, _r2,
    simplex_violation,
)


def parse_ints(xs): return [int(x) for x in xs]
def parse_floats(xs): return [float(x) for x in xs]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=['CartPole-v1','MountainCar-v0'])
    ap.add_argument('--clusters', nargs='+', default=['10','20','50','100'])
    ap.add_argument('--omegas', nargs='+', default=['1','2','4','8'])
    ap.add_argument('--train-episodes', type=int, default=8)
    ap.add_argument('--test-episodes', type=int, default=4)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--horizons', nargs='+', default=['1','5','10','20','50'])
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--nlms-epochs', type=int, default=20)
    ap.add_argument('--subspace-dim', type=int, default=20)
    ap.add_argument('--nb', type=int, default=100)
    ap.add_argument('--tau', type=float, default=1e-6)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--n-jobs', type=int, default=1)
    ap.add_argument('--train-seed', type=int, default=0)
    ap.add_argument('--test-seed', type=int, default=1000)
    ap.add_argument('--random-state', type=int, default=0)
    ap.add_argument('--output-dir', type=str, default='/mnt/data/kahkm_fast_tuning_outputs')
    args=ap.parse_args()
    clusters=parse_ints(args.clusters); omegas=parse_floats(args.omegas); horizons=parse_ints(args.horizons)
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    one=[]; multi=[]; index=[]
    for task in args.tasks:
        # defaults from experiment_16 DEFAULT_TASK_CONFIGS without import clutter
        default_steps = 250 if task=='CartPole-v1' else 200
        max_steps=args.max_steps if args.max_steps>0 else default_steps
        eps=0.05
        print(f'Collect {task}', flush=True)
        train=collect_closed_loop_data(task, episodes=args.train_episodes, max_steps=max_steps, seed=args.train_seed, exploration_eps=eps)
        test=collect_closed_loop_data(task, episodes=args.test_episodes, max_steps=max_steps, seed=args.test_seed, exploration_eps=eps)
        for C in clusters:
            for omega in omegas:
                print(f'RUN {task} C={C} omega={omega}', flush=True)
                t0=time.time(); status='ok'
                try:
                    fit=fit_kahkm(
                        train.X0, train.X1, n_clusters=C, subspace_dim=args.subspace_dim, Nb=args.nb,
                        omega=omega, tau=args.tau, beta=args.beta, nlms_epochs=args.nlms_epochs,
                        random_state=args.random_state, kmeans_kind='full', kmeans_batch_size=4096,
                        max_train_per_cluster=None, save_ae_to_disk=False, n_jobs=args.n_jobs,
                        batch_size=args.batch_size, project_stochastic=True, preload_classifier_after_fit=False,
                        verbose=False,
                    )
                    Phi_test=kahm_associations(fit.abstraction_model, test.X0, omega=fit.omega, tau=fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
                    Chi_test=kahm_associations(fit.abstraction_model, test.X1, omega=fit.omega, tau=fit.tau, n_jobs=args.n_jobs, batch_size=args.batch_size, show_progress=False)
                    pred=np.asarray(fit.B.T @ Phi_test, dtype=np.float64)
                    one.append({'task':task,'n_clusters':C,'omega':omega,'train_error':fit.train_closure_error,'train_r2':fit.association_r2,'test_error':_relative_error(pred,Chi_test),'test_r2':_r2(pred,Chi_test),'simplex_violation':simplex_violation(pred)})
                    for h in horizons:
                        try:
                            X_start, X_target=_episode_start_target_pairs(test.episodes,h)
                        except RuntimeError:
                            continue
                        Psi_start=kahm_associations(fit.abstraction_model,X_start,omega=fit.omega,tau=fit.tau,n_jobs=args.n_jobs,batch_size=args.batch_size,show_progress=False)
                        Psi_target=kahm_associations(fit.abstraction_model,X_target,omega=fit.omega,tau=fit.tau,n_jobs=args.n_jobs,batch_size=args.batch_size,show_progress=False)
                        row=evaluate_association_rollout(method_name='kahkm_nlms',horizon=h,start_phi=Psi_start,target_phi=Psi_target,B=np.asarray(fit.B,dtype=np.float64))
                        row['task']=task; row['n_clusters']=C; row['omega']=omega
                        multi.append(row)
                except Exception as e:
                    status='failed:'+repr(e)
                sec=time.time()-t0
                index.append({'task':task,'n_clusters':C,'omega':omega,'status':status,'seconds':sec})
                for name,rows in [('one_step.csv',one),('multistep.csv',multi),('index.csv',index)]:
                    if rows:
                        with (out/name).open('w',newline='',encoding='utf-8') as f:
                            w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
                print(f'DONE {task} C={C} omega={omega} {status} seconds={sec:.2f}', flush=True)
    # best
    best=[]
    for task in args.tasks:
        for objective in ['h50','mean_1_5_10_20_50']:
            candidates=[]
            configs={(r['n_clusters'],r['omega']) for r in multi if r['task']==task}
            for C,omega in configs:
                rs=[r for r in multi if r['task']==task and r['n_clusters']==C and r['omega']==omega]
                if objective=='h50':
                    rr=[r for r in rs if int(r['horizon'])==50]
                    if not rr: continue
                    score=float(rr[0]['relative_error']); r2=float(rr[0]['association_r2'])
                else:
                    vals=[float(r['relative_error']) for r in rs if int(r['horizon']) in {1,5,10,20,50}]
                    if not vals: continue
                    score=sum(vals)/len(vals); r2=math.nan
                candidates.append((score,C,omega,r2))
            if candidates:
                score,C,omega,r2=min(candidates)
                best.append({'task':task,'objective':objective,'n_clusters':C,'omega':omega,'score':score,'h50_r2':r2})
    if best:
        with (out/'best.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(best[0].keys())); w.writeheader(); w.writerows(best)
    with (out/'metadata.json').open('w',encoding='utf-8') as f: json.dump(vars(args),f,indent=2)
    print('BEST')
    for b in best: print(b, flush=True)
if __name__=='__main__': main()
