#!/usr/bin/env python3
"""Reconstruct retained endpoints, summarize all trials and build paper assets."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from scipy.stats import beta
from core import (ETA,SIZES,REPETITIONS,TOLERANCES,PRIMARY,METHODS,laws,grouped_objective,
                  qp_values,centered_from_counts,eb_lower,erm_kl_lower,stratified_from_summaries)

LABELS={'collapsed_binary':'Collapsed binary','two_atom_partial':'Two atoms',
        'three_point_v':'Three-point V','five_point_zigzag':'Five-point zigzag',
        'partial_collapse':'Partial collapse','noisy_affine':'Noisy affine',
        'rare_conflict':'Rare conflict','affine_mismatch':'Affine mismatch'}
METHOD_LABELS={'pair_eb':'Pair EB','centered_range':'Centered','erm_kl':'Empirical minimum',
               'stratified_eb':'Stratified EB'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary_from_matrix(matrix,scores):
    matrix=np.asarray(matrix)
    n=int(matrix.sum())
    if n==0:
        return {'mean':0.,'variance':0.,'n_pairs':0}
    mean=float(np.sum(matrix*scores)/n)
    variance=max(0.,float(np.sum(matrix*(scores-mean)**2)/(n-1))) if n>1 else 0.
    return {'mean':mean,'variance':variance,'n_pairs':n}


def reconstructed(law,row):
    n=row['n'];counts=np.asarray(row['counts'])
    _,w,m,c,a,b=grouped_objective(law,counts/n)
    qp=row['empirical_qp']
    lower,upper,violation,_,_=qp_values(w,m,c,a,b,qp['primal'],qp['dual_multipliers'])
    if violation>1e-10 or upper-lower>1e-8:
        raise AssertionError('Stored QP certificate is not within its declared tolerances')
    score=law.score_matrix()
    pair=summary_from_matrix(row['pair_counts'],score)
    strata=[summary_from_matrix(matrix,score) for matrix in row['stratum_pair_counts']]
    result={'pair_eb':eb_lower(pair['mean'],pair['variance'],pair['n_pairs'],2.,ETA),
            'pair_support_eb':eb_lower(pair['mean'],pair['variance'],pair['n_pairs'],float(score.max()),ETA),
            'centered_range':centered_from_counts(law,counts)[0],
            'erm_kl':erm_kl_lower(lower,n),
            'erm_hoeffding':max(0.,lower-4*math.sqrt(math.log(1/ETA)/(2*n)))}
    for name,support,known in (('stratified_eb',False,False),('stratified_support_eb',True,False),
                               ('stratified_knownmass_support_eb',True,True)):
        result[name]=stratified_from_summaries(law,strata,row['stratum_counts'],n,
                                              support=support,known_masses=known)[0]
    return result,upper-lower


def binomial_interval(successes,total):
    return (0. if successes==0 else float(beta.ppf(.025,successes,total-successes+1)),
            1. if successes==total else float(beta.ppf(.975,successes+1,total-successes)))


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def make_assets(population_rows,summary,paper,analysis):
    os.environ.setdefault('MPLCONFIGDIR',str(analysis/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tables=paper/'tables';figures=paper/'figures'
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    pop={p['name']:p for p in population_rows}
    lookup={(r['law'],r['n'],r['method']):r for r in summary}
    rows=[r'\begin{table}[t]',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',
          r'\begin{tabular}{lrrrrrrrr}',r'\toprule',
          r'& \multicolumn{4}{c}{Population quantities} & \multicolumn{4}{c}{Mean lower endpoint, $n=4096$} \\',
          r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
          r'Law & $R^*$ & $W$ & $W_{\rm str}$ & $C$ & Pair & Centered & ERM & Strat. \\',r'\midrule']
    for law in laws():
        p=pop[law.name]
        values=[(p['oracle']['lower']+p['oracle']['upper'])/2,p['pair_witness'],
                p['stratified_witness'],p['centered_bound']]
        values += [lookup[(law.name,4096,m)]['lower_mean'] for m in PRIMARY]
        rows.append(LABELS[law.name]+' & '+' & '.join(f'{v:.4f}' for v in values)+r' \\')
    rows += [r'\bottomrule',r'\end{tabular}',
             r'\caption{Finite-support comparison on the same full scalar Lipschitz class. $R^*$ is the minimum MSE, $W$ the unconditional pair witness, $W_{\rm str}$ the prespecified stratified witness, and $C$ the centered population bound. The last four columns average 300 independent calibration samples. Each method separately has nominal one-sided confidence $95\%$; no joint confidence across columns is asserted. Empirical risk minimization (ERM) uses a dual-certified empirical minimum and bounded-loss Kullback--Leibler (KL) inversion. Primary stratification estimates the stratum masses. All quantities are in MSE units.}',
             r'\label{tab:witness-comparison}',r'\end{table}','']
    (tables/'witness_comparison.tex').write_text('\n'.join(rows))
    timing=[r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{lrrr}',r'\toprule',
            r'Method & Violations / 9600 & Median cost ($\mu$s) & Law-wise median range ($\mu$s) \\',r'\midrule']
    for method in PRIMARY:
        subset=[r for r in summary if r['method']==method]
        costs=[r['seconds_median']*1e6 for r in subset if r['n']==4096]
        timing.append(f"{METHOD_LABELS[method]} & {sum(r['coverage_violations'] for r in subset)} & {np.median(costs):.1f} & {min(costs):.1f}--{max(costs):.1f}"+r' \\')
    timing += [r'\bottomrule',r'\end{tabular}',
               r'\caption{Coverage violations pooled over the prespecified eight laws and four calibration sizes, and certificate-only computation at $n=4096$. The cost column is the median of the eight law-wise median times; the range shows those medians. Timing starts after sample generation and representation lookup and includes optimization for ERM. It excludes feature learning. These single-machine measurements are descriptive; pooled coverage does not establish coverage uniformly over unseen laws.}',
               r'\label{tab:witness-cost}',r'\end{table}','']
    (tables/'witness_cost.tex').write_text('\n'.join(timing))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.titlesize':10,
                         'axes.labelsize':9.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
                         'legend.fontsize':9.5,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(4,2,figsize=(6,7.2))
    colors=['#126782','#c56b25','#7552a0','#2d804b'];markers=['o','s','^','D']
    for ax,law in zip(axes.flat,laws()):
        for method,color,marker in zip(PRIMARY,colors,markers):
            means=[lookup[(law.name,n,method)]['lower_mean'] for n in SIZES]
            ax.plot(SIZES,means,color=color,marker=marker,markersize=3.2,linewidth=1.2,label=METHOD_LABELS[method])
        # Display precision is much coarser than the guarded oracle bracket.
        truth=round((pop[law.name]['oracle']['lower']+pop[law.name]['oracle']['upper'])/2,10)
        ax.axhline(truth,color='0.25',linestyle='--',linewidth=1.,label='Minimum risk')
        ax.set_title(LABELS[law.name]);ax.set_xscale('log',base=2)
        ax.set_xticks(SIZES,[str(n) for n in SIZES]);ax.tick_params(axis='x',labelrotation=25)
        ax.set_ylim(0,max(.02,1.08*truth));ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    for ax in axes[:,0]:ax.set_ylabel('MSE lower endpoint')
    for ax in axes[-1,:]:ax.set_xlabel('Calibration observations')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='lower center',frameon=False,bbox_to_anchor=(.5,.0))
    fig.subplots_adjust(left=.12,right=.985,top=.965,bottom=.185,hspace=.65,wspace=.28)
    fig.savefig(figures/'witness_sample_size.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(figures/'witness_sample_size.png',dpi=180)
    plt.close(fig)
    paths=[tables/'witness_comparison.tex',tables/'witness_cost.tex',figures/'witness_sample_size.pdf',figures/'witness_sample_size.png']
    (analysis/'asset_sha256.json').write_text(json.dumps({str(p.relative_to(paper)):digest(p) for p in paths},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True,type=Path)
    parser.add_argument('--paper',required=True,type=Path)
    args=parser.parse_args()
    results=args.results.resolve();analysis=results/'analysis'
    analysis.mkdir(exist_ok=False)
    original_hashes=json.loads((results/'SHA256SUMS.json').read_text())
    for name,value in original_hashes.items():
        if digest(results/name)!=value:raise AssertionError(f'Retained file hash differs: {name}')
    population_rows=json.loads((results/'population.json').read_text())
    populations={p['name']:p for p in population_rows}
    law_by_name={law.name:law for law in laws()}
    groups=defaultdict(list);seen=set();maximum_difference=0.;maximum_gap=0.
    with gzip.open(results/'trials.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);key=(row['law'],row['n'],row['repetition'])
            if key in seen:raise AssertionError('Duplicate trial')
            seen.add(key)
            reproduced,gap=reconstructed(law_by_name[row['law']],row)
            maximum_gap=max(maximum_gap,gap)
            for method in METHODS:
                difference=abs(reproduced[method]-row['lower'][method])
                maximum_difference=max(maximum_difference,difference)
                if difference>2e-10:raise AssertionError(f'Endpoint reconstruction failed: {key}, {method}')
                groups[(row['law'],row['n'],method)].append(row)
    expected={(law.name,n,r) for law in laws() for n in SIZES for r in range(REPETITIONS)}
    if seen!=expected:raise AssertionError('Incomplete or unexpected trial schedule')
    summaries=[];exclusions=[]
    for (name,n,method),rows in sorted(groups.items()):
        p=populations[name];lo,hi=p['oracle']['lower'],p['oracle']['upper']
        lower=np.asarray([r['lower'][method] for r in rows]);timing=np.asarray([r['seconds'][method] for r in rows])
        raw=np.asarray([r['raw_statistics'][method] for r in rows])
        reference=(lo+hi)/2 if method.startswith('erm') else p['centered_bound'] if method=='centered_range' else p['stratified_witness'] if method.startswith('stratified') else p['pair_witness']
        violations=int(np.sum(lower>hi));ambiguity=int(np.sum((lower>lo)&(lower<=hi)))
        coverage_low,coverage_high=binomial_interval(len(rows)-violations,len(rows))
        summary={'law':name,'n':n,'method':method,'repetitions':len(rows),'oracle_lower':lo,'oracle_upper':hi,
                 'population_reference':reference,'population_slack':(lo+hi)/2-reference,
                 'lower_mean':float(lower.mean()),'lower_median':float(np.median(lower)),
                 'lower_q05':float(np.quantile(lower,.05)),'lower_q95':float(np.quantile(lower,.95)),
                 'mean_raw_statistic':float(raw.mean()),'mean_calibration_subtraction':float(np.mean(raw-lower)),
                 'mean_population_reference_minus_endpoint':float(reference-lower.mean()),
                 'coverage_violations':violations,'oracle_bracket_ambiguous':ambiguity,
                 'coverage_fraction':1-violations/len(rows),'coverage_mc_lower':coverage_low,'coverage_mc_upper':coverage_high,
                 'seconds_median':float(np.median(timing)),'seconds_q05':float(np.quantile(timing,.05)),
                 'seconds_q95':float(np.quantile(timing,.95))}
        summaries.append(summary)
        for epsilon in TOLERANCES:
            count=int(np.sum(lower>epsilon**2))
            true_exclusion=lo>epsilon**2
            false_exclusion=hi<=epsilon**2
            classification=('class_excluded' if true_exclusion else
                            'class_can_meet_tolerance' if false_exclusion else
                            'threshold_inside_numerical_oracle_bracket')
            low_ci,high_ci=binomial_interval(count,len(rows))
            exclusions.append({'law':name,'n':n,'method':method,'rmse_tolerance':epsilon,
                               'truth':classification,
                               'n_exclusions':count,'repetitions':len(rows),'exclusion_fraction':count/len(rows),
                               'mc_lower':low_ci,'mc_upper':high_ci})
    write_csv(analysis/'summary.csv',summaries);write_csv(analysis/'exclusion_power.csv',exclusions)
    make_assets(population_rows,summaries,args.paper.resolve(),analysis)
    audit={'status':'passed','trials':len(seen),'endpoints':len(seen)*len(METHODS),
           'maximum_reconstruction_difference':maximum_difference,'maximum_empirical_qp_gap':maximum_gap,
           'schedule_complete':True,'raw_input_hashes_verified':True,
           'source_sha256':{p.name:digest(p) for p in Path(__file__).resolve().parent.glob('*.py')},
           'primary_coverage_violations':{m:sum(r['coverage_violations'] for r in summaries if r['method']==m) for m in PRIMARY},
           'oracle_bracket_ambiguous_endpoints':sum(r['oracle_bracket_ambiguous'] for r in summaries),
           'audit_scope':'Reconstruction from retained sufficient statistics and dual multipliers, with source analytic and binomial tests separately checking the formulas.'}
    (analysis/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    hashes={p.name:digest(p) for p in analysis.iterdir() if p.is_file()}
    (analysis/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
