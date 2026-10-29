"""Reaggregate every frozen block-hierarchy endpoint and make paper assets."""
from pathlib import Path
import argparse
import io
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
LABELS={'collapsed_binary':'Collapsed binary','two_atom_partial':'Two-atom partial',
        'three_point_v':'Three-point V','five_point_zigzag':'Five-point zigzag',
        'partial_collapse':'Partial collapse','noisy_affine':'Noisy affine',
        'rare_conflict':'Rare conflict','affine_mismatch':'Affine mismatch'}


def main(results):
    data=pd.read_csv(results/'endpoints.csv')
    summary=data.groupby(['law','n','method'],sort=False).agg(
        mean_lower=('lower','mean'),sd_lower=('lower','std'),
        mean_score=('mean','mean'),sd_score=('mean','std'),
        repetitions=('lower','size'),coverage_failures=('coverage_failure','sum')).reset_index()
    summary.to_csv(results/'summary.csv',index=False)
    population={row['law']:row for row in json.loads((results/'population.json').read_text())}
    methods=['block_2','block_4','block_8','block_16','block_32','full_erm']
    table=[r'\begin{table}[t]',r'\centering\small',
           r'\caption{Mean lower endpoints at $n=4096$ over all 100 repetitions of each law. $R^*$ is the minimum risk in the same scalar Lipschitz class. Columns $L_b$ use disjoint blocks of size $b$; ERM uses the full sample. A single failure budget $0.05$ per law and repetition covers all six methods and all three sample sizes. The hierarchy orders population witnesses, not these finite-sample endpoints.}',
           r'\label{tab:block-hierarchy}',r'\begin{tabular}{lrrrrrrr}',r'\toprule',
           r'Law & $R^*$ & $L_2$ & $L_4$ & $L_8$ & $L_{16}$ & $L_{32}$ & ERM \\',r'\midrule']
    for law,label in LABELS.items():
        vals=[population[law]['optimum_lower']]
        for method in methods:
            row=summary[(summary.law==law)&(summary.n==4096)&(summary.method==method)]
            assert len(row)==1 and int(row.iloc[0].repetitions)==100
            vals.append(float(row.iloc[0].mean_lower))
        table.append(label+' & '+' & '.join(f'{v:.4f}' for v in vals)+r' \\')
    table.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (results/'block_hierarchy.tex').write_text('\n'.join(table)+'\n')
    (ROOT/'paper/tables/block_hierarchy.tex').write_text('\n'.join(table)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(8.2,6.3),layout='constrained')
    analytic=pd.read_csv(results/'analytic_hierarchy.csv')
    colors=['#236d9b','#c27021','#6a4492']
    for parameter,color in zip((.01,.05,.1),colors):
        r=analytic[(analytic.law=='rare_conflict')&(analytic.parameter==parameter)]
        axs[0,0].plot(r.block_size,r.ratio,color=color,label=fr'$\varepsilon={parameter:g}$')
    for parameter,color in zip((4,16,64),colors):
        r=analytic[(analytic.law=='separated_noise')&(analytic.parameter==parameter)]
        axs[0,1].plot(r.block_size,r.ratio,color=color,label=fr'$M={parameter}$')
    for ax,title in zip(axs[0],['(a) Rare conflicts: exact population ratio','(b) Independent noise: exact population ratio']):
        ax.set_xscale('log',base=2);ax.set_xticks([2,8,32,128,512],[2,8,32,128,512])
        ax.set_ylim(0,1.04);ax.set_xlabel('Block size b');ax.set_ylabel(r'$\mathcal{W}_b/R^*$')
        ax.set_title(title,fontsize=9.5);ax.legend(frameon=False,fontsize=8,loc='lower right')
        ax.grid(alpha=.18)
    for ax,law,title in zip(axs[1],['partial_collapse','two_atom_partial'],
                           ['(c) Partial collapse: calibrated lower endpoint','(d) Exact pair witness: calibrated lower endpoint']):
        for n,color in zip((256,1024,4096),colors):
            r=summary[(summary.law==law)&(summary.n==n)&(summary.method!='full_erm')].copy()
            r['b']=r.method.str.replace('block_','').astype(int);r=r.sort_values('b')
            ax.plot(r.b,r.mean_lower,marker='o',markersize=3,color=color,label=f'n={n}')
        ax.axhline(population[law]['optimum_lower'],color='#444444',linestyle='--',linewidth=1,label=r'$R^*$')
        ax.set_xscale('log',base=2);ax.set_xticks([2,4,8,16,32],[2,4,8,16,32])
        ax.set_ylim(bottom=0);ax.set_xlabel('Block size b');ax.set_ylabel('Mean lower endpoint (MSE)')
        ax.set_title(title,fontsize=9.5)
        if law=='partial_collapse':
            ax.legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(0.,.94))
        else:
            ax.legend(frameon=False,fontsize=8,loc='lower left')
        ax.grid(alpha=.18)
    pdf_buffer=io.BytesIO()
    fig.savefig(pdf_buffer,format='pdf')
    pdf_bytes=pdf_buffer.getvalue()
    if not pdf_bytes.startswith(b'%PDF-') or len(pdf_bytes)<1000:
        raise RuntimeError('Incomplete figure PDF')
    destination=ROOT/'paper/figures/block_hierarchy.pdf'
    staged=destination.with_suffix('.pdf.staged')
    staged.write_bytes(pdf_bytes)
    staged.replace(destination)
    fig.savefig(results/'block_hierarchy.png',dpi=160)
    plt.close(fig)
    print(summary[summary.n==4096].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True)
    main(parser.parse_args().results)
