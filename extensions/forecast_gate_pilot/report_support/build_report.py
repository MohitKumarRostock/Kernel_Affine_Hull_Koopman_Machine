from pathlib import Path
import csv, json, math, statistics, hashlib, argparse, os
os.environ.setdefault('MPLCONFIGDIR','/tmp/kahm_report_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape

parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
PROJECT=args.project.resolve()
RUN=PROJECT/'pilot_results'
OUT=args.output.resolve()
OUT.parent.mkdir(parents=True,exist_ok=True)
FIG=OUT.with_name('KAHM_Pilot_Comparison.png')
summary=json.loads((RUN/'summary.json').read_text())
protocol=json.loads((RUN/'protocol.json').read_text())
audit=json.loads((RUN/'audit_report.json').read_text())
online=json.loads((PROJECT/'check_online.json').read_text())
geometry_checks=json.loads((PROJECT/'check_geometry.json').read_text())
raw=list(csv.DictReader((RUN/'task_metrics.csv').open()))
rows=[{**r, **{k:float(r[k]) for k in ('mse','C_mse','mse_h1','mse_h4','mse_h12')}} for r in raw]
laws=['nonlinear_ar','van_der_pol','duffing']
lawlabels={'nonlinear_ar':'Nonlinear AR','van_der_pol':'Van der Pol','duffing':'Duffing'}
methods=protocol['methods']
labels={'raw_ridge':'Raw ridge','raw_rbf':'Raw history + RBF features','fixed_kahm':'Fixed KAHM','direct_kahm':'Direct KAHM',
        'fixed_affine':'Fixed affine','direct_affine':'Direct affine','fixed_locality':'Fixed locality','direct_locality':'Direct locality',
        'fixed_rbf_local':'Fixed RBF-local','direct_rbf_local':'Direct RBF-local','persistence':'Persistence'}
means={(a['law'],a['method']):a['mean_mse'] for a in summary['law_method_aggregates']}
horizon={(a['law'],a['method']):a['horizon_mse'] for a in summary['law_method_aggregates']}
diagnostics=[json.loads(p.read_text()) for p in sorted((RUN/'tasks').glob('*/diagnostics.json'))]
task_records=[json.loads(p.read_text()) for p in sorted((RUN/'tasks').glob('*/task.json'))]
costs={m:[d['methods'][m]['charged_wall_seconds'] for d in diagnostics] for m in methods if m!='persistence'}
gains={law:{ref:100*(means[law,ref]-means[law,'direct_kahm'])/means[law,ref]
            for ref in ('raw_ridge','raw_rbf','fixed_kahm','direct_affine','direct_locality','direct_rbf_local')}
       for law in laws}
opt=[o for d in diagnostics for m,v in d['methods'].items() if m.startswith('direct_') for h in v['horizons'] for o in h['optimizations']]
failures=sum(len(d.get('failures',[])) for d in diagnostics)
checks={k:v for k,v in audit.items() if 'check' in k or k in ['status','passed','errors','task_count','forecast_count']}
result_brief={'scope':'Exploratory known-law pilot; no confirmatory inference','methods':methods,
              'task_count':summary['task_count'],'available_method_task_outputs':summary['available_forecasts'],
              'gains_percent_positive_favors_direct_kahm':gains,
              'median_charged_wall_seconds':{m:float(np.median(v)) for m,v in costs.items()},
              'optimization_count':len(opt),'objective_evaluations':sum(o['nfev'] for o in opt),
              'budget_exhaustions':sum(o['status']=='evaluation_budget' for o in opt),
              'converged_optimizations':sum(o['converged'] for o in opt),
              'recorded_failure_events':failures,'audit':audit}
(PROJECT/'PILOT_FINDINGS.json').write_text(json.dumps(result_brief,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9.4,3.0))
gate_names=['kahm','affine','locality','rbf_local']
palette=['#157d85','#4b6fa4','#a38337','#9b638a']
x=np.arange(3); width=.18
for i,(gate,color) in enumerate(zip(gate_names,palette)):
    vals=[100*(means[law,'fixed_'+gate]-means[law,'direct_'+gate])/means[law,'fixed_'+gate] for law in laws]
    ax.bar(x+(i-1.5)*width,vals,width,label=gate.replace('_',' ').upper() if gate=='kahm' else gate.replace('_',' ').title(),color=color)
ax.axhline(0,color='#263748',linewidth=.8)
ax.set_xticks(x,[lawlabels[l] for l in laws]); ax.set_ylabel('MSE reduction from direct gate training (%)')
ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.2),frameon=False)
ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG,dpi=180,bbox_inches='tight'); plt.close(fig)

fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
for name,filename in [('Body','DejaVuSans.ttf'),('BodyBold','DejaVuSans-Bold.ttf'),('BodyItalic','DejaVuSans-Oblique.ttf'),('Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fontdir/filename)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
navy=colors.HexColor('#153047'); teal=colors.HexColor('#157d85'); gray=colors.HexColor('#526171'); light=colors.HexColor('#edf3f6')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom',fontName='BodyBold',fontSize=28,leading=34,textColor=navy,spaceAfter=16))
styles.add(ParagraphStyle(name='HeadingCustom',fontName='BodyBold',fontSize=19,leading=24,textColor=navy,spaceAfter=13))
styles.add(ParagraphStyle(name='SubCustom',fontName='BodyBold',fontSize=11.5,leading=16,textColor=teal,spaceBefore=9,spaceAfter=5))
styles.add(ParagraphStyle(name='BodyCustom',fontName='Body',fontSize=9.6,leading=14.1,textColor=navy,spaceAfter=7))
styles.add(ParagraphStyle(name='SmallCustom',fontName='Body',fontSize=8,leading=11.5,textColor=gray,spaceAfter=6))
styles.add(ParagraphStyle(name='CellCustom',fontName='Body',fontSize=8,leading=11,textColor=navy))
styles.add(ParagraphStyle(name='CellHead',fontName='BodyBold',fontSize=8,leading=11,textColor=colors.white))
styles.add(ParagraphStyle(name='Equation',fontName='Mono',fontSize=9.2,leading=14,textColor=navy,leftIndent=10,spaceBefore=4,spaceAfter=9))
styles.add(ParagraphStyle(name='Tag',fontName='BodyBold',fontSize=9,leading=13,textColor=teal,spaceAfter=12))
story=[]
def P(text,style='BodyCustom'): return Paragraph(text,styles[style])
def p(text,style='BodyCustom'): story.append(P(text,style))
def h(text): p(text,'SubCustom')
def eq(text): p(escape(text).replace('\n','<br/>'),'Equation')
def bullet(text): p('• '+text)
def table(headers,body,widths=None,numeric=None,padding=7):
    data=[[P(str(v),'CellHead') for v in headers]]+[[P(str(v),'CellCustom') for v in row] for row in body]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    cmds=[('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),
          ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
          ('TOPPADDING',(0,0),(-1,-1),padding),('BOTTOMPADDING',(0,0),(-1,-1),padding),
          ('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#bdcdd8'))]
    for i in range(1,len(data)):
        if i%2==1: cmds.append(('BACKGROUND',(0,i),(-1,i),light))
    t.setStyle(TableStyle(cmds)); story.append(t); story.append(Spacer(1,10))
def page(title,tag=None):
    if story: story.append(PageBreak())
    if tag: p(tag,'Tag')
    p(title,'HeadingCustom')
def signed(x): return f'{x:+.2f}%'
def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#cfdae2')); c.setLineWidth(.5); c.line(43,39,A4[0]-43,39)
    c.setFont('Body',7); c.setFillColor(gray)
    c.drawString(43,26,'KAHM research continuation | 31 October 2026 | Development evidence')
    c.drawRightString(A4[0]-43,26,str(doc.page))

# Page 1
p('RESEARCH CONTINUATION / 31 OCTOBER 2026','Tag')
p('A predictive contribution<br/>that KAHM must earn','TitleCustom')
p('Evidence review, executed gate-training pilot, bounded-update building blocks, and a protocol for genuinely reserved evaluation.','SubCustom')
p('<b>Recommendation.</b> Continue with a narrow, matched test of direct forecast training. Treat efficient updating as a separate empirical claim. The current manuscript supports neither a unique KAHM forecasting advantage nor a demonstrated streaming benefit.')
h('What was completed in this continuation')
table(['Work item','Completed result'],[
['Source review','All 118 manifest-listed attachment files passed independent size and SHA-256 checks. The referenced original forecast arrays and reproduction code were not available here.'],
['New executable pilot',f'{summary["complete_task_count"]}/9 tasks and {summary["available_forecasts"]}/99 method-task outputs, covering three known simulation laws and three new seeds. All four gates receive matched forecast supervision.'],
['Numerical machinery','Five geometry tests and eight bounded-update tests pass. Saved predictions, candidate losses, source/configuration hashes, and an independent audit are retained.'],
['Next-study protocol','A source-reserved comparison and delayed-label streaming design are specified. The source roster and application-specific accuracy/latency/memory requirements remain to be fixed before confirmation.']],[118,390])
h('What the new pilot can say')
p('It tests a minimal direct intervention: three horizon-specific temperatures on a fixed bank, with an A-fitted ridge correction and actual B forecast loss. It is not end-to-end optimization of KAHM reconstruction parameters. Three repetitions on previously known laws provide development evidence, not a generalization certificate.')
gain_text='; '.join(f'{lawlabels[l]} {signed(gains[l]["fixed_kahm"])}' for l in laws)
p('<b>Observed direct-versus-fixed KAHM MSE reductions:</b> '+gain_text+'. Positive is favorable. The complete matched-control comparisons are reported on page 4, including unfavorable outcomes.')
p('<b>Decision standard:</b> a reproducible improvement over the strongest eligible non-KAHM procedure selected without test outcomes, on genuinely reserved evidence, within declared total resource limits. Beating ridge alone does not meet that standard.')
p('Status: a completed exploratory extension plus an unexecuted confirmatory design. No manuscript submission, external registration, or new-source performance claim is implied.','SmallCustom')

# Page 2
page('The evidence that sets the next bar','01 / MANUSCRIPT EVIDENCE')
p('The most recent target-geometry study trains a shared Gaussian residual-regression metric proxy. It does not minimize the final KAHM forecast loss through the KAHM operator. Its archive endpoint averages series within each collection and then gives four collections equal weight. [M1]')
table(['Reported procedure','Scaled MSE','Change vs ridge','Charged seconds'],[
['Raw ridge','1.300458','reference','0.001299'],
['Fixed / in-sample KAHM','1.264589','-2.76%','about 0.051'],
['Learned / OOF KAHM','1.272128','-2.18%','0.1622'],
['Fixed / in-sample RBF-local','1.257888','-3.27%','about 0.027'],
['Learned / OOF affine','1.268679','-2.44%','about 0.142'],
['Learned / OOF locality','1.266922','-2.58%','about 0.141']],[205,87,110,106])
p('These are reported results from the supplied source, not newly reproduced forecasts. The learned KAHM arm is about 0.60% worse than fixed/in-sample KAHM and 1.13% worse than fixed/in-sample RBF-local, calculated from the reported means. The roughly 125-fold ridge cost ratio is host- and pipeline-specific; the absolute recorded time is also small. Neither fact alone establishes deployment utility.')
h('Synthetic evidence and attribution')
p('The manuscript reports 36 synthetic tasks and 12 archive tasks, with 768 complete method-task forecasts and no fallback in that latest study. Its 18 simultaneous synthetic contrasts yield three gains against ridge, five losses against matched local controls, and ten unresolved contrasts. None establishes an incremental KAHM benefit from the metric change or donor residuals. Raw RBF has lower descriptive mean loss on all three laws. [M1]')
p('The study is not a full metric-by-residual factorial: learned-metric/in-sample is absent. It therefore supports the declared local comparisons, not a complete interaction analysis. A new direct-objective experiment should hold the baseline and residual definition fixed rather than changing several causes together.')
h('Preserve earlier negative evidence')
p('The original natural comparison reports raw-plus-KAHM as 4.26% worse than raw RBF at 2.30 times the charged cost. The scaling follow-up reused inspected data. The additive utility extension retained eight geometric fit failures; its full-roster geometric losses remain undefined. These histories motivate the new test and must not be replaced or pooled with it. [M2]')
p('The attachment contains editable LaTeX, generated tables/figures and provenance. File integrity was verified here; the original 768-output audit is a reported prior audit. A referenced reproduction ZIP was not found, and the repository was not retrievable in this environment. [M3]','SmallCustom')

# Page 3
page('What the executable prototype trains','02 / DIRECT FORECAST OBJECTIVE')
p('For each gate family g, retain the shared partitions, references, rank-two principal coordinates and reconstruction rules. Train one log inverse-temperature per block b and forecast horizon h:')
eq('phi_c,h(x; eta) = softmax within block b [ exp(eta_b,h) * ell_c,g(x) ]\n-log(4) <= eta_b,h <= log(4)')
p('KAHM and affine use their radial/angular reconstruction scores; locality adds center distance to affine discrepancy; RBF-local uses center distance. All receive the same three parameters per horizon, identity initialization, bounds, optimizer, head grid and hard objective-evaluation ceiling.')
eq('F_g,eta(x) = [ phi_c(x), phi_c(x) P_c^T (z(x) - mu_c) ]_c\nr_A,h = y_A,h - f0,h(x_A)\nbeta(eta, alpha) = argmin_beta ||r_A,h - F_A,g,eta beta||^2 + alpha||beta||^2')
p('The feature columns and targets are centered on A, with an unpenalized intercept. At every temperature evaluation, the ridge head is refitted on A. The actual B forecast MSE of f0 + q is minimized; division by the A target variance only stabilizes optimizer scale. B fitting uses correction strength one; C subsequently selects shrinkage. This is a direct forecast objective, but not joint training of every component of the final selected procedure.')
table(['Split','Independent episodes / retained rows','Role'],[
['A','64 / 512','Fit bank and correction heads; select baseline penalty on an inner episode split, then fit baseline on all A.'],
['B','32 / 256','Train gate temperatures; select raw-history + RBF settings.'],
['C','32 / 256','Select correction penalty and weight, separately per horizon; zero first, strict improvement.'],
['D','64 / 832','Untuned exploratory evaluation. D targets never enter the fitting API.']],[40,172,296])
p('History length is 12; forecast offsets are 1, 4 and 12; each 72-value episode provides 13 origins. The bank has 2/4/8 groups, at most 24 references per group, and at most 42 local features. Each correction penalty is selected from {0.01, 0.1, 1}; correction weights are {0, 0.25, 0.5, 1}. Each (gate, horizon, penalty) gets at most 160 actual objective evaluations, including finite differences and line searches.')
p('The raw RBF reference concatenates standardized history and up to 64 Nystrom features. This, the episode counts, per-horizon selection and new numerical implementation differ from the historical procedures. No numerical replication of that paper is claimed. [N1]','SmallCustom')

# Page 4
page('Complete pilot outcomes','03 / EXECUTED DEVELOPMENT EVIDENCE')
p('Mean D MSE multiplied by 1,000, averaged over three independently generated tasks per known law. Lower is better. All 11 procedures are retained; no confidence intervals or significance claims are attached to these small exploratory means. [N1]')
table(['Procedure']+[lawlabels[l] for l in laws],[[labels[m]]+[f'{1000*means[l,m]:.3f}' for l in laws] for m in methods],[220,96,96,96])
h('Direct KAHM: reduction relative to each comparator')
table(['Comparator']+[lawlabels[l] for l in laws],[[labels[m]]+[signed(gains[l][m]) for l in laws]
      for m in ['fixed_kahm','direct_affine','direct_locality','direct_rbf_local','raw_rbf']],[220,96,96,96])
p('Positive values favor direct KAHM; negative values favor the comparator. Ratios compare law-level mean losses. Direct KAHM improves fixed KAHM on 8 of 9 tasks; nonlinear AR seed 73103 worsens by 0.48%. Direct RBF-local wins every AR and Van der Pol task against direct KAHM; direct locality wins every Duffing task. All per-task/horizon arrays are retained.')
p('The gate optimizer and all selection rules were frozen before D evaluation. The three seeds are 73101, 73102 and 73103. Fresh seeds test new stochastic draws from familiar laws; they do not supply untouched source collections or support broad transfer claims. No model changes were made in response to these D outcomes. [N1]','SmallCustom')

# Page 5
page('Training effects, computation and verification','04 / WHAT THE PILOT DOES AND DOES NOT ESTABLISH')
story.append(Image(str(FIG),width=508,height=165))
p('The chart compares each directly trained gate with its own fixed counterpart. It distinguishes a benefit from forecast supervision from evidence that one reconstruction family is distinctive. Three repetitions per law remain descriptive.','SmallCustom')
table(['Procedure','Median charged seconds','What is included'],[
['Fixed KAHM',f'{np.median(costs["fixed_kahm"]):.4f}','Base, bank, relevant transforms, head search and prediction'],
['Direct KAHM',f'{np.median(costs["direct_kahm"]):.4f}','Same dependencies plus the complete forecast-objective search'],
['Direct affine',f'{np.median(costs["direct_affine"]):.4f}','Matched supervision; own measured numerical work'],
['Direct locality',f'{np.median(costs["direct_locality"]):.4f}','Matched supervision; own measured numerical work'],
['Direct RBF-local',f'{np.median(costs["direct_rbf_local"]):.4f}','Matched supervision; own measured numerical work'],
['Raw history + RBF',f'{np.median(costs["raw_rbf"]):.4f}','Landmarks, kernel features, candidate fitting and prediction']],[148,117,243],padding=5)
p('Timings are single-run component sums on this host, using one numerical-library thread. They exclude data generation, imports, retained-model serialization, artifact writing and reporting. They are not repeated latency measurements or online update latency. Equal evaluation ceilings equalize optimization opportunity, not elapsed time or memory; this cost scope also differs from the historical manuscript pipeline.')
h('Audit and numerical checks')
p(f'The saved-evidence audit passed: reconstructed windows and C/D losses, recorded selection decisions, optimizer bounds/budgets, hashes and cost sums. There are {failures} fit-failure events. No full-pilot arm selected zero correction; separate unit and integration checks exercised exact baseline recovery. The audit does not independently regenerate every ODE trajectory or refit every candidate.')
p(f'{len(opt)} direct optimization instances used {sum(o["nfev"] for o in opt):,} objective evaluations in total; {sum(o["status"]=="evaluation_budget" for o in opt)} exhausted the explicit evaluation budget. Budget termination is recorded separately from convergence, with the best finite evaluated point retained, including identity.')
p('Five geometry tests include a nonzero-target check that independently reconstructs the actual B forecast objective. Eight online tests compare sliding-window ridge with independent batch solves, exercise failure paths and geometry-version guards, and verify fixed-bank query caching. These are computational checks, not extra empirical replication. [N2]','SmallCustom')

# Page 6
page('The next methodological intervention','05 / FROM TRAINED GATES TO TRAINED RECONSTRUCTION')
p('Use the gate pilot to decide whether a larger geometric parameterization is justified. The next candidate can train a KAHM bandwidth, reconstruction regularizer, or bounded horizon-specific metric through the actual forecast and ridge solve. Changing all of them together would make a favorable result difficult to attribute.')
h('A controlled sequence')
bullet('<b>Start with one intervention.</b> Hold the baseline, references, partitions, PCA rank and decoder family fixed. Compare fixed versus directly trained KAHM, and give the same target and optimization opportunity to affine, locality and RBF-local.')
bullet('<b>Separate shared supervision from reconstruction.</b> A common learned metric or a trained local affine readout is not uniquely KAHM. Matched controls must also train against the final forecast loss. Any additional KAHM parameters require their own capacity accounting.')
bullet('<b>Keep residuals attached to their baseline.</b> Learn y - f0(x) for the exact frozen baseline deployed with the correction. Changing to donor residuals or refitting f0 is a separately specified intervention, with new target alignment and validation.')
bullet('<b>Make horizon sharing explicit.</b> Compare one shared geometry with horizon-specific parameters under declared total parameter and computation budgets. Fix horizon weights before results; never select the favorable horizon on D.')
h('Differentiate the actual operator with numerical safeguards')
p('For fixed reference columns Q, let A = K + lambda I, a = solve(A,k), s = 1<super>T</super>a, and r = Qa/s. Away from s = 0, the differential is:')
eq('da = solve(A, dk - dA * a)\ndr = (Q - r * 1^T) * da / s')
p('This identity makes a central difficulty visible: KAHM weights are signed, so a small normalization denominator can amplify the derivative. Positive-definite kernel regularization and shifted Gaussian evaluation do not, by themselves, prevent cancellation. Prespecify a relative-denominator guard or another explicit rule, retain its failures/costs, and test the full declared procedure.')
p('Hard clustering, reference selection, PCA rank thresholds, the iterative reference-energy regularizer, clipped angles and rank-zero cases also need an explicit differentiation or freezing policy. A smooth surrogate or changed fallback is a new method, not a numerically invisible implementation detail.')
p('Differentiation through a closed-form ridge solve is established methodology [R1]. The research contribution would be its successful and attributable use with KAHM on the stated forecasting task, not the generic algebra itself.')

# Page 7
page('A reserved test that can support the claim','06 / CONFIRMATORY DESIGN, NOT YET EXECUTED')
p('Freeze a source roster and the complete learning-and-selection procedure before inspecting final outcomes. Existing archived collections and the new simulation pilot are now development material. A later untouched period supports temporal transfer on a known source; a genuinely new source roster addresses a different transfer question.')
table(['Design choice','Prespecified rule'],[
['Primary comparator','A non-KAHM policy selected on C from the eligible matched local and strong raw-input alternatives. Charge every candidate needed to make its choice.'],
['Mechanism contrasts','Direct KAHM versus fixed KAHM, and direct KAHM versus directly trained affine, locality and RBF-local. Retain strong fixed local and raw controls.'],
['Primary endpoint','Equal-collection mean of equal-series scaled MSE; scales use the initial training prefix. Freeze origin and horizon weights.'],
['Chronology','Train only on targets already released at each query origin. Respect the largest horizon and actual release delays at every split.'],
['Inference','Pair methods at independent source/stream units. Prespecify multiplicity and sampling assumptions; windows do not increase the source count.'],
['Failure accounting','Retain every task, fallback, warning and attempted cost. If a base forecast is unavailable, retain task failure rather than silently dropping it.']],[118,390])
h('Three separate success decisions')
p('<b>Mechanism:</b> forecast training improves fixed KAHM under the controlled design. <b>Distinctiveness:</b> the resulting KAHM procedure improves matched non-KAHM and practical raw-input alternatives on reserved evidence. <b>Utility:</b> that improvement exceeds the declared practical margin at acceptable complete cost.')
p('Statistical non-rejection is not evidence of equality. If sources are a small purposive collection, report a bounded roster result rather than treating a window bootstrap as population inference. If source-level intervals are desired, define a defensible sampling frame and use development source variability to plan sample size. Do not infer adequate power from many overlapping windows.')
p('The minimum useful accuracy gain, acceptable update/forecast latency and memory ceiling must be fixed for the intended application before confirmation. A generic 2% or 10 ms threshold would be arbitrary here. A complete accuracy-cost frontier is informative while those deployment requirements are undecided.')
p('Finite validation criteria can themselves be overfit [R3]. Accordingly, geometry-training queries B, final selection C and reserved outcomes D have separate roles. Any tuning prompted by D makes that data development evidence for the next iteration.')

# Page 8
page('Make useful geometry cheaper without stale state','07 / BOUNDED INCREMENTAL BUILDING BLOCKS')
p('The included online module implements exact-arithmetic sliding-window ridge recurrences for a fixed feature map. It does not update KAHM geometry and has not been benchmarked as a chronological forecaster. The sum-loss penalty is fixed and positive on every supplied feature, including any constant column.')
eq('G_t = alpha I + sum over active window a_i a_i^T\nC_t = sum over active window a_i r_i^T\nG_next = G_t + a_new a_new^T - a_old a_old^T\nC_next = C_t + a_new r_new^T - a_old r_old^T')
p('Rank-one inverse and coefficient recurrences update the capped window. Unsafe numerical steps trigger a logged, bounded replay solve. With p features, H outputs and W retained rows, ordinary update work is O(p² + pH); resident numerical state is O(W(p + H) + p² + pH). A replay costs O(Wp² + WpH + p³ + p²H), so a hard deadline must cover it or schedule the work explicitly.')
table(['Verified building block','Observed numerical agreement'],[
['300 sliding-window updates, W=19, p=7, H=3','Maximum coefficient difference from batch solves: 1.20 × 10<super>-14</super>; prediction difference: 3.86 × 10<super>-14</super>.'],
['Fixed-bank KAHM cache','Maximum reconstructed-query difference: 6.66 × 10<super>-16</super> on the test inputs.'],
['Version and failure guards','Eight tests pass, including forced fallback, reset/replay, capped storage, invalid-input rejection and the penalized-constant convention.']],[208,300])
h('Cache a fixed KAHM query first')
eq('A_cache = Q B; b_cache = B^T 1\nr(x) = A_cache k(x) / (b_cache^T k(x))')
p('For the fitted symmetric inverse, B<super>T</super>1 = B1. This algebra removes the repeated m-by-m multiplication from a query. Constructing the cache costs O(dm²); reconstruction then costs O(dm), excluding kernel evaluation. The separate cache rejects zero or relatively cancelled denominators, so its guard differs from the pilot’s historical exact-zero convention. Neither cache equivalence nor its complexity is an empirical speed benchmark.')
h('A geometry refresh invalidates more than a kernel factor')
p('References normally affect centers, PCA/covariance, the energy-based regularizer, gates and all head features. A changed scaler or metric similarly invalidates downstream state. Even a rank-two reference change under a frozen kernel changes normalized features for stored rows. Recompute features and residual targets from a bounded raw-data buffer, rebuild the head, and atomically publish a new model version. Never combine statistics from different feature maps.')
p('Fixed-budget kernel recursive methods are established [R2]. A KAHM systems contribution must show a better measured accuracy-cost tradeoff or a rigorously defined equivalent computation, including refreshes and fallback work.')

# Page 9
page('The chronological experiment and next decisions','08 / STREAMING PROTOCOL AND STOPPING RULES')
p('Use immutable geometry epochs with bounded, explicit refreshes. The active model predicts while a candidate is built from a fixed past-data watermark. Limit references, replay rows, partition iterations and optimizer evaluations; include active and candidate states in the memory budget.')
table(['Event','Causal rule'],[
['Issue forecast','Record origin, horizon, model version, raw forecast, correction and final prediction before the target is available.'],
['Receive label','Score the stored prediction. Only then may the released label influence future model state.'],
['Update readout','For a full multioutput target, wait until every required label is released. With no release delay: origin + largest horizon ≤ current time.'],
['Refresh geometry/base','Recompute replay features and residual targets for the new versions. Publish atomically or retain the previous usable model.'],
['Retire or reactivate','Use only matured losses from stored contemporaneous shadow forecasts. Freeze windows, thresholds, probes and tie handling; charge their cost.']],[119,389])
h('Compare the whole deployed procedure')
p('Include static models, fixed-geometry/readout-update models, bounded-refresh models, and scheduled full-refresh references. Give raw, affine, locality and RBF-local competitors the same label availability and refresh opportunities. Also test accuracy under equal measured time and memory ceilings. A cheaper KAHM implementation is not yet a competitive forecaster if simpler methods retain a better frontier.')
p('Measure forecast latency and update latency separately: median, p95, p99, maximum, refresh pause and publication delay. Repeat on declared hardware/thread settings. Report resident model bytes, raw replay, pending-label queues, working allocations and process peak memory. Distinguish bounded active state from disk audit logs that grow with the stream.')
h('Recommended order of work')
bullet('<b>Now:</b> inspect the complete pilot and use it to decide whether to develop a richer forecast-trained reconstruction. The saved outcome is not a reason to bypass the matched alternatives.')
bullet('<b>Before a new claim:</b> obtain the original reproduction dependencies, select the practical baseline portfolio and application budgets, then seal genuinely reserved sources or periods and the analysis plan.')
bullet('<b>After useful geometry is identified:</b> integrate exact fixed-map updates and budgeted geometry refreshes; run the separate delayed-label stream comparison with full timing and memory measurement.')
bullet('<b>Stop or simplify when appropriate:</b> if matched simple gates continue to match or beat KAHM, prefer the simpler predictor for that scope. Retain negative outcomes and do not expand complexity solely because an optimizer lowers its training objective.')

# Page 10
page('Evidence, artifacts and reproducibility','09 / SOURCE NOTES')
h('Manuscript sources')
p('<b>[M1]</b> Supplied <i>KAHM_Prediction_Revised_Manuscript_Source.zip</i>: section_target_geometry.tex; appendix_target_geometry.tex; generated/target_geometry/archive_main.tex, synthetic_main.tex, synthetic_intervals.tex, diagnostics.tex and source tables. Reported latest-stage results and scope.','SmallCustom')
p('<b>[M2]</b> Same attachment: natural_study.tex, scaling_sensitivity.tex, utility_extension.tex, discussion.tex; generated/natural_main_table.tex and utility/scaling displays. Historical limitations remain part of the evidence.','SmallCustom')
p('<b>[M3]</b> Same attachment: README_RELEASE.md and ARTIFACT_MANIFEST.json. All 118 listed files independently matched their recorded sizes and hashes; the manifest itself is the 119th archive file. Original experimental forecasts/code are referenced separately.','SmallCustom')
h('New results and verification')
p('<b>[N1]</b> This workbench: pilot_results/protocol.json and freeze.json; source_snapshot/; tasks/*/inputs.npz, predictions.npz, diagnostics.json and task.json; task_metrics.csv; summary.json; audit_report.json. Nine exploratory tasks, 11 methods, 832 D origins and three horizons per task. No confirmatory interval or new-source result.','SmallCustom')
p('<b>[N2]</b> check_geometry.json, check_online.json, test_geometry.py, test_online.py. These verify defined numerical properties on test inputs and do not prove prediction benefit, universal stability, latency or memory performance.','SmallCustom')
h('Primary methodological references')
p('<b>[R1]</b> Bertinetto, Henriques, Torr and Vedaldi (2019). <i>Meta-learning with differentiable closed-form solvers.</i> ICLR. Backpropagation through efficient regression/classification solvers is an established method.<br/><link href="https://openreview.net/forum?id=HyxnZh0ct7" color="#157d85">openreview.net/forum?id=HyxnZh0ct7</link>','SmallCustom')
p('<b>[R2]</b> Van Vaerenbergh, Santamaria, Liu and Principe (2010). <i>Fixed-Budget Kernel Recursive Least-Squares.</i> ICASSP, pp. 1882-1885. Fixed-memory kernel fitting and dictionary growth/pruning precede this proposal.<br/><link href="https://gtas.unican.es/files/pub/vanvaerenbergh2010_fbkrls.pdf" color="#157d85">gtas.unican.es/files/pub/vanvaerenbergh2010_fbkrls.pdf</link>','SmallCustom')
p('<b>[R3]</b> Cawley and Talbot (2010). <i>On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.</i> JMLR 11:2079-2107.<br/><link href="https://jmlr.org/papers/v11/cawley10a.html" color="#157d85">jmlr.org/papers/v11/cawley10a.html</link>','SmallCustom')
h('Run the supplied evidence audit')
eq('python run_pilot.py --audit --output pilot_results\npython test_geometry.py\npython test_online.py')
p('For a fresh run, choose an empty output directory and use <font face="Mono">python run_pilot.py --run --output ../new_pilot</font>. Exact tested package versions, seeds and numerical conventions are recorded. The runner refuses to overwrite a nonempty result directory. Fresh refitting is a reproducibility check, not additional source evidence.','SmallCustom')
h('Boundaries of this delivery')
p('The original manuscript is unchanged. The attached source package was inspected, but its reported prediction audit was not independently reproduced. The new code is an exploratory implementation from published equations; no full streaming benchmark, natural-source confirmation, deployment budget test or submission was completed. Those are the explicitly identified next research steps.','SmallCustom')

OUT.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=43,bottomMargin=53,
                      title='KAHM Research Continuation: Direct Forecast Training and Bounded Updates',author='Research working draft')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(json.dumps({'pdf':str(OUT),'figure':str(FIG),'result_brief':str(PROJECT/'PILOT_FINDINGS.json')},indent=2))
