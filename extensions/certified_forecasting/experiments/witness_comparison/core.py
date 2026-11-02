"""Finite-support lower certificates for one common scalar Lipschitz class.

No fitted primal QP objective is ever used as an empirical-risk lower bound.
Every such bound is evaluated at nonnegative dual multipliers.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np
from scipy.optimize import minimize, nnls, brentq

ETA = 0.05
SIZES = (64, 256, 1024, 4096)
REPETITIONS = 300
SEED = 202610290731
TOLERANCES = (0.10, 0.25, 0.50)
PRIMARY = ('pair_eb', 'centered_range', 'erm_kl', 'stratified_eb')
METHODS = PRIMARY + ('pair_support_eb', 'stratified_support_eb',
                     'stratified_knownmass_support_eb', 'erm_hoeffding')

@dataclass(frozen=True)
class Law:
    name: str
    u: tuple[float, ...]
    y: tuple[float, ...]
    probabilities: tuple[float, ...]
    strata: tuple[int, ...]
    kappa: float = 1.0

    def arrays(self):
        return tuple(np.asarray(x, dtype=dtype) for x, dtype in
                     ((self.u, float), (self.y, float), (self.probabilities, float),
                      (self.strata, int)))

    def score_matrix(self):
        u, y, _, _ = self.arrays()
        return 0.5 * np.maximum(np.abs(y[:, None]-y[None, :])-
                                self.kappa*np.abs(u[:, None]-u[None, :]), 0.0)**2


def laws():
    a = 0.6/math.sqrt(2)
    return (
        Law('collapsed_binary', (0.,0.), (-1.,1.), (.5,.5), (0,0)),
        Law('two_atom_partial', (-a,a), (-1.,1.), (.5,.5), (0,0)),
        Law('three_point_v', (-1.,0.,1.), (1.,-1.,1.), (1/3,)*3, (0,0,1)),
        Law('five_point_zigzag', (-1.,-.5,0.,.5,1.), (-1.,1.,-1.,1.,-1.),
            (.2,)*5, (0,0,1,1,1)),
        Law('partial_collapse', (-1.,-1.,0.,0.,1.,1.), (-1.,.2,-.2,.2,-.2,1.),
            (1/6,)*6, (0,0,1,1,2,2)),
        Law('noisy_affine', (-1.,-1.,0.,0.,1.,1.), (-1.,0.,-.5,.5,0.,1.),
            (1/6,)*6, (0,0,1,1,2,2)),
        Law('rare_conflict', (3.,0.,1.), (0.,-1.,1.), (.9,.05,.05), (0,1,1)),
        Law('affine_mismatch', (-1.,0.,1.), (-.5,.5,-.5), (1/3,)*3, (0,0,0)),
    )


def grouped_objective(law, atom_weights):
    """Aggregate positive-weight atoms; missing support extends Lipschitzly."""
    u, y, _, _ = law.arrays()
    atom_weights = np.asarray(atom_weights, float)
    active = atom_weights > 0
    unique, inverse = np.unique(u[active], return_inverse=True)
    weights = np.bincount(inverse, weights=atom_weights[active], minlength=len(unique))
    moments = np.bincount(inverse, weights=atom_weights[active]*y[active], minlength=len(unique))
    means = moments/weights
    within = float(np.sum(atom_weights[active]*(y[active]-means[inverse])**2))
    j = len(unique)
    difference = np.zeros((max(0,j-1),j))
    for i in range(j-1):
        difference[i,i], difference[i,i+1] = -1., 1.
    matrix = np.vstack((difference,-difference,np.eye(j),-np.eye(j)))
    bound = np.concatenate((law.kappa*np.diff(unique),law.kappa*np.diff(unique),
                            np.ones(2*j)))
    return unique, weights, means, within, matrix, bound


def qp_values(weights, means, within, matrix, bound, fitted, multipliers):
    """Feasible primal and dual objectives, with explicit roundoff guards."""
    fitted = np.asarray(fitted, float)
    multipliers = np.maximum(np.asarray(multipliers, float), 0.)
    unconstrained = means-matrix.T@multipliers/(2*weights)
    terms = weights*(unconstrained-means)**2
    penalty = multipliers*(matrix@unconstrained-bound)
    dual_raw = float(within+terms.sum()+penalty.sum())
    guard = 1e-11*(1+abs(within)+np.abs(terms).sum()+np.abs(penalty).sum())
    lower = max(0.,dual_raw-float(guard))
    primal_raw = float(within+np.sum(weights*(fitted-means)**2))
    upper = primal_raw+1e-11*(1+abs(primal_raw))
    violation = float(max(0.,np.max(matrix@fitted-bound)))
    return lower, upper, violation, dual_raw, float(guard)


def feasible_contract(vector, matrix, bound):
    """Contract toward the feasible zero function; all constraints are linear."""
    vector = np.asarray(vector, float)
    values = matrix@vector
    positive = values > bound
    scale = min(1.,float(np.min(bound[positive]/values[positive]))) if positive.any() else 1.
    return vector*scale*(1-16*np.finfo(float).eps)


def solve_lipschitz(law, atom_weights):
    unique, weights, means, within, matrix, bound = grouped_objective(law,atom_weights)
    def objective(f):
        return float(np.sum(weights*(f-means)**2))
    def gradient(f):
        return 2*weights*(f-means)
    initial = np.full(len(unique),float(weights@means/weights.sum()))
    unconstrained_ok = bool(np.max(matrix@means-bound)<=0.)
    if unconstrained_ok:
        fitted = feasible_contract(means,matrix,bound)
        multipliers = np.zeros(len(bound))
        iterations, fallback = 0, False
    else:
        result = minimize(objective,initial,jac=gradient,method='SLSQP',
                          constraints={'type':'ineq','fun':lambda f:bound-matrix@f,
                                       'jac':lambda f:-matrix},
                          options={'ftol':1e-13,'maxiter':1000})
        fitted = feasible_contract(result.x,matrix,bound)
        active = np.flatnonzero(bound-matrix@result.x<1e-7)
        multipliers = np.zeros(len(bound))
        if len(active):
            mult,_ = nnls(matrix[active].T,-gradient(result.x),maxiter=1000)
            multipliers[active] = mult
        iterations, fallback = int(result.nit), False
    lower, upper, violation, raw, guard = qp_values(weights,means,within,matrix,bound,fitted,multipliers)
    if upper-lower>1e-8 or violation>1e-10:
        fallback = True
        def dual_objective(mult):
            f = means-matrix.T@mult/(2*weights)
            value = within+np.sum(weights*(f-means)**2)+mult@(matrix@f-bound)
            return -float(value), bound-matrix@f
        result = minimize(dual_objective,multipliers,jac=True,method='L-BFGS-B',
                          bounds=[(0.,None)]*len(bound),
                          options={'ftol':1e-15,'gtol':1e-11,'maxiter':5000,'maxls':100})
        multipliers = np.maximum(result.x,0.)
        candidate = feasible_contract(means-matrix.T@multipliers/(2*weights),matrix,bound)
        if objective(candidate)<objective(fitted):
            fitted=candidate
        lower, upper, violation, raw, guard=qp_values(weights,means,within,matrix,bound,fitted,multipliers)
        iterations += int(result.nit)
    if upper-lower>1e-8 or violation>1e-10:
        raise ArithmeticError(f'Unresolved QP bracket {law.name}: gap={upper-lower}, violation={violation}')
    return {'lower':lower,'upper':upper,'gap':upper-lower,'primal':fitted.tolist(),
            'dual_multipliers':multipliers.tolist(),'support':unique.tolist(),
            'feasibility_residual':violation,'dual_raw':raw,'dual_guard':guard,
            'iterations':iterations,'fallback':fallback}


def affine_minimum(law):
    """Population affine diagnostic by enumerating all 0/1/2 active sets."""
    u,y,p,_=law.arrays()
    if np.ptp(u)==0:
        return float(p@(y-p@y)**2)
    features=np.column_stack((np.ones(len(u)),u))
    hessian=features.T@(p[:,None]*features)
    linear=features.T@(p*y)
    matrix=np.vstack((features,-features,[[0.,1.],[0.,-1.]]))
    bound=np.concatenate((np.ones(2*len(u)),[law.kappa,law.kappa]))
    candidates=[]
    for count in (0,1,2):
        for active in combinations(range(len(bound)),count):
            if count:
                a=matrix[list(active)]
                block=np.block([[hessian,a.T],[a,np.zeros((count,count))]])
                rhs=np.concatenate((linear,bound[list(active)]))
            else:
                block,rhs=hessian,linear
            try:
                solution=np.linalg.solve(block,rhs)[:2]
            except np.linalg.LinAlgError:
                continue
            if np.max(matrix@solution-bound)<=1e-9:
                candidates.append(float(p@(y-features@solution)**2))
    if not candidates:
        raise ArithmeticError('No feasible affine active set')
    return min(candidates)


def eb_lower(mean,variance,count,bound,eta):
    if bound==0 or count<2:
        return 0.
    radius=math.sqrt(2*max(0.,variance)*math.log(2/eta)/count)+7*bound*math.log(2/eta)/(3*(count-1))
    return max(0.,mean-radius)


def pair_summary(sample,score_matrix):
    end=2*(len(sample)//2)
    scores=score_matrix[sample[:end:2],sample[1:end:2]]
    if len(scores)==0:
        return {'mean':0.,'variance':0.,'n_pairs':0}
    return {'mean':float(scores.mean()),'variance':float(scores.var(ddof=1)) if len(scores)>1 else 0.,
            'n_pairs':len(scores)}


def centered_from_counts(law,counts,eta=ETA):
    u,y,_,_=law.arrays()
    n=int(np.sum(counts)); weights=np.asarray(counts,float)/n
    var_y=float(weights@(y-weights@y)**2)
    var_u=float(weights@(u-weights@u)**2)
    range_y,range_u=float(np.ptp(y)),float(np.ptp(u))
    alpha=eta/2 if range_u==0 else eta/3
    sd_y=math.sqrt(max(0.,var_y)*n/(n-1))
    target_direct=max(0.,sd_y-range_y*math.sqrt(2*math.log(1/alpha)/(n-1)))
    target_variance=math.sqrt(max(0.,var_y-range_y**2*math.sqrt(math.log(1/alpha)/(2*n))))
    lower_sd=max(target_direct,target_variance)
    upper_sd=0. if range_u==0 else min(range_u/2,math.sqrt(max(0.,var_u)*n/(n-1))+
                                      range_u*math.sqrt(2*math.log(1/alpha)/(n-1)))
    endpoint=max(0.,lower_sd-law.kappa*upper_sd)**2
    return endpoint,{'target_sd_direct':target_direct,'target_sd_variance':target_variance,
                     'input_sd_upper':upper_sd,'internal_tail':alpha}


def kl_bernoulli(q,p):
    if q==0:
        return -math.log1p(-p)
    if q==1:
        return -math.log(p)
    if p==0 or p==1:
        return math.inf
    return q*math.log(q/p)+(1-q)*math.log((1-q)/(1-p))


def erm_kl_lower(dual,n,eta=ETA,bound=4.):
    q=min(1.,max(0.,dual/bound))
    if q==0:
        return 0.
    target=math.log(1/eta)/n
    if q==1:
        return bound*math.exp(-target)
    # q>0 and the desired lower root is strictly between zero and q.
    low=np.finfo(float).tiny
    root=brentq(lambda p:kl_bernoulli(q,p)-target,low,q,xtol=5e-15,rtol=1e-14)
    return max(0.,bound*root-1e-12)


def mass_minimum(values,counts,n,eta):
    """Exact linear minimization over simultaneous Hoeffding mass intervals."""
    s=len(values)
    radius=math.sqrt(math.log(4*s/eta)/(2*n))
    frequencies=np.asarray(counts,float)/n
    lower=np.maximum(0.,frequencies-radius)
    upper=np.minimum(1.,frequencies+radius)
    probabilities=lower.copy()
    left=1-float(probabilities.sum())
    for i in np.argsort(values,kind='stable'):
        added=min(left,float(upper[i]-probabilities[i]))
        probabilities[i]+=added
        left-=added
    if abs(left)>1e-10:
        raise ArithmeticError('Mass intervals do not intersect simplex')
    return max(0.,float(probabilities@values)-1e-12), probabilities.tolist(), radius


def stratified_from_summaries(law,summaries,counts,n,eta=ETA,support=False,known_masses=False):
    _,_,probabilities,strata=law.arrays()
    score=law.score_matrix()
    s=len(np.unique(strata))
    beta=eta/s if known_masses else eta/(2*s)
    values=[]
    for label,summary in enumerate(summaries):
        indices=np.flatnonzero(strata==label)
        cap=float(score[np.ix_(indices,indices)].max()) if support else 2.
        values.append(eb_lower(summary['mean'],summary['variance'],summary['n_pairs'],cap,beta))
    if known_masses:
        masses=np.bincount(strata,weights=probabilities,minlength=s)
        return max(0.,float(masses@values)-1e-12),{'conditional_lower':values,'masses':masses.tolist(),'mass_radius':0.}
    endpoint,masses,radius=mass_minimum(values,counts,n,eta)
    return endpoint,{'conditional_lower':values,'masses':masses,'mass_radius':radius}


def population(law):
    u,y,p,strata=law.arrays()
    scores=law.score_matrix()
    pair=float(p@scores@p)
    sigma=math.sqrt(float(p@(y-p@y)**2))
    tau=math.sqrt(float(p@(u-p@u)**2))
    centered=max(0.,sigma-law.kappa*tau)**2
    stratified=0.
    conditional=[]
    for label in np.unique(strata):
        indices=np.flatnonzero(strata==label)
        mass=float(p[indices].sum())
        local=p[indices]/mass
        witness=float(local@scores[np.ix_(indices,indices)]@local)
        stratified+=mass*witness
        conditional.append({'stratum':int(label),'mass':mass,'witness':witness,
                            'score_bound':float(scores[np.ix_(indices,indices)].max())})
    oracle=solve_lipschitz(law,p)
    return {'name':law.name,'u':list(law.u),'y':list(law.y),'probabilities':list(law.probabilities),
            'strata':list(law.strata),'kappa':law.kappa,'oracle':oracle,
            'pair_witness':pair,'centered_bound':centered,'stratified_witness':stratified,
            'conditional_witnesses':conditional,'affine_minimum':affine_minimum(law)}
