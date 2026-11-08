"""Unchanged fit_affine function extracted from Step38, without unused QP dependencies."""
from fractions import Fraction as F
import numpy as np
import finite_feature35 as ff

def fit_affine(z,y,*,feature_id,kappa="2",ridge_lambda="1/1000000",training_binding=None):
    z=np.asarray(z,dtype=np.float64);ys=list(map(F,y));yf=np.asarray([float(v) for v in ys])
    xc=z-z.mean(axis=0);yc=yf-yf.mean()
    beta=np.linalg.solve(xc.T@xc+len(z)*float(F(ridge_lambda))*np.eye(z.shape[1]),xc.T@yc)
    intercept=float(yf.mean()-z.mean(axis=0)@beta)
    center=[sum((F.from_float(float(v)) for v in z[:,j]),F(0))/len(z) for j in range(z.shape[1])]
    receipt=ff.make_affine_head(beta.tolist(),intercept,feature_id=feature_id,kappa=kappa,output_interval=("0","1"),
        project=True,center_features=center,center_target=sum(ys,F(0))/len(ys),bits=80)
    receipt["training_binding"]=training_binding or {};return receipt
