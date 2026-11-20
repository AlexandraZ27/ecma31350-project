"""Known-truth check of the honest-forest exact-loss NPIV: Y = 10*A + 8*u, A = c*z + u-part, f(a)=10a."""
import sys; sys.argv=["x","d","o","0","0"]; sys.path.insert(0,"npiv")
import numpy as np, npiv
npiv.M=M=20
for zc,nu in [(1.0,0.3),(0.4,0.9)]:
    rng=np.random.default_rng(3); n=1010
    x=rng.standard_normal((n,2)); z=rng.standard_normal(n); u=rng.standard_normal(n)
    a=zc*z+nu*u+0.3*rng.standard_normal(n); y=10*a+8*nu*u
    va=a.var()-zc**2*z.var(); at=zc*z[:,None]+np.sqrt(va)*rng.standard_normal((n,M))
    (alpha,lam),cv=npiv.cv_select(at,x,y,np.arange(n))
    f=npiv.fit_forest(at,x,y,alpha,lam,200,1)
    print(f"zcoef={zc} noiseU={nu} R2(A|z)={zc**2/a.var():.2f}: true 10 | NPIV at steps {npiv.DELTAS} sd:",
          np.round(npiv.avg_deriv(f,a,x,npiv.DELTAS),2),"alpha",alpha,"ridge",lam)
