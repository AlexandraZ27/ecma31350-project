"""Replicate Tables 1-4 and 6-9 of Ajzenman, Dominguez & Undurraga (AEJ:Applied 2023).
Usage (from project/): python -I rep_scripts/replicate.py rep_data/Replication/data/analysis rep_results
Table 5 (homicides) needs confidential data not in the package."""
import sys, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
D, OUT = sys.argv[1], sys.argv[2]

Y = {
 1: ["del_problema3","del_afecta3","del_calvida3","del_inseg5","del_vict","pc_19"],
 2: ["del_tend_barrio","del_tend_com","del_tend_pais"],
 3: ["Index_Vivienda","Index_Vecinos","del_arma","pc_20"],
 4: ["vict_hurto","vict_robosorpr","vict_robovehi","vict_roboviv","vict_lesiones","vict_roboviolent","vict_agreg"],
}
ols = pd.read_stata(f"{D}/enusc_OLS.dta", convert_categoricals=False,
        columns=["lnrate_stock_imm","year","edad","ismale","cod_com"]+sum(Y.values(),[]))
iv = pd.read_stata(f"{D}/enusc_IV.dta", convert_categoricals=False)

def areg(d, y):
    """Stata: areg y lnrate_stock_imm i.year edad ismale, absorb(cod_com) vce(cluster cod_com)"""
    cols=["lnrate_stock_imm","edad","ismale"]
    d=d[[y,"year","cod_com"]+cols].dropna()
    X=pd.concat([d[cols], pd.get_dummies(d["year"],prefix="y",drop_first=True).astype(float)],axis=1)
    g=d["cod_com"].values; yv=d[y].values.astype(float)
    Xd=X.values.astype(float); Xm=Xd-pd.DataFrame(Xd).groupby(g).transform("mean").values
    ym=yv-pd.Series(yv).groupby(g).transform("mean").values
    b=np.linalg.lstsq(Xm,ym,rcond=None)[0]; e=ym-Xm@b
    N,K=Xm.shape; G=len(np.unique(g))
    XtXi=np.linalg.inv(Xm.T@Xm)
    sc=pd.DataFrame(Xm*e[:,None]).groupby(g).sum().values
    V=XtXi@(sc.T@sc)@XtXi*(G/(G-1))*((N-1)/(N-K))   # FE nested in clusters: not counted in K
    se=np.sqrt(V[0,0]); pv=2*(1-stats.t.cdf(abs(b[0]/se),G-1))
    r2=1-(e@e)/((yv-yv.mean())@(yv-yv.mean()))        # areg R2 includes absorbed FE
    return b[0],se,pv,N,r2
lines=[]
def p(s=""): print(s); lines.append(s)
def cell(b,se,pv): return f"{b:7.2f} ({se:.2f}) [{pv:.3f}]"

# ---- Tables 1-4: two-way FE, cluster by municipality
dv_mean={}
for t,ys in Y.items():
    p(f"\n=== Table {t} (TWFE) ===")
    for y in ys:
        d=ols.copy(); d[y]=100*d[y]; dv_mean[y]=d[y].mean()
        c,se,pv,N,r2=areg(d,y)
        p(f"{y:18s} {cell(c,se,pv)}  N={N:,}  R2={r2:.2f}  meanDV={dv_mean[y]:.2f}")

# ---- Tables 6-9: OLS & 2SLS in differences, robust SE
def fwl(v,Z): return v-Z@np.linalg.lstsq(Z,v,rcond=None)[0]
def run(y, hc):
    yy=100*iv[y].values; x=iv["deltaimm"].values; z=iv["deltaimm_instr"].values
    W=np.column_stack([np.ones(len(iv)),iv["age"],iv["hombre"]]); n=len(iv)
    def sandwich(X,Zi,e,k,hc=hc):  # just-identified/OLS: (Zi'X)^-1 Zi' diag(e^2) Zi (X'Zi)^-1
        A=np.linalg.inv(Zi.T@X); S=(Zi*e[:,None]**2).T@Zi
        V=A@S@A.T; return V*(n/(n-k) if hc==1 else 1.0)
    X=np.column_stack([x,W]); Zi=np.column_stack([z,W])
    bo=np.linalg.lstsq(X,yy,rcond=None)[0]; Vo=sandwich(X,X,yy-X@bo,X.shape[1])
    bi=np.linalg.solve(Zi.T@X,Zi.T@yy); Vi=sandwich(X,Zi,yy-X@bi,X.shape[1],hc=0)  # ivreg2, robust: no small-sample adj.
    # first stage
    Zf=np.column_stack([z,W]); bf=np.linalg.lstsq(Zf,x,rcond=None)[0]; Vf=sandwich(Zf,Zf,x-Zf@bf,Zf.shape[1])
    F=bf[0]**2/Vf[0,0]; rz,rx=fwl(z,W),fwl(x,W); pr2=np.corrcoef(rz,rx)[0,1]**2
    out=[]
    for b,V,normal in ((bo,Vo,False),(bi,Vi,True)):   # reg,robust -> t; ivreg2 -> z
        se=np.sqrt(V[0,0]); out.append((b[0],se,2*(1-stats.norm.cdf(abs(b[0]/se))) if normal else 2*(1-stats.t.cdf(abs(b[0]/se),n-X.shape[1]))))
    return out,(bf[0],np.sqrt(Vf[0,0]),F,pr2)
HC=int(sys.argv[3]) if len(sys.argv)>3 else 1
for t,ys in Y.items():
    p(f"\n=== Table {t+5} (OLS: HC1 / 2SLS: HC0, as in Stata reg/ivreg2) ===")
    for y in ys:
        (o,i),fs=run(y,HC)
        p(f"{y:18s} OLS {cell(*o)}   IV {cell(*i)}   meanDV={dv_mean[y]:.2f}")
    p(f"First stage: {fs[0]:.2f} ({fs[1]:.2f})  F={fs[2]:.2f}  partial R2={fs[3]:.2f}   N={len(iv)}")
open(f"{OUT}/replication_output.txt","w").write("\n".join(lines))
