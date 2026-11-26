"""KIV with commune/year indicators INSIDE the structural and conditioning kernels.

No additive commune/year terms and no fixed-effect absorption in KIV. Numerical
routines and data loading are shared with the archived additive-effects runner.
Raw 2008 shares are functions of commune ID: this specification does NOT identify
causal treatment derivatives from those excluded instruments. Outputs are
regularized fitted derivatives and prediction diagnostics, not causal inference.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ecma31350-kernel-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("kiv_absorbed_reference",HERE/"run_kernel_absorbed.py")
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
IDENTIFICATION="Raw shares are deterministic given commune identity; share-times-shock components are deterministic given commune and year. With these indicators unrestricted inside f and included as first-stage controls, the supplied IVs have no additional conditional information. Fitted derivatives are regularization-driven descriptive quantities, not identified causal effects."


def inputs(d):
    mode=d.attrs.get("instrument_mode","bartik")
    instruments=b.SHARES if mode=="raw_shares" else b.COMPONENTS if mode=="components" else ["Z"]
    S=d[["A","age","male_share","cod_com","year"]].to_numpy(float)
    V=d[instruments+["age","male_share","cod_com","year"]].to_numpy(float)
    return S,V


def widths(x, continuous_multiplier=1., categorical_multiplier=1.):
    # Last two columns are categorical LABELS, not numeric covariates. A common
    # width 1 is equivalent to positive-distance median 1 for each one-hot bit.
    return np.r_[b.widths(x[:,:-2])*continuous_multiplier,
                 np.repeat(categorical_multiplier,2)]


def mixed_kernel(x,z,h):
    k=np.exp(-.5*cdist(x[:,:-2]/h[:-2],z[:,:-2]/h[:-2],"sqeuclidean"))
    for j in [-2,-1]:
        k*=np.exp(-(x[:,j,None]!=z[None,:,j]).astype(float)/h[j]**2)
    return k


class WeightOnly:
    """Sample weights only. Identity at equal weights: no FE projection."""
    def __init__(self,d,weighting="equal"):
        w=np.ones(len(d)) if weighting=="equal" else d.respondents.to_numpy(float)
        self.sw=np.sqrt(w/w.mean())
    def transform(self,x):
        return self.sw*x if x.ndim==1 else self.sw[:,None]*x
    def cross_kernel(self,k,other):
        return self.sw[:,None]*k*other.sw[None,:]
    def raw_coefficients(self,a):return self.sw*a


def blocks(d1,d2,weighting="equal",structural_bandwidth=1.,instrument_bandwidth=1.,
           categorical_bandwidth=1.,excluded=True,shared_widths=None):
    s1,v1=inputs(d1);s2,v2=inputs(d2)
    if shared_widths is None:
        hs=widths(s1,structural_bandwidth,categorical_bandwidth)
        hv=widths(v1,1.,categorical_bandwidth)
    else:hs,hv=shared_widths
    hv=hv.copy();n_excluded=v1.shape[1]-4
    hv[:n_excluded]*=instrument_bandwidth
    if not excluded:v1,v2,hv=v1[:,n_excluded:],v2[:,n_excluded:],hv[n_excluded:]
    p1,p2=WeightOnly(d1,weighting),WeightOnly(d2,weighting)
    ks=p1.cross_kernel(mixed_kernel(s1,s1,hs),p1)
    ks12=p1.cross_kernel(mixed_kernel(s1,s2,hs),p2)
    ks22=p2.cross_kernel(mixed_kernel(s2,s2,hs),p2)
    kv=p1.cross_kernel(mixed_kernel(v1,v1,hv),p1)
    kv12=p1.cross_kernel(mixed_kernel(v1,v2,hv),p2)
    return p1,p2,s1,hs,hv,(kv,kv12,ks,ks12,ks22)


def fit(d1,d2,weighting="equal",structural_bandwidth=1.,instrument_bandwidth=1.,categorical_bandwidth=1.):
    p1,p2,s1,hs,hv,matrices=blocks(d1,d2,weighting,structural_bandwidth,instrument_bandwidth,categorical_bandwidth)
    gamma,meta=b.first_stage(*matrices)
    ks=matrices[2];y1=p1.transform(d1.y.to_numpy(float));y2=p2.transform(d2.y.to_numpy(float))
    ev,U=b.spectrum(gamma.T@ks@gamma);R=gamma@U;pred=ks@R;uy=U.T@y2
    grid=[];best=None
    for xi in b.XI_GRID:
        coeff=uy/(ev+len(d2)*xi)
        loss=float(np.mean((y1-pred@coeff)**2))
        if not np.isfinite(loss):raise FloatingPointError("Invalid structural validation loss")
        grid.append({"xi":float(xi),"structural_validation_mse":loss})
        if best is None or loss<best[0]:best=(loss,xi,R@coeff)
    loss,xi,alpha=best
    if not np.isfinite(alpha).all():raise FloatingPointError("Nonfinite coefficients")
    meta.update({"xi":float(xi),"xi_grid":grid,"structural_validation_mse":loss,
        "xi_at_grid_boundary":bool(xi==b.XI_GRID[0] or xi==b.XI_GRID[-1]),
        "second_stage_effective_df":float(np.sum(ev/(ev+len(d2)*xi))),
        "structural_bandwidths":hs.tolist(),"conditioning_bandwidths":hv.tolist(),
        "instrument_mode":d1.attrs.get("instrument_mode","bartik"),
        "categorical_bandwidth":categorical_bandwidth,"instrument_bandwidth_multiplier":instrument_bandwidth,
        "n_stage1":len(d1),"n_stage2":len(d2),"fixed_effect_absorption":False,
        "stage1_communes":sorted(d1.cod_com.unique().tolist()),"stage2_communes":sorted(d2.cod_com.unique().tolist()),
        "stage1_cells":d1[["cod_com","year"]].to_numpy(int).tolist(),
        "stage2_cells":d2[["cod_com","year"]].to_numpy(int).tolist()})
    return {"centers":s1,"widths":hs,"coef":p1.raw_coefficients(alpha)},meta


def evaluate(model,s):
    k=mixed_kernel(s,model["centers"],model["widths"])
    values=k@model["coef"]
    derivatives=(k*(model["centers"][:,0][None,:]-s[:,0,None])/model["widths"][0]**2)@model["coef"]
    if not np.isfinite(values).all() or not np.isfinite(derivatives).all():raise FloatingPointError("Nonfinite evaluation")
    return values,derivatives


def split_cells(d,seed):
    # Both stages must see each commune to fit its indicator-dependent terms.
    rng=np.random.default_rng(seed);left=[]
    for _,g in d.groupby("cod_com",sort=True):
        left.extend(rng.permutation(g.index.to_numpy())[:len(g)//2].tolist())
    left=set(left)
    return d.loc[d.index.isin(left)].copy(),d.loc[~d.index.isin(left)].copy()


def diagnostic_folds(d,seed,kind):
    rng=np.random.default_rng(seed)
    if kind=="unseen_communes":
        return [d.cod_com.isin(ids).to_numpy() for ids in np.array_split(rng.permutation(np.sort(d.cod_com.unique())),5)]
    # Cell interpolation: two held-out years per commune; all commune categories
    # remain in training. This is NOT independent-cluster validation.
    assignment=pd.Series(-1,index=d.index,dtype=int)
    for _,g in d.groupby("cod_com",sort=True):
        for k,idx in enumerate(np.array_split(rng.permutation(g.index.to_numpy()),5)):
            assignment.loc[idx]=k
    return [(assignment==k).to_numpy() for k in range(5)]


def outer_diagnostics(d,seed,kind="unseen_communes",instrument_bandwidth=1.,categorical_bandwidth=1.):
    rows=[]
    for fold,held in enumerate(diagnostic_folds(d,seed,kind)):
        train,test=d.loc[~held],d.loc[held]
        a,c=split_cells(train,seed+100+fold)
        shared_inner=(widths(inputs(a)[0],categorical_multiplier=categorical_bandwidth),widths(inputs(a)[1],categorical_multiplier=categorical_bandwidth))
        shared_outer=(widths(inputs(train)[0],categorical_multiplier=categorical_bandwidth),widths(inputs(train)[1],categorical_multiplier=categorical_bandwidth))
        row={"fold":fold,"test_n":len(test),"train_communes":sorted(train.cod_com.unique().tolist()),
             "test_communes":sorted(test.cod_com.unique().tolist()),"test_cells":test[["cod_com","year"]].to_numpy(int).tolist(),
             "train_test_commune_overlap":len(set(train.cod_com)&set(test.cod_com))}
        for include,label in [(True,"with_Z"),(False,"without_Z")]:
            *_,inner=blocks(a,c,excluded=include,shared_widths=shared_inner,instrument_bandwidth=instrument_bandwidth)
            _,chosen=b.first_stage(*inner)
            p1,p2,_,_,_,outer=blocks(train,test,excluded=include,shared_widths=shared_outer,instrument_bandwidth=instrument_bandwidth)
            gamma,score=b.first_stage(*outer,fixed_lambda=chosen["lambda"])
            target=p2.transform(test.A.to_numpy(float));prediction=gamma.T@p1.transform(train.A.to_numpy(float))
            baseline=p2.transform(test.A.to_numpy(float)-train.A.mean())
            row[label]={"lambda":chosen["lambda"],"embedding_loss":score["embedding_loss"],
                        "mean_prediction_sse":float(np.sum((target-prediction)**2)),"mean_prediction_sst":float(baseline@baseline)}
        row["embedding_loss_gain"]=row["without_Z"]["embedding_loss"]-row["with_Z"]["embedding_loss"]
        rows.append(row)
    result={"validation_target":kind,"fixed_effect_absorption":False,"folds":rows,
            "note":"Same target kernel and held-out cells in the paired instrument comparison; lambda tuned inside the training data. No outcome Y enters this loss. Cell interpolation has within-commune dependence across folds; unseen-commune validation cannot learn category-specific components for the held-out communes."}
    for label in ["with_Z","without_Z"]:
        result[label]={"embedding_loss":float(np.average([r[label]["embedding_loss"] for r in rows],weights=[r["test_n"] for r in rows])),
                       "oof_mean_R2":float(1-sum(r[label]["mean_prediction_sse"] for r in rows)/sum(r[label]["mean_prediction_sst"] for r in rows))}
    result["embedding_loss_gain"]=result["without_Z"]["embedding_loss"]-result["with_Z"]["embedding_loss"]
    return result


def ensemble(d,repeats,seed,out,label,weighting="equal",structural_bandwidth=1.,instrument_bandwidth=1.,categorical_bandwidth=1.):
    s,_=inputs(d);keep=d.trim_keep.to_numpy(bool)
    weights=np.ones(len(d)) if weighting=="equal" else d.respondents.to_numpy(float)
    models=[];records=[];all_values=[];all_derivatives=[];split_means=[]
    for repeat in range(repeats):
        a,c=split_cells(d,seed+repeat);pair=[]
        for direction,(d1,d2) in enumerate([(a,c),(c,a)]):
            model,meta=fit(d1,d2,weighting,structural_bandwidth,instrument_bandwidth,categorical_bandwidth)
            v,derivative=evaluate(model,s);all_values.append(v);all_derivatives.append(derivative);pair.append(derivative)
            meta.update({"split":repeat,"direction":direction});records.append(meta);models.append(model)
        split_means.append(float(np.average(np.mean(pair,axis=0)[keep],weights=weights[keep])))
        print(f"  {label} split {repeat+1}/{repeats}: fitted derivative {split_means[-1]:.4f}",flush=True)
    value=np.mean(all_values,axis=0);derivative=np.mean(all_derivatives,axis=0)
    payload={f"model_{j}_{key}":v for j,model in enumerate(models) for key,v in model.items()}
    np.savez_compressed(out/f"models_{label}.npz",**payload)
    cells=d.copy();cells["f_component"]=value;cells["structural_fitted"]=value
    cells["fitted_derivative_not_identified_causal_effect"]=derivative
    cells.to_csv(out/f"cells_{label}.csv",index=False)
    result={"identification_status":"not_identified_with_supplied_share_instruments","weighting":weighting,
            "fixed_effect_absorption":False,"instrument_mode":d.attrs.get("instrument_mode","bartik"),
            "fitting_split":"balanced within commune; dependent observations across stages",
            "structural_bandwidth_multiplier":structural_bandwidth,"instrument_bandwidth_multiplier":instrument_bandwidth,
            "categorical_bandwidth":categorical_bandwidth,
            "average_fitted_derivative_trimmed":float(np.average(derivative[keep],weights=weights[keep])),
            "average_fitted_derivative_all_cells":float(np.average(derivative,weights=weights)),
            "split_average_fitted_derivatives":split_means,"split_range_not_CI":[min(split_means),max(split_means)],"fits":records}
    return result,cells


def figures(d,outcome,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c=d[d.trim_keep].copy();value="fitted_derivative_not_identified_causal_effect"
    c["age_group"]=pd.qcut(c.age,3,labels=["Lower mean age","Middle mean age","Higher mean age"])
    c["male_group"]=pd.qcut(c.male_share,3,labels=["Lower male share","Middle male share","Higher male share"])
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True);rows=[]
    for ax,group,title in zip(axes,["year","age_group","male_group"],["By year","By mean age","By male share"]):
        tab=c.groupby(group,observed=True)[value].agg(["mean","size"])
        ax.plot(np.arange(len(tab)),tab["mean"],marker="o",color="#185b75")
        ax.set_xticks(np.arange(len(tab)),[str(i) for i in tab.index],rotation=45,ha="right")
        ax.axhline(0,color="grey",lw=.8);ax.set_title(title)
        for k,r in tab.iterrows():rows.append({"grouping":group,"group":str(k),"average_fitted_derivative":r['mean'],"cells":int(r['size'])})
    axes[0].set_ylabel("Fitted derivative\n(points / log immigration)")
    fig.suptitle(b.OUTCOMES[outcome]+" — indicators inside kernel\nRegularized fit; derivatives are not identified causal effects",fontsize=12)
    fig.savefig(out/"heterogeneity.png",dpi=150);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/"heterogeneity.csv",index=False)


def self_checks():
    checks=b.self_checks();rng=np.random.default_rng(631)
    s=np.column_stack([rng.normal(size=(30,3)),rng.integers(0,5,30),rng.integers(2008,2018,30)])
    h=widths(s)
    onehot=np.column_stack([pd.get_dummies(s[:,-2],dtype=float),pd.get_dummies(s[:,-1],dtype=float)])
    expanded=np.column_stack([s[:,:3]/h[:3],onehot])
    explicit=np.exp(-.5*cdist(expanded,expanded,"sqeuclidean"))
    assert np.allclose(mixed_kernel(s,s,h),explicit,atol=1e-12)
    renamed=s.copy();renamed[:,-2]=renamed[:,-2]*173+5000;renamed[:,-1]=2025-(renamed[:,-1]-2008)
    assert np.allclose(mixed_kernel(s,s,h),mixed_kernel(renamed,renamed,h))
    model={"centers":s,"widths":h,"coef":rng.normal(size=len(s))};eps=1e-5
    sp=s.copy();sm=s.copy();sp[:,0]+=eps;sm[:,0]-=eps
    assert np.allclose(evaluate(model,s)[1],(evaluate(model,sp)[0]-evaluate(model,sm)[0])/(2*eps),atol=1e-7)
    d=pd.DataFrame({"cod_com":np.repeat(np.arange(10),10),"year":np.tile(np.arange(2008,2018),10),"respondents":100})
    a,c=split_cells(d,100)
    assert set(a.cod_com)==set(c.cod_com)==set(d.cod_com)
    assert not set(a.index)&set(c.index)
    x=rng.normal(size=(len(d),5));assert np.array_equal(WeightOnly(d).transform(x),x)
    checks.update({"indicator_kernel_equals_explicit_one_hot_RBF":True,"category_labels_have_no_numeric_order":True,
                   "mixed_kernel_derivatives_match_finite_differences":True,"both_stages_contain_all_communes":True,
                   "KIV_has_no_fixed_effect_absorption":True})
    return checks


def simulation_check(out):
    rng=np.random.default_rng(5102);G,T=60,10;n=G*T
    ids=np.repeat(np.arange(G),T);years=np.tile(np.arange(2008,2018),G);ts=(years-2012.5)/3
    age=rng.normal(size=n);male=rng.normal(size=n);z=rng.normal(size=n);u=rng.normal(size=n)
    a=1.5*z+.3*u+.2*rng.normal(size=n)+.1*age
    f=4*a+.8*a*age+.4*a*ts+.3*male**2
    y=f+2*u+np.repeat(rng.normal(size=G),T)+.5*ts
    d=pd.DataFrame({"cod_com":ids,"year":years,"age":age,"male_share":male,"A":a,"Z":z,"y":y,"respondents":100})
    d["trim_keep"]=d.groupby("year").A.transform(lambda s:s.between(s.quantile(.05),s.quantile(.95)))
    left,right=split_cells(d,79);derivatives=[]
    for d1,d2 in [(left,right),(right,left)]:
        model,_=fit(d1,d2);derivatives.append(evaluate(model,inputs(d)[0])[1])
    truth=4+.8*age+.4*ts;estimate=np.mean(derivatives,axis=0);keep=d.trim_keep.to_numpy(bool)
    result={"true_average_derivative":float(truth[keep].mean()),"estimated_average_derivative":float(estimate[keep].mean()),
            "derivative_rmse":float(np.sqrt(np.mean((estimate[keep]-truth[keep])**2))),
            "derivative_correlation":float(np.corrcoef(estimate[keep],truth[keep])[0,1]),
            "note":"Numerical/estimation diagnostic only. This simulation has a stochastic instrument varying within communes, unlike the deterministic baseline shares in the application. It does not establish identification or inference for the application."}
    b.save_json(out/"simulation_check.json",result)
    print(f"Simulation derivative {result['estimated_average_derivative']:.3f} vs {result['true_average_derivative']:.3f}",flush=True)


def report(results,out):
    lines=["KERNEL NPIV WITH COMMUNE AND YEAR INDICATORS INSIDE f","",
      "Specification: Ybar_mt = f(A_mt, agebar_mt, male_share_mt, D_commune, D_year) + u_mt.",
      "No separate alpha_m or tau_t terms. No commune/year demeaning or feature absorption in KIV.",
      "Commune/year indicators appear in both the structural kernel and first-stage conditioning kernel.",
      "Numeric year has been replaced by year indicators; there is no imposed ordering of years in the categorical kernel.",
      "", "IDENTIFICATION",IDENTIFICATION,
      "The extra controls change the identifying model. A fit can be calculated, but regularization determines how variation is assigned to immigration versus commune/year-dependent functions.",
      "A positive held-out improvement from redundant shares is a finite-sample feature/regularization effect, not new conditional IV information.",
      "", "KERNEL", "Gaussian product kernel on continuous inputs, multiplied by commune and year categorical kernels.",
      "Each categorical factor is exp(-1[labels differ]/h_category^2), exactly the Gaussian kernel on full one-hot indicators with a common bandwidth within each indicator block.",
      "Baseline categorical bandwidths: 1. Continuous widths: training-sample median pairwise distances, with positive-distance fallback for ties.",
      "Categorical labels are compared only for equality; commune codes are never treated as numeric distances.",
      "Both stages and all indicator-dependent terms are ridge-regularized; there are no unpenalized group intercepts.",
      "", "FIT AND VALIDATION", "Point estimates average reciprocal within-commune half-sample fits, so all commune categories occur in both stages.",
      "These stage samples are dependent within commune. The original independent-sample KIV guarantees do not automatically apply.",
      "Lambda is selected by held-out feature error; xi follows the original paper's structural-prediction validation criterion.",
      "Outer first-stage checks are reported separately for unseen communes and cell interpolation (two held-out years per observed commune).",
      "Unseen-commune checks cannot learn that commune's category-specific terms. Cell-interpolation checks are not independent-cluster validation.",
      "The same target kernel, cells and training-only bandwidth choices are used in each with/without-instrument comparison.",
      "L1 values are NOT directly comparable to the prior absorbed-effects run: both the target RKHS and the treatment of fixed effects have changed.",
      "", "REFERENCE AND INFERENCE", "The linear 2SLS reference retains additive commune/year effects and 11 share-times-shock instruments; it is a different, restricted model.",
      "Its partial R2/F are reference-model diagnostics, not evidence of conditional IV relevance after unrestricted commune/year conditioning.",
      "Fitted derivative split ranges are not confidence intervals. No causal confidence intervals or significance tests are supplied.",
      "Previous raw-share/additive-effects results: absorbed_effects/. Original aggregate-Bartik results: bartik/.", ""]
    for name,r in results.items():
        p=r['primary'];lin=r['linear_additive_reference']
        lines += [b.OUTCOMES[name]+f" ({name})",f"  Cells: {r['data']['cells']}; eligible respondents: {r['data']['eligible_respondents']:,}",
                  f"  Average fitted derivative (not identified causal effect): {p['average_fitted_derivative_trimmed']:.6f}",
                  f"  Split range (not CI): {p['split_range_not_CI']}",
                  f"  Additive linear reference: {lin['estimate']:.4f}, cluster SE {lin['cluster_se']:.4f}"]
        for kind,d in r['first_stage_diagnostics'].items():
            lines.append(f"  {kind}: L1 with shares {d['with_Z']['embedding_loss']:.6f}, without shares {d['without_Z']['embedding_loss']:.6f}, improvement {d['embedding_loss_gain']:.6f}")
        for label,s in r['sensitivity'].items():lines.append(f"  Sensitivity {label}: average fitted derivative {s['average_fitted_derivative_trimmed']:.6f}")
        lines.append("")
    lines += ["Run: python3 project/npiv/kernel/run_kernel.py", ""]
    (out/"report.txt").write_text("\n".join(lines))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--outcomes",nargs="+",default=["pc_19","pc_20","vict_agreg"],choices=list(b.OUTCOMES))
    ap.add_argument("--repeats",type=int,default=5);ap.add_argument("--sensitivity-repeats",type=int,default=3)
    ap.add_argument("--seed",type=int,default=20261125);ap.add_argument("--output",type=Path,default=HERE)
    ap.add_argument("--check-only",action="store_true");ap.add_argument("--simulation-only",action="store_true")
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);start=time.time()
    with threadpool_limits(limits=2):
        b.save_json(out/"self_checks.json",self_checks());print("Numerical checks passed",flush=True)
        if args.check_only:return
        simulation_check(out)
        if args.simulation_only:return
        results={};rows=[]
        for outcome in args.outcomes:
            print(outcome,flush=True);od=out/outcome;od.mkdir(exist_ok=True)
            d,metadata=b.load_cells(outcome,"raw_shares")
            metadata.update({"commune_and_year_indicators_inside_f":True,"numeric_year_in_kernel":False,
                             "fixed_effect_absorption":False,"identification_status":"not_identified_with_supplied_share_instruments"})
            primary,cells=ensemble(d,args.repeats,args.seed,od,"primary")
            diagnostics={kind:outer_diagnostics(d,args.seed+10000,kind) for kind in ["unseen_communes","cell_interpolation"]}
            sensitivity={}
            for label,cat,iv in [("categorical_bandwidth_half",.5,1.),("categorical_bandwidth_double",2.,1.),("share_bandwidth_double",1.,2.)]:
                sensitivity[label],_=ensemble(d,args.sensitivity_repeats,args.seed,od,label,categorical_bandwidth=cat,instrument_bandwidth=iv)
            results[outcome]={"data":metadata,"primary":primary,"first_stage_diagnostics":diagnostics,
                              "linear_additive_reference":b.linear_benchmark(d),"sensitivity":sensitivity}
            figures(cells,outcome,od)
            rows.append({"outcome":outcome,"identification_status":"not_identified_causal_effect",
                         "average_fitted_derivative":primary['average_fitted_derivative_trimmed'],
                         "split_min_not_CI":primary['split_range_not_CI'][0],"split_max_not_CI":primary['split_range_not_CI'][1],
                         "unseen_communes_L1_with":diagnostics['unseen_communes']['with_Z']['embedding_loss'],
                         "unseen_communes_L1_without":diagnostics['unseen_communes']['without_Z']['embedding_loss'],
                         "unseen_communes_L1_improvement":diagnostics['unseen_communes']['embedding_loss_gain'],
                         "cell_interpolation_L1_with":diagnostics['cell_interpolation']['with_Z']['embedding_loss'],
                         "cell_interpolation_L1_without":diagnostics['cell_interpolation']['without_Z']['embedding_loss'],
                         "cell_interpolation_L1_improvement":diagnostics['cell_interpolation']['embedding_loss_gain']})
            b.save_json(out/"results.json",{"specification":"commune_year_indicators_inside_kernel_no_additive_FE",
                         "identification_note":IDENTIFICATION,"seed":args.seed,"repeats":args.repeats,
                         "sensitivity_repeats":args.sensitivity_repeats,"elapsed_seconds":time.time()-start,
                         "source_sha256":hashlib.sha256(b.DATA.read_bytes()).hexdigest(),"outcomes":results})
            pd.DataFrame(rows).to_csv(out/"summary.csv",index=False);report(results,out)
        print(f"Saved results to {out}; {time.time()-start:.1f} seconds",flush=True)


if __name__=="__main__":main()
