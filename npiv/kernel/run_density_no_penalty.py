"""Conditional Gaussian KDE + unpenalized integrated-loss Gaussian-basis NPIV.

Panel specification and raw-share instruments match run_kernel.py. This is the
density/integrated-loss estimator in NPIV.tex, NOT Singh et al.'s embedding KIV.
Both stages have zero explicit ridge penalties. Bandwidths are selected on
commune-held-out losses. Gaussian integration is analytic, without simulation.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ecma31350-kernel-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("panel_kernel_helpers", HERE / "run_kernel.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

# Multipliers of training-only coordinate median distances; density A widths
# instead multiply the training standard deviation of A.
SHARE_GRID = [.5, 1., 2., 4.]
CONTROL_GRID = [.5, 1., 2., 4., 8.]
TIME_GRID = [.25, .5, 1., 2.]
DENSITY_A_GRID = [.15, .3, .6, 1.2]
STRUCTURAL_GRID = [.1, .15, .2, .3, .6, 1., 2.]
RIDGE = 0.0
SPECIFICATION = dict(b.SPECIFICATION,
    time_kernel="categorical year indicators; bandwidth selected by held-out integrated loss",
    feature_absorption="commune and year main effects projected from integrated second-stage features and outcomes; raw A never demeaned",
    first_stage="conditional Gaussian density mixture with nonnegative normalized kernel weights",
    structural_class="signed linear combinations of Gaussian bumps at observed (A, age, male share, year)",
    coefficient_penalty=RIDGE)


def folds(d, seed, n_folds=5):
    groups = np.random.default_rng(seed).permutation(np.sort(d.cod_com.unique()))
    return [d.cod_com.isin(ids).to_numpy() for ids in np.array_split(groups, n_folds)]


def density_distances(train, target, excluded=True):
    _, v = b.inputs(train)
    _, vt = b.inputs(target)
    h = b.widths(v)
    nz = v.shape[1] - 3
    dz = cdist(vt[:,:nz] / h[:nz], v[:,:nz] / h[:nz], "sqeuclidean") if excluded else np.zeros((len(target),len(train)))
    dx = cdist(vt[:,nz:-1] / h[nz:-1], v[:,nz:-1] / h[nz:-1], "sqeuclidean")
    dt = (vt[:,-1,None] != v[None,:,-1]).astype(float)
    sa = max(float(train.A.std(ddof=0)), 1e-10)
    da = ((target.A.to_numpy(float)[:,None] - train.A.to_numpy(float)[None,:]) / sa)**2
    return dz, dx, dt, da, h, sa


def log_weights(distances, params):
    dz, dx, dt = distances[:3]
    hz, hx, ht, _ = params
    lw = -.5 * dx / hx**2 - dt / ht**2
    if hz is not None:
        lw = lw - .5 * dz / hz**2
    return lw - logsumexp(lw, axis=1, keepdims=True)


def choose_density_bandwidth(d, split, excluded=True):
    zgrid = SHARE_GRID if excluded else [None]
    candidates = list(itertools.product(zgrid, CONTROL_GRID, TIME_GRID, DENSITY_A_GRID))
    losses = np.zeros(len(candidates))
    for held in split:
        train, target = d.loc[~held], d.loc[held]
        distances = density_distances(train, target, excluded)
        cached = {}
        for j, params in enumerate(candidates):
            hz, hx, ht, ha = params
            key = params[:3]
            if key not in cached:
                cached[key] = log_weights(distances, params)
            lp = logsumexp(cached[key] - .5 * distances[3] / ha**2, axis=1)
            lp -= np.log(ha * distances[5] * np.sqrt(2*np.pi))
            losses[j] -= lp.sum()
    losses /= len(d)
    selected = int(np.argmin(losses))
    records = [{"share_multiplier":p[0],"control_multiplier":p[1],"time_bandwidth":p[2],
                "density_A_sd_multiplier":p[3],"heldout_density_negative_log_likelihood":float(loss)}
               for p,loss in zip(candidates,losses)]
    return candidates[selected], {"selected":records[selected],"grid":records,
        "selection_loss":"held-out conditional-density negative log likelihood; folds grouped by commune",
        "excluded_shares":excluded,"ridge_penalty":RIDGE}


def density_model(train, target, params, excluded=True):
    distances = density_distances(train, target, excluded)
    lw = log_weights(distances, params)
    ha = params[3] * distances[5]
    w = np.exp(lw)
    assert np.allclose(w.sum(axis=1),1.,atol=1e-12)
    log_density = logsumexp(lw - .5 * distances[3] / params[3]**2,axis=1) - np.log(ha*np.sqrt(2*np.pi))
    return {"target":b.inputs(target)[0],"donor_A":train.A.to_numpy(float),"weights":w,
            "density_A_bandwidth":ha,"conditioning_base_widths":distances[4],
            "mean_prediction":w @ train.A.to_numpy(float),"log_density":log_density}


def crossfit_density(d, split, params):
    return [(np.flatnonzero(held),density_model(d.loc[~held],d.loc[held],params)) for held in split]


def structural_widths(d, candidate):
    h = b.widths(b.inputs(d)[0])
    h[:-1] *= candidate[0]
    h[-1] = candidate[1]
    return h


def integrated_features(density, centers, h):
    """Exact expectation of Gaussian treatment bumps under a Gaussian mixture."""
    variance = h[0]**2 + density['density_A_bandwidth']**2
    treatment = h[0]/np.sqrt(variance) * np.exp(-.5*(density['donor_A'][:,None]-centers[None,:,0])**2/variance)
    target = density['target']
    controls = np.exp(-.5*cdist(target[:,1:-1]/h[1:-1],centers[:,1:-1]/h[1:-1],"sqeuclidean"))
    years = np.exp(-(target[:,-1,None]!=centers[None,:,-1]).astype(float)/h[-1]**2)
    return (density['weights'] @ treatment) * controls * years


def crossfit_features(density_parts, centers, h, n):
    matrix = np.empty((n,len(centers)))
    for idx, model in density_parts:
        matrix[idx] = integrated_features(model,centers,h)
    return matrix


def least_squares(matrix, y, numerical_multiplier=1.):
    """No ridge; Moore-Penrose minimum-norm solution at numerical rank."""
    rcond = numerical_multiplier * np.finfo(float).eps * max(matrix.shape)
    coef, _, rank, singular = linalg.lstsq(matrix,y,cond=rcond,lapack_driver="gelsd",check_finite=False)
    if not np.isfinite(coef).all():
        raise FloatingPointError("Nonfinite unpenalized least-squares coefficients")
    error = y-matrix@coef
    scale = np.linalg.norm(matrix,ord='fro')*np.linalg.norm(error)+1e-30
    diag = {"ridge_penalty":RIDGE,"solver":"SVD minimum-Euclidean-norm least squares, LAPACK gelsd",
            "relative_numerical_rank_cutoff":rcond,"absolute_singular_value_cutoff":float(rcond*singular[0]),
            "numerical_rank":int(rank),"columns":matrix.shape[1],"rows":matrix.shape[0],
            "coefficient_norm":float(np.linalg.norm(coef)),"max_absolute_coefficient":float(np.max(np.abs(coef))),
            "retained_condition_number":float(singular[0]/singular[rank-1]) if rank else None,
            "projected_training_integrated_MSE":float(np.mean(error**2)),
            "relative_normal_equation_error":float(np.linalg.norm(matrix.T@error)/scale),
            "singular_values":singular.tolist(),
            "rank_note":"Cutoff is machine precision times matrix dimension, not a tuned statistical penalty. Rank deficiency selects a minimum-norm solution."}
    return coef, diag


def fit_at_width(d, density_parts, candidate):
    centers = b.inputs(d)[0]
    h = structural_widths(d,candidate)
    integrated = crossfit_features(density_parts,centers,h,len(d))
    p = b.Absorb(d)
    matrix, y = p.transform(integrated),p.transform(d.y.to_numpy(float))
    coef, diag = least_squares(matrix,y)
    return {"centers":centers,"widths":h,"coef":coef},diag,integrated


def tune_pipeline(d, seed):
    candidates = list(itertools.product(STRUCTURAL_GRID,TIME_GRID))
    sum_loss = np.zeros(len(candidates))
    records = []
    first_stage = []
    for fold, held in enumerate(folds(d,seed)):
        train,target = d.loc[~held].copy(),d.loc[held].copy()
        inner = folds(train,seed+100+fold,3)
        params,tuning = choose_density_bandwidth(train,inner)
        baseline,baseline_tuning = choose_density_bandwidth(train,inner,False)
        train_density = crossfit_density(train,inner,params)
        test_density = density_model(train,target,params)
        test_baseline = density_model(train,target,baseline,False)
        fs = {"fold":fold,"test_communes":sorted(target.cod_com.unique().tolist()),"test_n":len(target),
              "with_shares":{"density_NLL":float(-test_density['log_density'].mean()),
                             "mean_prediction_SSE":float(np.sum((target.A.to_numpy()-test_density['mean_prediction'])**2)),
                             "bandwidth_selection":tuning},
              "without_shares":{"density_NLL":float(-test_baseline['log_density'].mean()),
                                "mean_prediction_SSE":float(np.sum((target.A.to_numpy()-test_baseline['mean_prediction'])**2)),
                                "bandwidth_selection":baseline_tuning},
              "mean_baseline_SSE":float(np.sum((target.A.to_numpy()-train.A.mean())**2))}
        first_stage.append(fs)
        pt = b.Absorb(target)
        row = {"fold":fold,"train_communes":sorted(train.cod_com.unique().tolist()),
               "test_communes":fs['test_communes'],"test_n":len(target),"candidates":[]}
        for j,candidate in enumerate(candidates):
            model,diag,_ = fit_at_width(train,train_density,candidate)
            prediction = integrated_features(test_density,model['centers'],model['widths']) @ model['coef']
            residual = pt.transform(target.y.to_numpy(float)-prediction)
            loss = float(residual@residual)
            if not np.isfinite(loss):
                raise FloatingPointError("Nonfinite validation loss")
            sum_loss[j] += loss
            row['candidates'].append({"continuous_multiplier":candidate[0],"time_bandwidth":candidate[1],
                                      "heldout_projected_integrated_MSE":loss/len(target),
                                      "training_numerical_rank":diag['numerical_rank'],
                                      "coefficient_norm":diag['coefficient_norm']})
        records.append(row)
        print(f"  bandwidth CV fold {fold+1}/5 complete",flush=True)
    loss = sum_loss/len(d)
    chosen = int(np.argmin(loss))
    grid = [{"continuous_multiplier":c[0],"time_bandwidth":c[1],"heldout_projected_integrated_MSE":float(v)} for c,v in zip(candidates,loss)]
    diagnostics = {"folds":first_stage,"nested_density_tuning":True,
                   "note":"Held-out density NLL and mean-prediction error; not the KIV embedding loss L1."}
    for name in ['with_shares','without_shares']:
        diagnostics[name] = {"density_NLL":float(np.average([r[name]['density_NLL'] for r in first_stage],weights=[r['test_n'] for r in first_stage])),
                             "mean_prediction_R2":float(1-sum(r[name]['mean_prediction_SSE'] for r in first_stage)/sum(r['mean_baseline_SSE'] for r in first_stage))}
    diagnostics['density_loss_improvement'] = diagnostics['without_shares']['density_NLL']-diagnostics['with_shares']['density_NLL']
    tuning = {"selected":grid[chosen],"grid":grid,"folds":records,"ridge_penalty":RIDGE,
              "continuous_grid_boundary":bool(candidates[chosen][0] in [STRUCTURAL_GRID[0],STRUCTURAL_GRID[-1]]),
              "time_grid_boundary":bool(candidates[chosen][1] in [TIME_GRID[0],TIME_GRID[-1]]),
              "selection_loss":"commune-held-out integrated NPIV squared loss after absorbing additive effects",
              "caveat":"The minimum CV loss selects bandwidths; it is not an independent assessment of the selected structural model.",
              "heldout_effects":"Outcome and integrated features are projected on held-out commune/year indicators for within-panel validation, not outcome-level forecasting."}
    return candidates[chosen],tuning,diagnostics


def save_estimate(d,model,integrated,out):
    g,derivative = b.evaluate(model,b.inputs(d)[0])
    p = b.Absorb(d)
    nuisance = linalg.lstsq(p.D,d.y.to_numpy(float)-g)[0]
    cells = d.copy()
    cells['g_component'] = g
    cells['commune_fixed_effect_component'] = p.D[:,:p.n_communes] @ nuisance[:p.n_communes]
    cells['year_main_effect_inside_f'] = p.D[:,p.n_communes:] @ nuisance[p.n_communes:]
    cells['f_component'] = g + cells.year_main_effect_inside_f
    cells['structural_fitted'] = cells.f_component + cells.commune_fixed_effect_component
    cells['marginal_effect'] = derivative
    cells['integrated_moment_fitted'] = d.y.to_numpy(float)-p.project(d.y.to_numpy(float)) + p.project(integrated @ model['coef'])
    cells.to_csv(out/'cells_primary.csv',index=False)
    np.savez_compressed(out/'model_primary.npz',**model)
    if not np.isfinite(cells.select_dtypes(include=[np.number]).to_numpy()).all():
        raise FloatingPointError("Nonfinite saved estimate")
    # Reuse the overlap rule/figures. Duplicate one fit only to satisfy the
    # helper's paired-fit interface; remove its artificial range fields below.
    time_result = b.time_heterogeneity(d,[model,model],out,'primary')
    frame = pd.read_csv(out/'time_heterogeneity_primary.csv')
    drop = ['common_profile_split_min_not_CI','common_profile_split_max_not_CI']
    frame.drop(columns=drop).to_csv(out/'time_heterogeneity_primary.csv',index=False)
    for row in time_result.get('estimates',[]):
        for key in drop: row.pop(key,None)
    b.figures(cells,out.name,out)
    keep = cells.trim_keep.to_numpy(bool)
    return {"average_derivative_trimmed":float(derivative[keep].mean()),
            "average_derivative_all_cells":float(derivative.mean()),
            "derivative_percentiles":dict(zip(['min','p05','median','p95','max'],np.percentile(derivative,[0,5,50,95,100]).tolist())),
            "time_heterogeneity":time_result},cells


def self_checks():
    rng = np.random.default_rng(671)
    centers = np.column_stack([rng.normal(size=(8,3)),rng.integers(2008,2011,8)])
    target = np.column_stack([rng.normal(size=(4,3)),rng.integers(2008,2011,4)])
    w = rng.random((4,6));w/=w.sum(axis=1,keepdims=True)
    density = {'target':target,'donor_A':rng.normal(size=6),'weights':w,'density_A_bandwidth':.25}
    h = np.array([.7,1.1,.8,1.])
    exact = integrated_features(density,centers,h)
    nodes,weights = np.polynomial.hermite.hermgauss(60)
    numerical = np.zeros_like(exact)
    for i in range(4):
        for j in range(6):
            points = np.tile(target[i],(len(nodes),1));points[:,0] = density['donor_A'][j]+np.sqrt(2)*.25*nodes
            numerical[i] += w[i,j]*(weights @ b.kernel(points,centers,h))/np.sqrt(np.pi)
    assert np.allclose(exact,numerical,atol=1e-11)
    matrix = rng.normal(size=(40,8));y = rng.normal(size=40)
    coef,diag = least_squares(matrix,y)
    assert diag['ridge_penalty']==0 and diag['numerical_rank']==8
    assert np.allclose(coef,np.linalg.lstsq(matrix,y,rcond=None)[0],atol=1e-12)
    panel = pd.DataFrame({'cod_com':np.repeat(np.arange(8),5),'year':np.tile(np.arange(5),8),'respondents':100})
    p = b.Absorb(panel)
    coef,_ = least_squares(p.project(matrix),p.project(y))
    direct = np.linalg.lstsq(np.column_stack([p.D,matrix]),y,rcond=None)[0][-8:]
    assert np.allclose(coef,direct,atol=1e-10)
    model = {'centers':centers,'widths':h,'coef':rng.normal(size=8)}
    plus,minus = target.copy(),target.copy();plus[:,0]+=1e-5;minus[:,0]-=1e-5
    assert np.allclose(b.evaluate(model,target)[1],(b.evaluate(model,plus)[0]-b.evaluate(model,minus)[0])/2e-5,atol=1e-7)
    assert all(not set(panel.loc[s].cod_com)&set(panel.loc[~s].cod_com) for s in folds(panel,65))
    return {'analytic_density_integration_matches_Gauss_Hermite':True,
            'zero_penalty_matches_ordinary_least_squares':True,'feature_absorption_matches_joint_unpenalized_effects':True,
            'derivative_matches_finite_difference':True,'commune_folds_disjoint':True}


def report(results,out):
    lines = ['DENSITY-BASED PANEL NPIV WITHOUT RIDGE PENALTIES','',
        'Specification: Y_mt = alpha_m + f(A_mt, age_mt, male_share_mt, year_t) + u_mt; f=b_t+g.',
        'Additive commune intercepts and year main effects are unpenalized. No commune identity interactions in g.',
        'First stage: normalized Gaussian conditional density mixture on 11 raw shares, demographics and categorical year.',
        'First-stage bandwidths minimize held-out density negative log likelihood; there is no first-stage ridge penalty.',
        'Structural class: signed linear combinations of Gaussian bumps at observed immigration/demographic/year profiles.',
        'The structural response is not constrained to be a density; its coefficients can be positive or negative.',
        'Second-stage bandwidths minimize held-out integrated NPIV squared loss. Explicit coefficient ridge is exactly ZERO.',
        'Bandwidth minimization is over the saved candidate grids; it is not a claim of a global continuous optimum.',
        'The grid was extended after the initial fits selected its smallest structural bandwidth; initial_grid_results.json preserves that first run.',
        'Both bandwidth searches use commune folds; outer structural-validation groups are excluded from density tuning and fitting.',
        'Gaussian integration is analytic, not simulated. Fixed effects are absorbed after constructing the integrated features, never from raw A.',
        'A minimum-norm SVD solution is used at numerical rank (relative cutoff = machine epsilon * max(matrix dimensions)).',
        'This numerical convention is reported explicitly; it can matter when the unpenalized objective has nonunique or nearly singular solutions.',
        'Bandwidth, finite basis size and minimum-norm selection still restrict the fit; zero ridge does not mean unrestricted estimation.',
        'The first-stage diagnostics here are density log loss and mean-prediction R2, NOT KIV L1.',
        'The old KIV run uses a different first-stage estimator and tuning loss; differences from it cannot be attributed only to ridge removal.',
        'No causal confidence intervals, bootstrap standard errors or significance tests are reported.',
        'Year comparisons hold the same observed immigration/demographic profiles fixed across years, with empirical overlap checks.',
        'A numerical-rank-cutoff sensitivity is saved separately; it is not tuning or a confidence interval.','']
    for name,r in results.items():
        p=r['primary'];s=r['solver'];fs=r['first_stage_diagnostics']
        lines += [b.OUTCOMES[name]+f' ({name})',
                  f"  Cells: {r['data']['cells']}; communes: {r['data']['communes']}",
                  f"  Average derivative: {p['average_derivative_trimmed']:.8g}",
                  f"  Selected structural bandwidths: {r['structural_bandwidth_selection']['selected']}",
                  f"  Structural grid boundary selections: continuous={r['structural_bandwidth_selection']['continuous_grid_boundary']}, time={r['structural_bandwidth_selection']['time_grid_boundary']}",
                  f"  Selected density bandwidths: {r['density_bandwidth_selection']['selected']}",
                  f"  Training integrated MSE: {s['projected_training_integrated_MSE']:.8g}",
                  f"  Numerical rank: {s['numerical_rank']}/{s['columns']}; retained condition number: {s['retained_condition_number']:.6g}",
                  f"  Coefficient norm: {s['coefficient_norm']:.6g}; relative normal-equation error: {s['relative_normal_equation_error']:.6g}",
                  f"  Held-out density loss: shares {fs['with_shares']['density_NLL']:.6g}, no shares {fs['without_shares']['density_NLL']:.6g}; improvement {fs['density_loss_improvement']:.6g}",
                  f"  Held-out treatment mean R2: shares {fs['with_shares']['mean_prediction_R2']:.6g}, no shares {fs['without_shares']['mean_prediction_R2']:.6g}",
                  f"  Numerical-cutoff sensitivity: {r['numerical_cutoff_sensitivity']}",'']
    lines += ['Run: python3 project/npiv/kernel/run_density_no_penalty.py',
              'Per outcome: cells_primary.csv, model_primary.npz, structural_bandwidth_cv.csv, density_bandwidth_cv.csv, time_heterogeneity_primary.csv/.png.','']
    (out/'report.txt').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outcomes',nargs='+',default=['pc_19','pc_20','vict_agreg'],choices=list(b.OUTCOMES))
    parser.add_argument('--output',type=Path,default=HERE/'no_penalty')
    parser.add_argument('--seed',type=int,default=20261125)
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);start=time.time()
    with threadpool_limits(limits=2):
        b.save_json(out/'self_checks.json',self_checks());print('Numerical checks passed',flush=True)
        if args.check_only:return
        results={};summary=[]
        for outcome in args.outcomes:
            print(outcome,flush=True);od=out/outcome;od.mkdir(exist_ok=True)
            d,metadata=b.load_cells(outcome,'raw_shares')
            candidate,structural_tuning,diagnostics=tune_pipeline(d,args.seed)
            full_folds=folds(d,args.seed)
            params,density_tuning=choose_density_bandwidth(d,full_folds)
            density_parts=crossfit_density(d,full_folds,params)
            model,solver,integrated=fit_at_width(d,density_parts,candidate)
            primary,cells=save_estimate(d,model,integrated,od)
            p=b.Absorb(d);matrix=p.transform(integrated);y=p.transform(d.y.to_numpy(float))
            sensitivity=[]
            for multiplier in [.1,1.,10.]:
                coef,diag=least_squares(matrix,y,multiplier)
                derivative=b.evaluate(dict(model,coef=coef),b.inputs(d)[0])[1]
                sensitivity.append({'machine_cutoff_multiplier':multiplier,'relative_cutoff':diag['relative_numerical_rank_cutoff'],
                                    'numerical_rank':diag['numerical_rank'],'average_derivative':float(derivative[d.trim_keep].mean()),
                                    'training_integrated_MSE':diag['projected_training_integrated_MSE']})
            r={'data':metadata,'primary':primary,'solver':solver,'density_bandwidth_selection':density_tuning,
               'structural_bandwidth_selection':structural_tuning,'first_stage_diagnostics':diagnostics,
               'numerical_cutoff_sensitivity':sensitivity,'linear_reference':b.linear_benchmark(d)}
            results[outcome]=r
            pd.DataFrame(structural_tuning['grid']).to_csv(od/'structural_bandwidth_cv.csv',index=False)
            pd.DataFrame(density_tuning['grid']).to_csv(od/'density_bandwidth_cv.csv',index=False)
            summary.append({'outcome':outcome,'ridge_penalty':RIDGE,'average_derivative':primary['average_derivative_trimmed'],
                            'selected_heldout_integrated_MSE':structural_tuning['selected']['heldout_projected_integrated_MSE'],
                            'numerical_rank':solver['numerical_rank'],'coefficient_norm':solver['coefficient_norm'],
                            'density_loss_improvement':diagnostics['density_loss_improvement']})
            b.save_json(out/'results.json',{'method':'conditional_KDE_integrated_loss_Gaussian_basis_no_ridge','ridge_penalty':RIDGE,
                'specification':SPECIFICATION,'seed':args.seed,'elapsed_seconds':time.time()-start,
                'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'helper_sha256':hashlib.sha256((HERE/'run_kernel.py').read_bytes()).hexdigest(),
                'source_sha256':hashlib.sha256(b.DATA.read_bytes()).hexdigest(),'outcomes':results})
            pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);report(results,out)
            print(f"  average derivative {primary['average_derivative_trimmed']:.6g}; rank {solver['numerical_rank']}/{solver['columns']}",flush=True)
        print(f'Saved {out}; elapsed {time.time()-start:.1f} seconds',flush=True)


if __name__=='__main__':main()
