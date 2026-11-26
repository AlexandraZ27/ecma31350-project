"""Commune pairs bootstrap of the selected full neural panel estimator.

Retunes neither architecture nor stopping time. Refits every cross-fitted MDN
and all five outcome initializations. Resampling is stratified by the original
commune folds so duplicates of a source commune cannot cross fitting folds.
Reported inference is conditional/approximate, not weak-IV-robust inference.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from mixture_quadrature import mixture_gauss

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('full_neural',HERE/'run_full_neural.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
SOURCE=HERE/'full_neural'
BOOT_SEED=20261126
DATA={};RESULTS=None


def setup_worker():
    global RESULTS
    n.torch.set_num_threads(1)
    try:n.torch.set_num_interop_threads(1)
    except RuntimeError:pass
    RESULTS=json.loads((SOURCE/'results.json').read_text())
    for name in RESULTS['outcomes']:
        d=pd.read_csv(SOURCE/name/'cells_primary.csv')
        d.attrs['instrument_mode']='raw_shares'
        DATA[name]=d


def sample_communes(d,replicate,identity=False):
    """Sample complete histories, with distinct FE IDs for each sampled copy."""
    rng=np.random.default_rng(BOOT_SEED+replicate)
    rows=[];draws=[];clone=0
    for fold,held in enumerate(n.k.folds(d,RESULTS['seed'])):
        ids=np.sort(d.loc[held,'cod_com'].unique())
        sampled=ids if identity else rng.choice(ids,len(ids),replace=True)
        for source in sampled:
            block=d.loc[d.cod_com==source].copy()
            block['source_commune']=int(source);block['cod_com']=clone;block['bootstrap_fold']=fold
            rows.append(block);draws.append({'copy':clone,'source_commune':int(source),'fold':fold});clone+=1
    boot=pd.concat(rows,ignore_index=True);boot.attrs['instrument_mode']='raw_shares'
    boot['trim_keep']=boot.groupby('year').A.transform(lambda x:x.between(x.quantile(.05),x.quantile(.95)))
    assert len(boot)==1010 and boot.cod_com.nunique()==101
    assert boot.groupby('source_commune').bootstrap_fold.nunique().max()==1
    assert boot.groupby('cod_com').size().eq(10).all()
    return boot,draws


def fit_statistic(d,settings,profiles):
    selected=settings['density_selection']['selected']
    config=(tuple(selected['hidden']),selected['components'],selected['steps'])
    density={};exclusions=[]
    for fold in range(5):
        held=d.bootstrap_fold.to_numpy()==fold
        train,target=d.loc[~held],d.loc[held]
        assert set(train.source_commune).isdisjoint(target.source_commune)
        prediction,models=n.fit_density_ensemble(train,target,config,True)
        if not density:density={key:np.empty((len(d),value.shape[1])) for key,value in prediction.items()}
        for key,value in prediction.items():density[key][held]=value
        exclusions.append({'fold':fold,'train_distinct_source_communes':int(train.source_commune.nunique()),
                           'test_distinct_source_communes':int(target.source_commune.nunique())})
    nodes,weights=mixture_gauss(density,n.hermite_quadrature,32)
    integration_method='32-node Gaussian mixture rule, checked against Hermite64 per component'
    check_nodes,check_weights=n.hermite_quadrature(density,64)
    hidden=tuple(settings['network_selection']['selected_hidden']);steps=settings['network_selection']['selected_steps']
    s=n.b.inputs(d)[0];derivatives=[];time_derivatives=[];errors=[]
    years=np.sort(d.year.unique())
    reference=np.concatenate([np.column_stack([profiles[['A','age','male_share']].to_numpy(),np.full(len(profiles),year)]) for year in years])
    scale_y=max(float(np.std(n.b.Absorb(d).project(d.y.to_numpy()))),1e-10)
    for attempt in range(2):
        derivatives=[];time_derivatives=[];errors=[]
        for seed in n.h.FINAL_INITIALIZATIONS:
            model,_=n.train_outcome(d,nodes,weights,hidden,seed,steps)
            derivatives.append(n.h.evaluate(model,s)[1])
            time_derivatives.append(n.h.evaluate(model,reference)[1].reshape(len(years),len(profiles)).mean(axis=1))
            moment=n.integrate(model,s,nodes,weights);check=n.integrate(model,s,check_nodes,check_weights)
            errors.append(float(np.sqrt(np.mean((moment-check)**2))))
        if max(errors)<=1e-5*scale_y:break
        if attempt==1:raise ValueError(f'Bootstrap quadrature failed: {max(errors):g}')
        nodes,weights=n.quadrature(density,RESULTS['quadrature_start_nodes'])
        integration_method='adaptive Gauss-Legendre fallback, checked against Hermite64 per component'
    slope=np.mean(derivatives,axis=0);time_slope=np.mean(time_derivatives,axis=0)
    return {'average_derivative':float(slope[d.trim_keep].mean()),
            'average_derivative_all_cells':float(slope.mean()),
            'time_common_profile_derivative':time_slope.tolist(),'years':years.tolist(),
            'trimmed_cells':int(d.trim_keep.sum()),'quadrature_nodes_max':nodes.shape[1],
            'max_quadrature_RMSE':max(errors),'fold_exclusions':exclusions,
            'integration_method':integration_method}


def one_job(job):
    name,replicate,output=job;start=time.time();path=Path(output)/'draws'/f'{name}_{replicate:04}.json'
    with threadpool_limits(limits=1):
        d,draws=sample_communes(DATA[name],replicate)
        profiles=pd.read_csv(SOURCE/name/'common_time_profiles.csv')
        result=fit_statistic(d,RESULTS['outcomes'][name],profiles)
    record={'outcome':name,'replicate':replicate,'seed':BOOT_SEED+replicate,'commune_draws':draws,
            'result':result,'elapsed_seconds':time.time()-start}
    n.b.save_json(path,record)
    return name,replicate,result['average_derivative'],record['elapsed_seconds']


def summarize(out,B):
    summary=[];inference={};year_rows=[];all_rows=[]
    for name,settings in RESULTS['outcomes'].items():
        records=[json.loads((out/'draws'/f'{name}_{b:04}.json').read_text()) for b in range(B)]
        values=np.array([r['result']['average_derivative'] for r in records])
        point=settings['primary']['average_derivative_trimmed'];se=float(values.std(ddof=1));errors=values-point
        p_wald=float(2*norm.sf(abs(point/se))) if se>0 else float(point==0)
        p_centered=float((1+np.count_nonzero(np.abs(errors)>=abs(point)))/(B+1))
        lo,hi=np.quantile(errors,[.025,.975]);basic=[float(point-hi),float(point-lo)]
        critical=float(np.sort(np.abs(errors))[min(B-1,int(np.ceil(.95*(B+1)))-1)])
        row={'outcome':name,'point_estimate':point,'bootstrap_SE':se,'normal_Wald_p':p_wald,
             'centered_bootstrap_p':p_centered,'basic_CI_low':basic[0],'basic_CI_high':basic[1],
             'symmetric_CI_low':float(point-critical),'symmetric_CI_high':float(point+critical),
             'bootstrap_mean':float(values.mean()),'bootstrap_bias_diagnostic':float(values.mean()-point),
             'replicates':B,'SE_Monte_Carlo_relative_approx':float(1/np.sqrt(2*(B-1)))}
        summary.append(row);inference[name]=row
        times=np.array([r['result']['time_common_profile_derivative'] for r in records])
        original=settings['primary']['time_heterogeneity']['estimates']
        for j,v in enumerate(original):
            p=v['common_profile_average_derivative'];sd=float(times[:,j].std(ddof=1))
            low,high=np.quantile(times[:,j]-p,[.025,.975])
            errors_t=times[:,j]-p
            critical_t=float(np.sort(np.abs(errors_t))[min(B-1,int(np.ceil(.95*(B+1)))-1)])
            year_rows.append({'outcome':name,'year':v['year'],'point_estimate':p,'bootstrap_SE':sd,
                             'pointwise_basic_CI_low':float(p-high),'pointwise_basic_CI_high':float(p-low),
                             'pointwise_symmetric_CI_low':float(p-critical_t),'pointwise_symmetric_CI_high':float(p+critical_t),
                             'centered_bootstrap_p':float((1+np.count_nonzero(np.abs(errors_t)>=abs(p)))/(B+1)),
                             'normal_Wald_p':float(2*norm.sf(abs(p/sd)))})
        for rec in records:
            all_rows.append({'outcome':name,'replicate':rec['replicate'],
                             'estimate':rec['result']['average_derivative'],
                             'distinct_source_communes':len({x['source_commune'] for x in rec['commune_draws']}),
                             'trimmed_cells':rec['result']['trimmed_cells'],
                             'max_quadrature_RMSE':rec['result']['max_quadrature_RMSE']})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(year_rows).to_csv(out/'time_inference.csv',index=False)
    pd.DataFrame(all_rows).to_csv(out/'bootstrap_draws.csv',index=False)
    metadata={'method':'commune pairs bootstrap stratified by original cross-fitting folds',
        'replicates':B,'seed':BOOT_SEED,'cluster_count':101,'years_per_cluster':10,
        'refitted':'both stages: five cross-fitted MDN ensembles and five outcome initializations per draw',
        'held_fixed':'selected architectures, mixture count, stopping steps, optimizer settings, original fold strata, initialization seeds',
        'resampled':'whole commune histories, including outcomes, treatments, instruments and controls',
        'trimming':'recompute within-year 5th and 95th treatment percentiles in each resample',
        'time_reference_profiles':'hold original common-support profiles fixed for time comparisons',
        'integration':'32-node Gaussian mixture integration with a Hermite64 check and adaptive Legendre fallback; earlier checkpoint draws used the original adaptive Legendre rule',
        'p_value':'two-sided centered bootstrap tail probability, with plus-one finite simulation correction',
        'alternative_p_value':'two-sided normal Wald p using the bootstrap standard error',
        'reported_interval':'symmetric centered-bootstrap interval from the absolute-error order statistic matching the reported test; equal-tailed basic intervals retained as an alternative',
        'limitations':'conditional-on-selection approximation; no theorem for unrestricted neural NPIV derivatives, no correction for approximation/early-stopping bias or weak identification; assumes independent commune clusters conditional on common time shocks',
        'source_results_sha256':hashlib.sha256((SOURCE/'results.json').read_bytes()).hexdigest(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fit_runner_sha256':hashlib.sha256((HERE/'run_full_neural.py').read_bytes()).hexdigest(),
        'quadrature_helper_sha256':hashlib.sha256((HERE/'mixture_quadrature.py').read_bytes()).hexdigest(),
        'outcomes':inference}
    n.b.save_json(out/'results.json',metadata)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--replicates',type=int,default=399)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=Path,default=SOURCE/'inference')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();out=args.output;(out/'draws').mkdir(parents=True,exist_ok=True)
    setup_worker();start=time.time()
    check_path=out/'identity_checks_single_thread.json'
    if not check_path.exists():
        checks={}
        for name,settings in RESULTS['outcomes'].items():
            with threadpool_limits(limits=1):
                d,_=sample_communes(DATA[name],0,identity=True)
                result=fit_statistic(d,settings,pd.read_csv(SOURCE/name/'common_time_profiles.csv'))
            original=settings['primary']['average_derivative_trimmed']
            assert np.isclose(result['average_derivative'],original,atol=1e-6,rtol=1e-6)
            assert np.allclose(result['time_common_profile_derivative'],
                [r['common_profile_average_derivative'] for r in settings['primary']['time_heterogeneity']['estimates']],atol=1e-6,rtol=1e-6)
            checks[name]={'original':original,'identity_refit':result['average_derivative'],
                          'max_quadrature_RMSE':result['max_quadrature_RMSE']}
            print(f'{name}: identity refit matches original estimate',flush=True)
        n.b.save_json(check_path,checks)
    if args.check_only:return
    jobs=[(name,b,str(out)) for b in range(args.replicates) for name in RESULTS['outcomes']
          if not (out/'draws'/f'{name}_{b:04}.json').exists()]
    print(f'Running {len(jobs)} remaining outcome/replicate fits with {args.workers} workers',flush=True)
    pool=ProcessPoolExecutor(max_workers=args.workers,initializer=setup_worker)
    try:
        futures=[pool.submit(one_job,job) for job in jobs]
        for i,future in enumerate(as_completed(futures),1):
            name,replicate,estimate,seconds=future.result()
            if i%15==0 or i==len(jobs):
                print(f'{i}/{len(jobs)} fits complete; latest {name} draw {replicate}: {estimate:.5f}; elapsed {time.time()-start:.1f}s',flush=True)
    except BaseException:
        for future in futures:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    else:pool.shutdown(wait=True)
    summary=summarize(out,args.replicates)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print(f'Saved {out}; elapsed {time.time()-start:.1f}s',flush=True)


if __name__=='__main__':main()
