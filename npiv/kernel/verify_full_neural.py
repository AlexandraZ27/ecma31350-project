"""Independent reconstruction of both stages of the saved panel DeepIV run."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('full_neural',ROOT/'run_full_neural.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
OUT=ROOT/'full_neural'


def check_density_models(models,train,target,with_shares):
    predicted=[];floor_ratios=[]
    for model in models:
        expected=n.density_scaling(train,with_shares)
        for key in expected:assert np.allclose(model[key],expected[key],rtol=1e-11,atol=1e-11)
        assert set(model['train_communes'])==set(train.cod_com.unique())
        assert float(model['sigma_floor'])==n.SIGMA_FLOOR
        assert model['weight_0'].shape[1]==(23 if with_shares else 12)
        pred=n.density_predict(model,target)
        assert np.allclose(pred['pi'].sum(axis=1),1,atol=1e-12)
        assert np.isfinite(n.log_density(pred,target.A.to_numpy())).all()
        ratio=float(pred['sigma'].min()/model['A_sd']);floor_ratios.append(ratio)
        assert ratio>=n.SIGMA_FLOOR
        altered=target.copy();altered['A']+=100;altered['y']+=100
        other=n.density_predict(model,altered)
        for key in pred:assert np.allclose(pred[key],other[key],atol=1e-12)
        predicted.append(pred)
    return n.ensemble_density(predicted),min(floor_ratios)


def verify_selection(selection):
    chosen=selection['selected'];grid=selection['grid']
    assert np.isclose(chosen['heldout_density_NLL'],min(row['heldout_density_NLL'] for row in grid),rtol=1e-12)
    assert selection['weight_decay']==0 and selection['dropout']==0
    for row in grid:
        total=0.;count=0
        for fold in selection['folds']:
            assert set(fold['train_communes']).isdisjoint(fold['test_communes'])
            match=[v for v in fold['grid'] if all(v[key]==row[key] for key in ['hidden','components','steps'])]
            assert len(match)==1
            total+=match[0]['heldout_density_NLL']*fold['test_n'];count+=fold['test_n']
        assert np.isclose(total/count,row['heldout_density_NLL'],rtol=1e-12)


def main():
    r=json.loads((OUT/'results.json').read_text())
    hybrid=json.loads((ROOT/'neural_outcome/results.json').read_text())
    assert set(r['outcomes'])=={'pc_19','pc_20','vict_agreg'}
    assert r['runner_sha256']==hashlib.sha256((ROOT/'run_full_neural.py').read_bytes()).hexdigest()
    for file,sha in r['helper_sha256'].items():assert sha==hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    assert r['source_sha256']==hashlib.sha256(n.b.DATA.read_bytes()).hexdigest()
    assert r['reference_sha256']==hashlib.sha256((ROOT/'neural_outcome/results.json').read_bytes()).hexdigest()
    assert all(json.loads((OUT/'self_checks.json').read_text()).values())
    summary=pd.read_csv(OUT/'summary.csv').set_index('outcome')
    audit={'source_hashes_match':True,'all_numerical_self_checks_pass':True,'outcomes':{}}
    with threadpool_limits(limits=2):
        for name,v in r['outcomes'].items():
            od=OUT/name;d,meta=n.b.load_cells(name,'raw_shares')
            cells=pd.read_csv(od/'cells_primary.csv')
            assert len(d)==1010 and d.cod_com.nunique()==101 and d.year.nunique()==10
            assert np.allclose(cells[d.columns].select_dtypes(include=[np.number]),d.select_dtypes(include=[np.number]))
            assert np.isfinite(cells.select_dtypes(include=[np.number]).to_numpy()).all()
            cv=v['network_selection'];fs=v['first_stage_diagnostics'];oof=np.load(od/'cv_predictions.npz')
            selected_key=f"hidden_{'_'.join(map(str,cv['selected_hidden']))}_steps_{cv['selected_steps']}"
            assert np.isclose(cv['selected_heldout_integrated_MSE'],min(row['heldout_integrated_MSE'] for row in cv['grid']),rtol=1e-12)
            errors=[];density_floor_ratios=[];all_communes=[];scores={key:0. for key in oof.files};baseline_sse=0.
            first_scores={label:{'nll':0.,'sse':0.,'baseline':0.} for label in ['with_shares','without_shares']}
            for fold,held in enumerate(n.k.folds(d,r['seed'])):
                train,target=d.loc[~held].copy(),d.loc[held].copy();fd=od/f'fold_{fold}'
                assert sorted(target.cod_com.unique().tolist())==hybrid['outcomes'][name]['network_selection']['folds'][fold]['test_communes']
                assert sorted(target.cod_com.unique().tolist())==fs['folds'][fold]['test_communes']
                all_communes.extend(target.cod_com.unique().tolist())
                stored=np.load(fd/'density_predictions.npz')
                assert np.array_equal(stored['train_indices'],np.flatnonzero(~held))
                assert np.array_equal(stored['test_indices'],np.flatnonzero(held))
                reconstructed={}
                for with_shares,label in [(True,'with_shares'),(False,'without_shares')]:
                    tuning=json.loads((fd/f'density_{label}_selection.json').read_text())
                    verify_selection(tuning)
                    assert tuning==fs['folds'][fold][label]['selection']
                    models=n.load_models(fd/f'density_{label}_outer.npz')
                    for model in models:assert set(model['train_communes']).isdisjoint(target.cod_com.unique())
                    pred,ratio=check_density_models(models,train,target,with_shares);density_floor_ratios.append(ratio)
                    reconstructed[label]=pred
                    for key in pred:assert np.allclose(pred[key],stored[f'test_{label}_{key}'],atol=1e-10,rtol=1e-10)
                    diagnostics=n.density_diagnostics(target,pred,train.A.mean())
                    for key,value in diagnostics.items():assert np.isclose(value,fs['folds'][fold][label][key],rtol=1e-10)
                    first_scores[label]['nll']+=diagnostics['density_NLL']*len(target)
                    first_scores[label]['sse']+=diagnostics['mean_prediction_SSE']
                    first_scores[label]['baseline']+=diagnostics['mean_baseline_SSE']
                    crossfit={key:np.empty((len(train),pred[key].shape[1])) for key in pred}
                    inner_nll=0.
                    for j,inside in enumerate(n.k.folds(train,r['seed']+100+fold,3)):
                        part_train,part_target=train.loc[~inside],train.loc[inside]
                        models=n.load_models(fd/f'density_{label}_inner_{j}.npz')
                        for model in models:assert set(model['train_communes']).isdisjoint(part_target.cod_com.unique())
                        prediction,ratio=check_density_models(models,part_train,part_target,with_shares)
                        density_floor_ratios.append(ratio)
                        inner_nll-=float(n.log_density(prediction,part_target.A.to_numpy()).sum())
                        for key in pred:crossfit[key][inside]=prediction[key]
                    assert np.isclose(inner_nll/len(train),tuning['selected']['heldout_density_NLL'],rtol=1e-10)
                    if with_shares:
                        for key in crossfit:assert np.allclose(crossfit[key],stored[f'train_{key}'],atol=1e-10,rtol=1e-10)
                a,q=n.quadrature(reconstructed['with_shares'],r['quadrature_start_nodes'])
                a2,q2=n.hermite_quadrature(reconstructed['with_shares'],64)
                checkpoints=n.load_models(fd/'outcome_cv_models.npz')
                p=n.b.Absorb(target);y=target.y.to_numpy();s=n.b.inputs(target)[0]
                baseline_sse+=float(np.sum(p.project(y)**2))
                for row in cv['grid']:
                    key=f"hidden_{'_'.join(map(str,row['hidden']))}_steps_{row['steps']}"
                    models=[m for m in checkpoints if m['hidden'].tolist()==row['hidden'] and int(m['epochs'])==row['steps']]
                    assert len(models)==len(n.h.CV_INITIALIZATIONS)
                    for model in models:
                        assert np.allclose(model['mean'],n.h.scaling(train)['mean'])
                        assert np.allclose(model['sd'],n.h.scaling(train)['sd'])
                    predictions=[n.integrate(model,s,a,q) for model in models]
                    prediction=np.mean(predictions,axis=0)
                    assert np.allclose(prediction,oof[key][held],atol=1e-9,rtol=1e-9)
                    scores[key]+=float(np.sum(p.project(y-prediction)**2))
                    if key==selected_key:
                        for model,prediction in zip(models,predictions):
                            error=float(np.sqrt(np.mean((prediction-n.integrate(model,s,a2,q2))**2)));errors.append(error)
                            assert error<1e-3*max(np.std(p.project(y)),1e-10)
                print(f'{name}: fold {fold+1}/5 both stages reconstructed',flush=True)
            assert len(all_communes)==len(set(all_communes))==101
            assert np.isclose(baseline_sse/len(d),cv['FE_only_heldout_MSE'],rtol=1e-12)
            for row in cv['grid']:
                key=f"hidden_{'_'.join(map(str,row['hidden']))}_steps_{row['steps']}"
                assert np.isclose(scores[key]/len(d),row['heldout_integrated_MSE'],rtol=1e-10)
                if row['steps']==0:assert np.all(oof[key]==0)
            for label in first_scores:
                assert np.isclose(first_scores[label]['nll']/len(d),fs[label]['density_NLL'],rtol=1e-10)
                assert np.isclose(1-first_scores[label]['sse']/first_scores[label]['baseline'],fs[label]['mean_prediction_R2'],rtol=1e-10)
            assert np.isclose(fs['without_shares']['density_NLL']-fs['with_shares']['density_NLL'],fs['density_loss_improvement'])
            tuning=v['density_selection'];verify_selection(tuning)
            saved=np.load(od/'density_crossfit_predictions.npz')
            density={key:np.empty_like(saved[key]) for key in saved.files};total_nll=0.
            for j,held in enumerate(n.k.folds(d,r['seed'])):
                models=n.load_models(od/f'density_crossfit_{j}.npz')
                pred,ratio=check_density_models(models,d.loc[~held],d.loc[held],True);density_floor_ratios.append(ratio)
                for key in density:density[key][held]=pred[key]
                total_nll-=float(n.log_density(pred,d.loc[held,'A'].to_numpy()).sum())
            assert np.isclose(total_nll/len(d),tuning['selected']['heldout_density_NLL'],rtol=1e-10)
            for key in density:assert np.allclose(density[key],saved[key],atol=1e-10,rtol=1e-10)
            check_density_models(n.load_models(od/'density_full_models.npz'),d,d,True)
            a,q=n.quadrature(density,r['quadrature_start_nodes']);a2,q2=n.hermite_quadrature(density,64)
            s=n.b.inputs(d)[0];models=n.load_models(od/'models_primary.npz');values=[];derivatives=[];moments=[];final_errors=[]
            for model,fit in zip(models,v['primary']['initialization_fits']):
                value,derivative=n.h.evaluate(model,s);moment=n.integrate(model,s,a,q)
                values.append(value);derivatives.append(derivative);moments.append(moment)
                assert np.isclose(derivative[d.trim_keep].mean(),fit['average_derivative'],atol=1e-9)
                plus=s.copy();minus=s.copy();plus[:,0]+=1e-5;minus[:,0]-=1e-5
                numerical=(n.h.evaluate(model,plus)[0]-n.h.evaluate(model,minus)[0])/(2e-5)
                assert np.allclose(numerical,derivative,atol=1e-7,rtol=1e-6)
                altered=s.copy();altered[:,0]+=100
                assert np.allclose(moment,n.integrate(model,altered,a,q),atol=1e-12)
                error=float(np.sqrt(np.mean((moment-n.integrate(model,s,a2,q2))**2)));final_errors.append(error)
                assert np.isclose(error,fit['quadrature_RMSE_versus_Hermite64'],atol=1e-9)
            assert len(models)==len(n.h.FINAL_INITIALIZATIONS)
            assert np.allclose(np.mean(values,axis=0),cells.g_component,atol=1e-9,rtol=1e-9)
            assert np.allclose(np.mean(derivatives,axis=0),cells.marginal_effect,atol=1e-9,rtol=1e-9)
            assert np.allclose(np.min(derivatives,axis=0),cells.initialization_min_not_CI,atol=1e-9)
            assert np.allclose(np.max(derivatives,axis=0),cells.initialization_max_not_CI,atol=1e-9)
            moment=np.mean(moments,axis=0);assert np.allclose(moment,cells.integrated_g_component,atol=1e-9,rtol=1e-9)
            effect=cells.loc[cells.trim_keep,'marginal_effect'].mean()
            assert np.isclose(effect,v['primary']['average_derivative_trimmed'],atol=1e-9)
            assert np.isclose(effect,summary.loc[name,'average_derivative'],atol=1e-9)
            p=n.b.Absorb(d);nuisance=np.linalg.lstsq(p.D,d.y.to_numpy()-moment,rcond=None)[0]
            assert np.allclose(p.D@nuisance+moment,cells.integrated_moment_fitted,atol=1e-8)
            assert np.allclose(p.D@nuisance+np.mean(values,axis=0),cells.structural_fitted,atol=1e-8)
            residual=cells.y-cells.integrated_moment_fitted
            assert residual.groupby(cells.cod_com).mean().abs().max()<1e-8
            assert residual.groupby(cells.year).mean().abs().max()<1e-8
            assert np.isclose(np.mean(residual**2),v['primary']['ensemble_training_integrated_MSE'],rtol=1e-10)
            assert max(final_errors)<1e-3*np.std(p.project(d.y.to_numpy()))
            profiles=pd.read_csv(od/'common_time_profiles.csv')
            assert np.allclose(profiles,pd.read_csv(ROOT/'no_penalty'/name/'common_time_profiles.csv'))
            times=pd.read_csv(od/'time_heterogeneity.csv');assert len(times)==10
            for row in times.itertuples():
                points=np.column_stack([profiles[['A','age','male_share']].to_numpy(),np.full(len(profiles),row.year)])
                estimates=np.array([n.h.evaluate(model,points)[1].mean() for model in models])
                assert np.isclose(estimates.mean(),row.common_profile_average_derivative,atol=1e-9)
                assert np.isclose(estimates.min(),row.common_profile_initialization_min_not_CI,atol=1e-9)
                assert np.isclose(estimates.max(),row.common_profile_initialization_max_not_CI,atol=1e-9)
            audit['outcomes'][name]={'sample_and_outer_folds_match_hybrid':True,
                'MDN_density_selection_minimizes_saved_heldout_loss':True,
                'both_stage_predictions_reproduced_from_saved_models':True,
                'first_stage_training_communes_exclude_validation_communes':True,
                'MDN_prediction_inputs_exclude_outcome_and_observed_treatment':True,
                'all_structural_CV_scores_reproduced':True,
                'structural_selection_minimizes_saved_heldout_loss':True,
                'cell_and_time_derivatives_reproduced':True,
                'derivatives_agree_with_finite_differences':True,
                'fixed_effects_satisfy_integrated_normal_equations':True,
                'minimum_predicted_component_sd_in_training_A_sd':min(density_floor_ratios),
                'max_selected_CV_quadrature_RMSE':max(errors),
                'max_final_quadrature_RMSE':max(final_errors)}
    n.b.save_json(OUT/'validation.json',audit)
    report=(OUT/'report.txt').read_text().split('\nPOST-RUN VALIDATION\n')[0]
    extra=['','POST-RUN VALIDATION',
        'Both stages, all structural CV predictions, and the selected nested first-stage predictions were reconstructed from saved networks.',
        'Validation communes are excluded from network training and scaling. Density and structural choices minimize their saved held-out grids.',
        'Integrated predictions, derivatives, fixed effects and common-profile time estimates reproduce the reported results.',
        'Selected validation and final predictions pass independent Gauss-Hermite quadrature accuracy checks.',
        'Run: python project/npiv/kernel/verify_full_neural.py','']
    (OUT/'report.txt').write_text(report+'\n'.join(extra))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
