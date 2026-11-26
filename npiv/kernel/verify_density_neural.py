"""Reconstruct and verify the saved hybrid KDE/neural NPIV results."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

root=Path(__file__).resolve().parent
out=root/'neural_outcome'
spec=importlib.util.spec_from_file_location('density_neural',root/'run_density_neural.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
n.torch.set_num_threads(2);n.torch.set_num_interop_threads(1)
results=json.loads((out/'results.json').read_text())
reference=json.loads((root/'no_penalty/results.json').read_text())
assert set(results['outcomes'])=={'pc_19','pc_20','vict_agreg'}
assert results['runner_sha256']==hashlib.sha256((root/'run_density_neural.py').read_bytes()).hexdigest()
for file,sha in results['helper_sha256'].items():
    assert sha==hashlib.sha256((root/file).read_bytes()).hexdigest()
assert results['reference_sha256']==hashlib.sha256((root/'no_penalty/results.json').read_bytes()).hexdigest()
assert results['source_sha256']==hashlib.sha256(n.b.DATA.read_bytes()).hexdigest()
assert results['specification']['coefficient_penalty']==0
assert results['specification']['dropout']==0
assert all(json.loads((out/'self_checks.json').read_text()).values())
summary=pd.read_csv(out/'summary.csv').set_index('outcome')
audit={'source_hashes_match':True,'numerical_self_checks_pass':True,
       'zero_explicit_weight_decay_and_dropout':True,'outcomes':{}}

with threadpool_limits(limits=2):
    for name,r in results['outcomes'].items():
        od=out/name
        d=pd.read_csv(od/'cells_primary.csv');d.attrs['instrument_mode']='raw_shares'
        raw,metadata=n.b.load_cells(name,'raw_shares')
        assert len(d)==1010 and d.cod_com.nunique()==101 and d.year.nunique()==10
        assert np.allclose(d[raw.columns].select_dtypes(include=[np.number]),raw.select_dtypes(include=[np.number]))
        assert np.isfinite(d.select_dtypes(include=[np.number]).to_numpy()).all()
        cv=r['network_selection'];old=reference['outcomes'][name]
        assert r['density_bandwidth_selection']==old['density_bandwidth_selection']
        assert r['first_stage_diagnostics']==old['first_stage_diagnostics']
        table=pd.read_csv(od/'network_cv.csv',dtype={'hidden':str})
        assert len(table)==len(n.ARCHITECTURES)*len(n.CHECKPOINTS)
        chosen=table.loc[(table.hidden=='x'.join(map(str,cv['selected_hidden'])))&(table.steps==cv['selected_steps'])]
        assert len(chosen)==1
        assert np.isclose(chosen.heldout_integrated_MSE.iloc[0],table.heldout_integrated_MSE.min(),rtol=1e-12)
        assert np.isclose(chosen.heldout_integrated_MSE.iloc[0],cv['selected_heldout_integrated_MSE'],rtol=1e-12)
        oof=np.load(od/'cv_predictions.npz')
        accumulated={key:0. for key in oof.files};baseline=0.;communes=[];cv_quadrature_errors=[]
        for fold in cv['folds']:
            train,test=set(fold['train_communes']),set(fold['test_communes'])
            assert train.isdisjoint(test) and len(train|test)==101
            assert sorted(test)==old['first_stage_diagnostics']['folds'][fold['fold']]['test_communes']
            assert np.allclose(fold['density_metadata']['density_params'],n.density_params(old['first_stage_diagnostics']['folds'][fold['fold']]['with_shares']['bandwidth_selection']))
            communes.extend(test);held=d.cod_com.isin(test).to_numpy()
            panel=d.loc[held];p=n.b.Absorb(panel);y=panel.y.to_numpy(float)
            baseline+=float(np.sum(p.project(y)**2))
            for key in oof.files:
                accumulated[key]+=float(np.sum(p.project(y-oof[key][held])**2))
            # Replay only the selected training path, and double the validation
            # integration grid. This checks the saved out-of-fold predictions
            # against reconstructed densities and actual trained networks.
            train_d,target,grid_f,q_f,qt_f,_=n.build_fold_densities(
                raw,old,results['seed'],fold['fold'],held,results['quadrature_nodes'])
            params=n.density_params(old['first_stage_diagnostics']['folds'][fold['fold']]['with_shares']['bandwidth_selection'])
            density=n.k.density_model(train_d,target,params)
            grid_f2,w_f2=n.integration_rule([density],2*results['quadrature_nodes'])
            qt_f2,_=n.density_quadrature(density,grid_f2,w_f2)
            replay=[]
            for seed in n.CV_INITIALIZATIONS:
                model,_=n.train_network(train_d,grid_f,q_f,tuple(cv['selected_hidden']),seed,cv['selected_steps'])
                pred=n.integrate_saved(model,n.b.inputs(target)[0],grid_f,qt_f)
                check=n.integrate_saved(model,n.b.inputs(target)[0],grid_f2,qt_f2)
                cv_quadrature_errors.append(float(np.sqrt(np.mean((pred-check)**2))))
                replay.append(pred)
            selected_key=f"hidden_{'_'.join(map(str,cv['selected_hidden']))}_steps_{cv['selected_steps']}"
            assert np.allclose(np.mean(replay,axis=0),oof[selected_key][held],atol=1e-8,rtol=1e-8)
            assert max(cv_quadrature_errors[-len(n.CV_INITIALIZATIONS):])<1e-3*max(np.std(p.project(y)),1e-10)
        assert len(communes)==len(set(communes))==101
        assert np.isclose(baseline/len(d),cv['FE_only_heldout_MSE'],rtol=1e-12)
        for row in table.itertuples():
            key=f"hidden_{row.hidden.replace('x','_')}_steps_{row.steps}"
            assert np.isclose(accumulated[key]/len(d),row.heldout_integrated_MSE,rtol=1e-11)
            if row.steps==0:
                assert np.all(oof[key]==0)
                assert np.isclose(row.heldout_integrated_MSE,baseline/len(d),rtol=1e-12)
        stored=np.load(od/'models_primary.npz')
        models=[]
        for j,seed in enumerate(n.FINAL_INITIALIZATIONS):
            prefix=f'model_{j}_'
            model={key[len(prefix):]:stored[key] for key in stored.files if key.startswith(prefix)}
            assert int(model['seed'])==seed and int(model['epochs'])==cv['selected_steps']
            assert model['hidden'].tolist()==cv['selected_hidden']
            assert model['weight_0'].shape[1]==13  # A, age, male share, ten years; no commune ID
            models.append(model)
        s=n.b.inputs(d)[0]
        values,derivatives=zip(*(n.evaluate(model,s) for model in models))
        assert np.allclose(np.mean(values,axis=0),d.g_component,atol=1e-9,rtol=1e-9)
        assert np.allclose(np.mean(derivatives,axis=0),d.marginal_effect,atol=1e-9,rtol=1e-9)
        assert np.allclose(np.min(derivatives,axis=0),d.initialization_min_not_CI,atol=1e-9)
        assert np.allclose(np.max(derivatives,axis=0),d.initialization_max_not_CI,atol=1e-9)
        effect=d.loc[d.trim_keep,'marginal_effect'].mean()
        assert np.isclose(effect,r['primary']['average_derivative_trimmed'],atol=1e-9)
        assert np.isclose(effect,summary.loc[name,'average_derivative'],atol=1e-9)
        # Rebuild cross-fitted densities independently from the saved estimates.
        params=n.density_params(r['density_bandwidth_selection'])
        parts=n.k.crossfit_density(d,n.k.folds(d,results['seed']),params)
        grid,w=n.integration_rule([m for _,m in parts],results['quadrature_nodes'])
        q,mass_error=n.combined_quadrature(parts,len(d),grid,w)
        grid2,w2=n.integration_rule([m for _,m in parts],2*results['quadrature_nodes'])
        q2,_=n.combined_quadrature(parts,len(d),grid2,w2)
        moments=[n.integrate_saved(model,s,grid,q) for model in models]
        moment=np.mean(moments,axis=0)
        assert np.allclose(moment,d.integrated_g_component,atol=1e-9,rtol=1e-9)
        quadrature_errors=[]
        for model,prediction,fit in zip(models,moments,r['primary']['initialization_fits']):
            error=float(np.sqrt(np.mean((prediction-n.integrate_saved(model,s,grid2,q2))**2)))
            quadrature_errors.append(error)
            assert np.isclose(error,fit['quadrature_prediction_RMSE_versus_double_nodes'],atol=1e-10)
            plus=s.copy();minus=s.copy();plus[:,0]+=1e-5;minus[:,0]-=1e-5
            numerical=(n.evaluate(model,plus)[0]-n.evaluate(model,minus)[0])/(2e-5)
            assert np.allclose(numerical,n.evaluate(model,s)[1],atol=1e-7,rtol=1e-6)
        p=n.b.Absorb(d)
        nuisance=np.linalg.lstsq(p.D,d.y.to_numpy()-moment,rcond=None)[0]
        assert np.allclose(p.D@nuisance+moment,d.integrated_moment_fitted,atol=1e-8)
        assert np.allclose(d.structural_fitted,p.D@nuisance+d.g_component,atol=1e-8)
        assert np.allclose(d.f_component,d.g_component+d.year_main_effect_inside_f,atol=1e-9)
        residual=d.y-d.integrated_moment_fitted
        assert residual.groupby(d.cod_com).mean().abs().max()<1e-8
        assert residual.groupby(d.year).mean().abs().max()<1e-8
        assert np.allclose(d.loc[d.year==2008,'year_main_effect_inside_f'],0)
        assert d.groupby('cod_com').commune_fixed_effect_component.nunique().max()==1
        assert d.groupby('year').year_main_effect_inside_f.nunique().max()==1
        assert np.isclose(np.mean(residual**2),r['primary']['ensemble_training_integrated_MSE'],atol=1e-9)
        assert max(quadrature_errors)<=1e-3*np.std(p.project(d.y.to_numpy()))
        profiles=pd.read_csv(od/'common_time_profiles.csv')
        assert np.allclose(profiles,pd.read_csv(root/'no_penalty'/name/'common_time_profiles.csv'))
        time=pd.read_csv(od/'time_heterogeneity.csv')
        assert len(time)==10 and len(profiles)==r['primary']['time_heterogeneity']['profile_count']
        for row in time.itertuples():
            target=np.column_stack([profiles[['A','age','male_share']].to_numpy(),np.full(len(profiles),row.year)])
            estimates=np.array([n.evaluate(model,target)[1].mean() for model in models])
            assert np.isclose(estimates.mean(),row.common_profile_average_derivative,atol=1e-9)
            assert np.isclose(estimates.min(),row.common_profile_initialization_min_not_CI,atol=1e-9)
            assert np.isclose(estimates.max(),row.common_profile_initialization_max_not_CI,atol=1e-9)
        audit['outcomes'][name]={
            'network_selection_minimizes_saved_heldout_loss_grid':True,
            'heldout_scores_reproduced_from_saved_predictions':True,
            'selected_CV_fits_reproduced_by_retraining':True,
            'communes_disjoint_between_training_and_validation':True,
            'first_stage_matches_kernel_comparison':True,
            'saved_models_reproduce_cell_and_time_derivatives':True,
            'analytic_derivatives_agree_with_finite_differences':True,
            'reconstructed_KDE_reproduces_integrated_predictions':True,
            'fixed_effects_satisfy_integrated_loss_normal_equations':True,
            'quadrature_mass_error':mass_error,
            'max_quadrature_prediction_RMSE':max(quadrature_errors),
            'max_selected_CV_quadrature_prediction_RMSE':max(cv_quadrature_errors),
            'average_derivative':effect,
            'selected_heldout_integrated_MSE':cv['selected_heldout_integrated_MSE'],
            'FE_only_heldout_MSE':cv['FE_only_heldout_MSE']}

n.b.save_json(out/'validation.json',audit)
report=(out/'report.txt').read_text().split('\nPOST-RUN VALIDATION\n')[0]
extra=['','POST-RUN VALIDATION',
       'All three saved models, cell derivatives, time comparisons and integrated predictions were independently reconstructed.',
       'All held-out CV scores were reproduced from saved predictions; selected candidates attain the minimum saved loss.',
       'Commune folds are disjoint. The first-stage densities and bandwidths match the kernel comparison.',
       'Fixed effects satisfy the normal equations of the integrated training loss.',
       'Numerical integration and analytic derivatives pass independent accuracy checks.',
       'Run verification with the same Python environment: python project/npiv/kernel/verify_density_neural.py','']
(out/'report.txt').write_text(report+'\n'.join(extra))
print(json.dumps(audit,indent=2))
