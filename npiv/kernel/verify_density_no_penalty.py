from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

root=Path('project/npiv/kernel')
out=root/'no_penalty'
spec=importlib.util.spec_from_file_location('density_no_penalty',root/'run_density_no_penalty.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
r=json.loads((out/'results.json').read_text())
assert set(r['outcomes'])=={'pc_19','pc_20','vict_agreg'}
assert r['ridge_penalty']==0 and n.RIDGE==0
assert r['runner_sha256']==hashlib.sha256((root/'run_density_no_penalty.py').read_bytes()).hexdigest()
assert r['helper_sha256']==hashlib.sha256((root/'run_kernel.py').read_bytes()).hexdigest()
assert all(json.loads((out/'self_checks.json').read_text()).values())
audit={'zero_explicit_ridge_in_both_stages':True,'source_hashes_match':True,'all_numerical_checks_passed':True,'outcomes':{}}
summary=pd.read_csv(out/'summary.csv').set_index('outcome')
with threadpool_limits(limits=2):
    for name,v in r['outcomes'].items():
        od=out/name
        d=pd.read_csv(od/'cells_primary.csv');d.attrs['instrument_mode']='raw_shares'
        stored=np.load(od/'model_primary.npz')
        model={key:stored[key] for key in ['centers','widths','coef']}
        assert model['centers'].shape==(1010,4)
        assert len(d)==1010 and d.cod_com.nunique()==101 and d.year.nunique()==10
        assert v['solver']['ridge_penalty']==0
        assert np.isfinite(d.select_dtypes(include=[np.number]).to_numpy()).all()
        for key,loss in [('structural_bandwidth_selection','heldout_projected_integrated_MSE'),('density_bandwidth_selection','heldout_density_negative_log_likelihood')]:
            tuning=v[key]
            assert tuning['ridge_penalty']==0
            assert np.isclose(tuning['selected'][loss],min(row[loss] for row in tuning['grid']),rtol=1e-12)
        held=[]
        baseline_sse=0.
        for fold in v['structural_bandwidth_selection']['folds']:
            assert set(fold['train_communes']).isdisjoint(fold['test_communes'])
            assert len(set(fold['train_communes'])|set(fold['test_communes']))==101
            held.extend(fold['test_communes'])
            test=d.loc[d.cod_com.isin(fold['test_communes'])]
            baseline=n.b.Absorb(test).transform(test.y.to_numpy(float))
            baseline_sse+=float(baseline@baseline)
        assert len(held)==len(set(held))==101
        raw,derivative=n.b.evaluate(model,n.b.inputs(d)[0])
        assert np.allclose(raw,d.g_component,rtol=1e-8,atol=1e-8)
        assert np.allclose(derivative,d.marginal_effect,rtol=1e-8,atol=1e-8)
        assert np.isclose(derivative[d.trim_keep].mean(),v['primary']['average_derivative_trimmed'],rtol=1e-10,atol=1e-8)
        assert np.isclose(summary.loc[name,'average_derivative'],v['primary']['average_derivative_trimmed'])
        assert np.allclose(d.f_component,d.g_component+d.year_main_effect_inside_f)
        assert np.allclose(d.structural_fitted,d.f_component+d.commune_fixed_effect_component)
        assert d.groupby('cod_com').commune_fixed_effect_component.nunique().max()==1
        assert d.groupby('year').year_main_effect_inside_f.nunique().max()==1
        assert np.allclose(d.loc[d.year==2008,'year_main_effect_inside_f'],0)
        residual=d.y-d.structural_fitted
        assert residual.groupby(d.cod_com).mean().abs().max()<1e-7
        assert residual.groupby(d.year).mean().abs().max()<1e-7
        chosen=v['density_bandwidth_selection']['selected']
        params=(chosen['share_multiplier'],chosen['control_multiplier'],chosen['time_bandwidth'],chosen['density_A_sd_multiplier'])
        parts=n.crossfit_density(d,n.folds(d,r['seed']),params)
        first=parts[0][1]
        altered=dict(first,target=first['target'].copy())
        altered['target'][:,0]+=100.
        assert np.allclose(n.integrated_features(first,model['centers'],model['widths']),n.integrated_features(altered,model['centers'],model['widths']))
        B=n.crossfit_features(parts,model['centers'],model['widths'],len(d))
        p=n.b.Absorb(d)
        actual=d.y.to_numpy()-p.project(d.y.to_numpy())+p.project(B@model['coef'])
        assert np.allclose(actual,d.integrated_moment_fitted,rtol=1e-8,atol=1e-7)
        mse=np.mean(p.project(d.y.to_numpy()-B@model['coef'])**2)
        assert np.isclose(mse,v['solver']['projected_training_integrated_MSE'],atol=1e-12,rtol=1e-6)
        profiles=pd.read_csv(od/'common_time_profiles.csv')
        time=pd.read_csv(od/'time_heterogeneity_primary.csv')
        assert len(time)==10 and not any('split' in col for col in time.columns)
        assert len(profiles)==v['primary']['time_heterogeneity']['profile_count']
        for row in time.itertuples():
            reference=np.column_stack([profiles[['A','age','male_share']].to_numpy(),np.full(len(profiles),row.year)])
            effect=n.b.evaluate(model,reference)[1].mean()
            assert np.isclose(effect,row.common_profile_average_derivative,rtol=1e-8,atol=1e-8)
        audit['outcomes'][name]={'bandwidths_minimize_saved_heldout_losses':True,'outer_commune_groups_disjoint':True,
            'saved_model_reproduces_cell_and_time_derivatives':True,'density_reconstruction_reproduces_integrated_predictions':True,
            'integrated_prediction_does_not_use_target_observed_treatment':True,
            'fixed_effect_normal_equations_hold':True,'average_derivative':v['primary']['average_derivative_trimmed'],
            'numerical_rank':v['solver']['numerical_rank'],'columns':v['solver']['columns'],
            'heldout_FE_only_MSE':baseline_sse/len(d),
            'selected_heldout_integrated_MSE':v['structural_bandwidth_selection']['selected']['heldout_projected_integrated_MSE'],
            'heldout_loss_improvement_over_FE_only':baseline_sse/len(d)-v['structural_bandwidth_selection']['selected']['heldout_projected_integrated_MSE'],
            'density_loss_improvement':v['first_stage_diagnostics']['density_loss_improvement']}
(out/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
report_path=out/'report.txt'
report=report_path.read_text().split('\nPOST-RUN VALIDATION\n')[0]
extra=['','POST-RUN VALIDATION',
       'Saved models and reconstructed conditional densities reproduce the reported cell and time estimates.',
       'Integrated held-out predictions do not use the held-out observed treatment value.',
       'Both bandwidth choices attain the minimum of their saved held-out loss grids.',
       'The following FE-only benchmark uses the same held-out commune/year projection as the structural CV criterion:']
for name,v in audit['outcomes'].items():
    extra.append(f"  {name}: FE-only MSE {v['heldout_FE_only_MSE']:.6g}; selected unpenalized integrated MSE {v['selected_heldout_integrated_MSE']:.6g}; improvement {v['heldout_loss_improvement_over_FE_only']:.6g}.")
extra.extend(['The unpenalized fit interpolates the training moment equations but has worse held-out loss than the FE-only benchmark for all three outcomes.',
              'Continuous bandwidth is at the lower edge of the expanded grid in all three cases. These estimates are sensitive to the allowed bandwidth range.',
              'Verification: python3 project/npiv/kernel/verify_density_no_penalty.py',''])
report_path.write_text(report+'\n'.join(extra))
print(json.dumps(audit,indent=2))
