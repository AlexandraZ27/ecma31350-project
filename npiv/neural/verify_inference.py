"""Verify full-panel primary and trimmed sensitivity inference from saved draws."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'full_neural'


def check_scalar(row,point,draws):
    errors=draws-point;B=len(draws)
    critical=np.sort(np.abs(errors))[min(B-1,int(np.ceil(.95*(B+1)))-1)]
    low,high=np.quantile(errors,[.025,.975])
    expected={'point_estimate':point,'bootstrap_SE':draws.std(ddof=1),
              'centered_bootstrap_p':(1+np.count_nonzero(np.abs(errors)>=abs(point)))/(B+1),
              'symmetric_CI_low':point-critical,'symmetric_CI_high':point+critical,
              'basic_CI_low':point-high,'basic_CI_high':point-low,
              'bootstrap_mean':draws.mean(),'bootstrap_bias_diagnostic':draws.mean()-point}
    for key,value in expected.items():
        assert np.isclose(float(row[key]),value,atol=1e-12,rtol=1e-12),(key,row[key],value)
    assert int(row['replicates'])==B
    return {key:float(value) for key,value in expected.items()}


def main():
    results=json.loads((SOURCE/'results.json').read_text())
    inference=SOURCE/'inference';meta=json.loads((inference/'results.json').read_text())
    assert meta['source_results_sha256']==hashlib.sha256((SOURCE/'results.json').read_bytes()).hexdigest()
    assert meta['runner_sha256']==hashlib.sha256((HERE/'bootstrap_full_neural.py').read_bytes()).hexdigest()
    summary=pd.read_csv(inference/'summary.csv').set_index('outcome')
    sensitivity=pd.read_csv(inference/'trimmed_sensitivity.csv').set_index('outcome')
    exported=pd.read_csv(inference/'bootstrap_draws.csv').set_index(['outcome','replicate'])
    references=json.loads((HERE/'reference_design.json').read_text())['outcomes']
    B=meta['replicates'];paired={};methods=Counter();outcomes={}
    for name,result in results['outcomes'].items():
        cells=pd.read_csv(SOURCE/name/'cells_primary.csv')
        assert len(cells)==1010 and int(cells.trim_keep.sum())==910
        point=float(cells.marginal_effect.mean())
        trimmed=float(cells.loc[cells.trim_keep,'marginal_effect'].mean())
        assert np.isclose(point,result['primary']['average_derivative_all_cells'],atol=1e-12)
        assert np.isclose(trimmed,result['primary']['average_derivative_trimmed'],atol=1e-12)
        records=[json.loads((inference/'draws'/f'{name}_{b:04}.json').read_text()) for b in range(B)]
        full=np.array([r['result']['average_derivative_all_cells'] for r in records])
        trim=np.array([r['result'].get('average_derivative_trimmed',r['result']['average_derivative']) for r in records])
        main_stats=check_scalar(summary.loc[name],point,full)
        trim_stats=check_scalar(sensitivity.loc[name],trimmed,trim)
        check_scalar(meta['outcomes'][name],point,full)
        check_scalar(meta['sensitivity_outcomes'][name],trimmed,trim)
        assert summary.loc[name,'estimand']=='full_panel'
        assert sensitivity.loc[name,'estimand']=='within_year_5_95_sensitivity'
        assert int(summary.loc[name,'original_cells'])==1010
        assert int(sensitivity.loc[name,'original_cells'])==910
        original_fold={source:k for k,ids in enumerate(references[name]['outer_test_communes']) for source in ids}
        for b,record in enumerate(records):
            assert record['outcome']==name and record['replicate']==b
            assert len(record['commune_draws'])==101
            assert record['commune_draws']==paired.setdefault(b,record['commune_draws'])
            for copy in record['commune_draws']:
                assert copy['fold']==original_fold[copy['source_commune']]
            assert np.isclose(exported.loc[(name,b),'estimate'],full[b],atol=1e-12)
            assert np.isclose(exported.loc[(name,b),'trimmed_sensitivity_estimate'],trim[b],atol=1e-12)
            methods[record['result'].get('integration_method','original adaptive rule')]+=1
        outcomes[name]={'draws':B,'full_panel':main_stats,'trimmed_sensitivity':trim_stats,
                        'max_quadrature_RMSE':max(r['result']['max_quadrature_RMSE'] for r in records)}
    prefix=(SOURCE/'memo_work/model_exposition.tex').read_text().rstrip('\n')+'\n\n\n'
    assert (HERE.parents[1]/'NPIV.tex').read_text().startswith(prefix)
    audit={'all_checks_passed':True,'headline_estimand':'full_panel',
           'trimmed_average_is_sensitivity_only':True,
           'cell_means_and_both_bootstrap_distributions_verified':True,
           'original_exposition_and_template_unchanged':True,
           'paired_draws_across_outcomes':True,'source_communes_remain_in_original_folds':True,
           'reported_intervals_match_tests':True,'integration_rule_counts':dict(methods),'outcomes':outcomes}
    (inference/'validation.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
