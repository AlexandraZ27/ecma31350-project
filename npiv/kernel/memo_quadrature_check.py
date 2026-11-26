"""Check a shorter Gaussian integration rule against the saved full estimator."""
import importlib.util
from pathlib import Path
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('boot', HERE / 'bootstrap_full_neural.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
n = b.n


from mixture_quadrature import mixture_gauss


def main():
    b.setup_worker(); rows=[]
    for name, settings in b.RESULTS['outcomes'].items():
        for rep in [0, 7, 20]:
            start=time.time()
            with threadpool_limits(limits=1):
                d,_=b.sample_communes(b.DATA[name],rep)
                profiles=b.pd.read_csv(b.SOURCE/name/'common_time_profiles.csv')
                fitted=b.fit_statistic(d,settings,profiles)
            saved=json.loads((b.SOURCE/'inference'/'draws'/f'{name}_{rep:04}.json').read_text())['result']
            assert 'integration_method' not in saved, 'Comparison must use an original adaptive-rule draw'
            difference=fitted['average_derivative']-saved['average_derivative']
            time_difference=float(np.max(np.abs(np.array(saved['time_common_profile_derivative'])-fitted['time_common_profile_derivative'])))
            assert abs(difference)<1e-5
            assert time_difference<3e-5
            row={'outcome':name,'replicate':rep,'point_difference':difference,
                 'max_time_difference':time_difference,'quadrature_RMSE':fitted['max_quadrature_RMSE'],
                 'seconds':time.time()-start}
            rows.append(row);print(row,flush=True)
    n.b.save_json(b.SOURCE/'inference'/'quadrature_refit_validation.json',rows)


if __name__=='__main__':main()
