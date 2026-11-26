"""Hybrid NPIV: existing conditional KDE first stage + a compact outcome network.

Same panel, raw shares, density bandwidths and commune folds as no_penalty/.
Outcome-network size and training steps minimize held-out integrated NPIV loss.
Explicit weight decay and dropout are zero. Smooth tanh networks, early stopping
and finite architecture choices control complexity. This is not full MDN DeepIV.
Run using a Python environment with torch, numpy, scipy, pandas and matplotlib.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ecma31350-kernel-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy import linalg
import torch
from torch import nn
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('density_reference',HERE/'run_density_no_penalty.py')
k=importlib.util.module_from_spec(spec);spec.loader.exec_module(k)
b=k.b
ARCHITECTURES=[(8,),(16,8)]
CHECKPOINTS=[0,50,100,200,400,600]
CV_INITIALIZATIONS=[17,43]
FINAL_INITIALIZATIONS=[101,211,307,401,503]
LEARNING_RATE=.005
DTYPE=torch.float64
SPECIFICATION=dict(k.SPECIFICATION,
    structural_class="compact tanh neural network in immigration, demographics and year indicators",
    time_kernel="year indicators inside the outcome network, with unrestricted network interactions",
    interactions="immigration, demographics and categorical year enter jointly through smooth tanh hidden layers",
    feature_absorption="commune and year main effects projected from integrated network predictions and outcomes; raw A never demeaned",
    coefficient_penalty=0.,dropout=0.,
    complexity_selection="network architecture and number of Adam steps chosen by commune-grouped held-out integrated loss",
    first_stage="unchanged conditional Gaussian KDE, including the previously selected fold-specific bandwidths")


def density_params(selection):
    p=selection['selected']
    return (p['share_multiplier'],p['control_multiplier'],p['time_bandwidth'],p['density_A_sd_multiplier'])


def integration_rule(densities,nodes):
    lower=min(float(d['donor_A'].min()-8*d['density_A_bandwidth']) for d in densities)
    upper=max(float(d['donor_A'].max()+8*d['density_A_bandwidth']) for d in densities)
    x,w=np.polynomial.legendre.leggauss(nodes)
    return (x+1)*(upper-lower)/2+lower,w*(upper-lower)/2


def density_quadrature(model,grid,weights):
    h=model['density_A_bandwidth']
    pdf=np.exp(-.5*((model['donor_A'][:,None]-grid[None,:])/h)**2)/(h*np.sqrt(2*np.pi))
    q=(model['weights']@pdf)*weights[None,:]
    error=float(np.max(np.abs(q.sum(axis=1)-1)))
    if error>1e-7:
        raise ValueError(f'Density quadrature mass error {error:g}; increase --nodes')
    return q/q.sum(axis=1,keepdims=True),error


def combined_quadrature(parts,n,grid,weights):
    q=np.empty((n,len(grid)));errors=[]
    for idx,model in parts:
        q[idx],error=density_quadrature(model,grid,weights);errors.append(error)
    return q,max(errors)


def scaling(d):
    s=b.inputs(d)[0]
    return {'mean':s[:,:3].mean(axis=0),'sd':np.maximum(s[:,:3].std(axis=0),1e-10),
            'years':np.sort(d.year.unique()).astype(float)}


def context(s,scale):
    year=(s[:,-1,None]==scale['years'][None,:]).astype(float)
    if not np.all(year.sum(axis=1)==1):raise ValueError('Unknown year category')
    return np.column_stack([(s[:,1:3]-scale['mean'][1:3])/scale['sd'][1:3],year])


class OutcomeNetwork(nn.Module):
    def __init__(self,context_dimension,hidden):
        super().__init__()
        dims=[context_dimension+1,*hidden,1]
        self.layers=nn.ModuleList([nn.Linear(a,c,dtype=DTYPE) for a,c in zip(dims[:-1],dims[1:])])
        # Epoch zero is the explicit f=0 / additive-effects-only candidate.
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)

    def grid(self,standardized_a,ctx):
        first=self.layers[0]
        value=(ctx@first.weight[:,1:].T+first.bias)[:,None,:]
        value=torch.tanh(value+standardized_a[None,:,None]*first.weight[:,0][None,None,:])
        for layer in self.layers[1:-1]:value=torch.tanh(layer(value))
        return self.layers[-1](value).squeeze(-1)

    def points(self,x):
        for layer in self.layers[:-1]:x=torch.tanh(layer(x))
        return self.layers[-1](x).squeeze(-1)


def export_model(network,scale,y_scale,hidden,seed,epochs):
    model={**scale,'y_scale':np.array(y_scale),'hidden':np.array(hidden,dtype=int),
           'seed':np.array(seed),'epochs':np.array(epochs)}
    for j,layer in enumerate(network.layers):
        model[f'weight_{j}']=layer.weight.detach().numpy().copy()
        model[f'bias_{j}']=layer.bias.detach().numpy().copy()
    return model


def evaluate(model,s):
    x=np.column_stack([(s[:,0]-model['mean'][0])/model['sd'][0],context(s,model)])
    derivative=np.zeros_like(x);derivative[:,0]=1/model['sd'][0]
    layers=len(model['hidden'])+1
    for j in range(layers):
        x=x@model[f'weight_{j}'].T+model[f'bias_{j}']
        derivative=derivative@model[f'weight_{j}'].T
        if j<layers-1:
            x=np.tanh(x);derivative*=1-x*x
    value=x[:,0]*float(model['y_scale']);derivative=derivative[:,0]*float(model['y_scale'])
    if not np.isfinite(value).all() or not np.isfinite(derivative).all():raise FloatingPointError('Nonfinite network evaluation')
    return value,derivative


def integrate_saved(model,s,grid,q):
    # Grid evaluation without replicating the full one-hot input matrix.
    c=context(s,model);a=(grid-model['mean'][0])/model['sd'][0]
    w=model['weight_0']
    value=np.tanh((c@w[:,1:].T+model['bias_0'])[:,None,:]+a[None,:,None]*w[:,0][None,None,:])
    for j in range(1,len(model['hidden'])):
        value=np.tanh(value@model[f'weight_{j}'].T+model[f'bias_{j}'])
    last=len(model['hidden'])
    value=(value@model[f'weight_{last}'].T+model[f'bias_{last}'])[:,:,0]
    return (value*q).sum(axis=1)*float(model['y_scale'])


def tensors(d,grid,q,scale,y_scale):
    p=b.Absorb(d)
    return {'context':torch.tensor(context(b.inputs(d)[0],scale),dtype=DTYPE),
            'a':torch.tensor((grid-scale['mean'][0])/scale['sd'][0],dtype=DTYPE),
            'weights':torch.tensor(q,dtype=DTYPE),'Q':torch.tensor(p.Q,dtype=DTYPE),
            'target':torch.tensor(p.transform(d.y.to_numpy(float))/y_scale,dtype=DTYPE)}


def expected(network,pack):
    return (network.grid(pack['a'],pack['context'])*pack['weights']).sum(dim=1)


def projected_residual(moment,pack):
    return pack['target']-(moment-pack['Q']@(pack['Q'].T@moment))


def train_network(d,grid,q,hidden,seed,epochs,validation=None,record_steps=None):
    scale=scaling(d);p=b.Absorb(d)
    y_scale=max(float(np.std(p.transform(d.y.to_numpy(float)))),1e-10)
    pack=tensors(d,grid,q,scale,y_scale)
    test=tensors(*validation,scale,y_scale) if validation is not None else None
    torch.manual_seed(seed)
    network=OutcomeNetwork(pack['context'].shape[1],hidden)
    optimizer=torch.optim.Adam(network.parameters(),lr=LEARNING_RATE,weight_decay=0.)
    records=[];record_steps=set(record_steps or [epochs])
    for step in range(epochs+1):
        if step in record_steps:
            with torch.no_grad():
                fitted=expected(network,pack)
                record={'steps':step,'training_integrated_MSE':float(torch.mean(projected_residual(fitted,pack)**2))*y_scale**2}
                if test is not None:
                    prediction=expected(network,test)
                    record['validation_integrated_MSE']=float(torch.mean(projected_residual(prediction,test)**2))*y_scale**2
                    record['validation_prediction']=prediction.numpy()*y_scale
                records.append(record)
        if step==epochs:break
        optimizer.zero_grad()
        loss=torch.mean(projected_residual(expected(network,pack),pack)**2)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward();optimizer.step()
    return export_model(network,scale,y_scale,hidden,seed,epochs),records


def build_fold_densities(d,reference,seed,fold,held,nodes):
    train,target=d.loc[~held].copy(),d.loc[held].copy()
    fs=reference['first_stage_diagnostics']['folds'][fold]
    assert sorted(target.cod_com.unique().tolist())==fs['test_communes']
    params=density_params(fs['with_shares']['bandwidth_selection'])
    parts=k.crossfit_density(train,k.folds(train,seed+100+fold,3),params)
    test_density=k.density_model(train,target,params)
    grid,weights=integration_rule([m for _,m in parts]+[test_density],nodes)
    q,error=combined_quadrature(parts,len(train),grid,weights)
    qt,error_t=density_quadrature(test_density,grid,weights)
    return train,target,grid,q,qt,{'density_params':params,'mass_error':max(error,error_t)}


def tune_network(d,reference,seed,nodes,out):
    candidates=[(hidden,steps) for hidden in ARCHITECTURES for steps in CHECKPOINTS]
    losses={candidate:0. for candidate in candidates}
    oof={candidate:np.empty(len(d)) for candidate in candidates}
    baseline_sse=0.;records=[]
    for fold,held in enumerate(k.folds(d,seed)):
        train,target,grid,q,qt,density_meta=build_fold_densities(d,reference,seed,fold,held,nodes)
        p=b.Absorb(target);yt=target.y.to_numpy(float)
        baseline_sse+=float(np.sum(p.project(yt)**2))
        fold_records=[]
        for hidden in ARCHITECTURES:
            histories=[]
            for init in CV_INITIALIZATIONS:
                _,history=train_network(train,grid,q,hidden,init,max(CHECKPOINTS),
                    validation=(target,grid,qt),record_steps=CHECKPOINTS)
                histories.append(history)
            for j,steps in enumerate(CHECKPOINTS):
                prediction=np.mean([history[j]['validation_prediction'] for history in histories],axis=0)
                error=p.project(yt-prediction);sse=float(error@error)
                candidate=(hidden,steps);losses[candidate]+=sse;oof[candidate][held]=prediction
                fold_records.append({'hidden':list(hidden),'steps':steps,'ensemble_heldout_integrated_MSE':sse/len(target),
                    'individual_initialization_MSE':[history[j]['validation_integrated_MSE'] for history in histories],
                    'mean_training_integrated_MSE':float(np.mean([history[j]['training_integrated_MSE'] for history in histories]))})
        records.append({'fold':fold,'train_communes':sorted(train.cod_com.unique().tolist()),
                        'test_communes':sorted(target.cod_com.unique().tolist()),'test_n':len(target),
                        'density_metadata':density_meta,'candidates':fold_records})
        print(f'  neural validation fold {fold+1}/5 complete',flush=True)
    selected=min(candidates,key=lambda candidate:losses[candidate])
    table=[{'hidden':'x'.join(map(str,hidden)),'steps':steps,
            'heldout_integrated_MSE':losses[(hidden,steps)]/len(d)} for hidden,steps in candidates]
    pd.DataFrame(table).to_csv(out/'network_cv.csv',index=False)
    np.savez_compressed(out/'cv_predictions.npz',**{f"hidden_{'_'.join(map(str,h))}_steps_{s}":oof[(h,s)] for h,s in candidates})
    meta={'selected_hidden':list(selected[0]),'selected_steps':selected[1],
          'selected_heldout_integrated_MSE':losses[selected]/len(d),'FE_only_heldout_MSE':baseline_sse/len(d),
          'heldout_loss_improvement_over_FE_only':(baseline_sse-losses[selected])/len(d),
          'cv_initializations':CV_INITIALIZATIONS,'grid':table,'folds':records,
          'weight_decay':0.,'dropout':0.,'learning_rate':LEARNING_RATE,
          'selection_note':'Architecture and a common stopping step minimize the five-fold integrated validation loss of a two-initialization ensemble. This is a selection score, not an untouched test-set estimate.',
          'selected_maximum_steps':selected[1]==max(CHECKPOINTS)}
    return selected,meta


def time_results(d,models,out,reference_directory):
    profiles=pd.read_csv(reference_directory/out.name/'common_time_profiles.csv')
    # Exact same reference profiles as the kernel comparison.
    source=d.set_index(['cod_com','year'])
    for row in profiles.itertuples():
        actual=source.loc[(row.source_commune,row.source_year),['A','age','male_share']].to_numpy(float)
        assert np.allclose(actual,[row.A,row.age,row.male_share],atol=1e-10)
    rows=[]
    for year in sorted(d.year.unique()):
        reference=np.column_stack([profiles[['A','age','male_share']].to_numpy(float),np.full(len(profiles),year)])
        means=np.array([evaluate(model,reference)[1].mean() for model in models])
        observed=d.loc[(d.year==year)&d.trim_keep]
        observed_means=[evaluate(model,b.inputs(observed)[0])[1].mean() for model in models]
        rows.append({'year':int(year),'common_profile_average_derivative':float(means.mean()),
                     'common_profile_initialization_min_not_CI':float(means.min()),
                     'common_profile_initialization_max_not_CI':float(means.max()),
                     'observed_profile_average_derivative':float(np.mean(observed_means)),
                     'common_profile_count':len(profiles),
                     **{f'initialization_{int(model["seed"])}':float(effect) for model,effect in zip(models,means)}})
    frame=pd.DataFrame(rows);frame.to_csv(out/'time_heterogeneity.csv',index=False)
    profiles.to_csv(out/'common_time_profiles.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
    for j,model in enumerate(models):
        ax.plot(frame.year,frame[f'initialization_{int(model["seed"])}'],color='grey',alpha=.3,lw=.8,
                label='Individual initializations (not CIs)' if j==0 else None)
    ax.plot(frame.year,frame.common_profile_average_derivative,color='#185b75',marker='o',label=f'Same {len(profiles)} profiles each year')
    ax.plot(frame.year,frame.observed_profile_average_derivative,color='#bb6b31',marker='s',ls='--',label="Each year's observed profiles")
    ax.axhline(0,color='grey',lw=.8);ax.set_xticks(frame.year,frame.year.astype(str),rotation=45)
    ax.set_ylabel('Estimated marginal response\n(outcome points / log immigration)')
    ax.set_title(b.OUTCOMES[out.name]+' — neural outcome function\nAdditive commune effects; no confidence intervals',fontsize=11)
    ax.legend(fontsize=8);fig.savefig(out/'time_heterogeneity.png',dpi=150);plt.close(fig)
    return {'profile_count':len(profiles),'estimates':rows,
            'reference':'same empirical common-support profiles as the no-penalty kernel run',
            'initialization_ranges_are_not_confidence_intervals':True}


def fit_final(d,reference,selected,seed,nodes,out,reference_directory):
    params=density_params(reference['density_bandwidth_selection'])
    parts=k.crossfit_density(d,k.folds(d,seed),params)
    grid,weights=integration_rule([m for _,m in parts],nodes)
    q,mass_error=combined_quadrature(parts,len(d),grid,weights)
    check_grid,check_weights=integration_rule([m for _,m in parts],2*nodes)
    qcheck,check_mass_error=combined_quadrature(parts,len(d),check_grid,check_weights)
    s=b.inputs(d)[0];models=[];means=[];values=[];derivatives=[];moments=[];fits=[]
    p=b.Absorb(d);scale_y=max(float(np.std(p.project(d.y.to_numpy(float)))),1e-10)
    for init in FINAL_INITIALIZATIONS:
        model,history=train_network(d,grid,q,selected[0],init,selected[1])
        value,derivative=evaluate(model,s)
        moment=integrate_saved(model,s,grid,q)
        checked=integrate_saved(model,s,check_grid,qcheck)
        quadrature_error=float(np.sqrt(np.mean((moment-checked)**2)))
        if quadrature_error>1e-3*scale_y:
            raise ValueError(f'Network quadrature error {quadrature_error:g} exceeds tolerance; increase --nodes')
        mean=float(derivative[d.trim_keep].mean())
        models.append(model);means.append(mean);values.append(value);derivatives.append(derivative);moments.append(moment)
        fits.append({'initialization':init,'average_derivative':mean,'training_integrated_MSE':history[-1]['training_integrated_MSE'],
                     'quadrature_prediction_RMSE_versus_double_nodes':quadrature_error,
                     'parameter_count':sum(model[f'weight_{j}'].size+model[f'bias_{j}'].size for j in range(len(model['hidden'])+1))})
        print(f'  final initialization {init}: derivative {mean:.6g}',flush=True)
    payload={f'model_{j}_{key}':value for j,model in enumerate(models) for key,value in model.items()}
    np.savez_compressed(out/'models_primary.npz',**payload)
    g=np.mean(values,axis=0);derivative=np.mean(derivatives,axis=0);moment=np.mean(moments,axis=0)
    # Recover the same nuisance coefficients profiled from the training
    # objective: its fitted response is D beta + E[g(A,X,t)|Z,X,t].
    nuisance=linalg.lstsq(p.D,d.y.to_numpy(float)-moment)[0]
    cells=d.copy();cells['g_component']=g;cells['marginal_effect']=derivative
    cells['initialization_min_not_CI']=np.min(derivatives,axis=0);cells['initialization_max_not_CI']=np.max(derivatives,axis=0)
    cells['commune_fixed_effect_component']=p.D[:,:p.n_communes]@nuisance[:p.n_communes]
    cells['year_main_effect_inside_f']=p.D[:,p.n_communes:]@nuisance[p.n_communes:]
    cells['f_component']=g+cells.year_main_effect_inside_f
    cells['structural_fitted']=cells.f_component+cells.commune_fixed_effect_component
    cells['integrated_g_component']=moment
    cells['integrated_moment_fitted']=p.D@nuisance+moment
    cells.to_csv(out/'cells_primary.csv',index=False)
    if not np.isfinite(cells.select_dtypes(include=[np.number]).to_numpy()).all():raise FloatingPointError('Nonfinite saved results')
    return {'average_derivative_trimmed':float(derivative[d.trim_keep].mean()),
            'average_derivative_all_cells':float(derivative.mean()),'initialization_range_not_CI':[min(means),max(means)],
            'initialization_fits':fits,'ensemble_training_integrated_MSE':float(np.mean(p.project(d.y.to_numpy()-moment)**2)),
            'quadrature_nodes':nodes,'quadrature_check_nodes':2*nodes,
            'quadrature_mass_error':max(mass_error,check_mass_error),
            'time_heterogeneity':time_results(d,models,out,reference_directory)},cells


def self_checks():
    rng=np.random.default_rng(713);torch.manual_seed(713)
    target=np.column_stack([rng.normal(size=(30,3)),np.tile(np.arange(2008,2013),6)])
    scale={'mean':np.zeros(3),'sd':np.array([1.2,.8,1.1]),'years':np.arange(2008,2013)}
    network=OutcomeNetwork(7,(8,))
    with torch.no_grad():network.layers[-1].weight.normal_(std=.2)
    model=export_model(network,scale,2.,(8,),0,0)
    features=np.column_stack([target[:,0]/scale['sd'][0],context(target,scale)])
    tensor=torch.tensor(features,dtype=DTYPE,requires_grad=True)
    values=network.points(tensor)*2
    deriv=torch.autograd.grad(values.sum(),tensor)[0][:,0]/scale['sd'][0]
    actual,gradient=evaluate(model,target)
    assert np.allclose(values.detach().numpy(),actual,atol=1e-12)
    assert np.allclose(deriv.detach().numpy(),gradient,atol=1e-12)
    w=rng.random((30,7));w/=w.sum(axis=1,keepdims=True)
    density={'donor_A':rng.normal(size=7),'weights':w,'density_A_bandwidth':.6}
    grid,weights=integration_rule([density],64);q,error=density_quadrature(density,grid,weights)
    assert np.allclose(q@grid,w@density['donor_A'],atol=1e-10)
    assert np.allclose(q@(grid**2),w@(density['donor_A']**2+.6**2),atol=1e-10)
    ctx=torch.tensor(context(target,scale),dtype=DTYPE)
    aq=torch.tensor(grid/scale['sd'][0],dtype=DTYPE);qt=torch.tensor(q,dtype=DTYPE)
    expected_value=(network.grid(aq,ctx)*qt).sum(dim=1)*2
    assert np.allclose(expected_value.detach().numpy(),integrate_saved(model,target,grid,q),atol=1e-12)
    # Different observed A cannot enter the integrated prediction: it is
    # integrated out, while demographics and year remain fixed.
    altered=target.copy();altered[:,0]+=100
    assert np.allclose(integrate_saved(model,target,grid,q),integrate_saved(model,altered,grid,q))
    panel=pd.DataFrame({'cod_com':np.repeat(np.arange(6),5),'year':np.tile(np.arange(2008,2013),6),'respondents':100})
    Q=torch.tensor(b.Absorb(panel).Q,dtype=DTYPE);yy=torch.tensor(rng.normal(size=30),dtype=DTYPE)
    def loss():
        pred=(network.grid(aq,ctx)*qt).sum(dim=1)
        res=yy-pred;res=res-Q@(Q.T@res)
        return torch.mean(res**2)
    network.zero_grad();loss().backward();analytic=network.layers[0].weight.grad[0,0].item()
    with torch.no_grad():
        old=network.layers[0].weight[0,0].item();eps=1e-6
        network.layers[0].weight[0,0]=old+eps;plus=loss().item()
        network.layers[0].weight[0,0]=old-eps;minus=loss().item()
        network.layers[0].weight[0,0]=old
    assert np.isclose(analytic,(plus-minus)/(2*eps),atol=1e-8)
    return {'quadrature_recovers_Gaussian_mixture_mean_and_variance':True,
            'serialized_network_matches_torch_values_and_derivatives':True,
            'integrated_loss_gradient_matches_finite_difference':True,
            'integrated_prediction_does_not_use_target_observed_treatment':True,
            'saved_and_torch_integration_match':True,'weight_decay_and_dropout_are_zero':True}


def report(results,out,reference_directory,python):
    lines=['HYBRID NPIV: KERNEL DENSITY FIRST STAGE + NEURAL OUTCOME FUNCTION','',
        'Y_mt = alpha_m + b_t + g(A_mt, mean_age_mt, male_share_mt, year_indicators) + u_mt.',
        'Commune intercepts and year main effects are unpenalized and profiled from the integrated loss.',
        'Commune identity is not a network input. Immigration/demographic/year interactions are learned by a smooth tanh network.',
        'The kernel density estimator, 11 raw shares, bandwidths and commune folds match the no_penalty comparison.',
        'Fold-specific first-stage bandwidths were originally tuned only on each structural training fold; they are reused without retuning here.',
        'Two architectures are considered: one hidden layer with 8 units; two hidden layers with 16 and 8 units.',
        'Explicit L2 weight decay and dropout are ZERO. Architecture and stopping steps control complexity.',
        'Adam learning rate is 0.005. Network size and steps minimize the held-out integrated NPIV loss of a two-initialization ensemble.',
        'All folds are grouped by commune. Validation profiles out additive effects in the held-out panel; this is not outcome-level forecasting.',
        'The minimum CV score is a selection score, not an untouched-test-set performance estimate.',
        'Treatment is integrated with deterministic Gauss-Legendre quadrature over the KDE; it is not replaced by its conditional mean.',
        'Final fits use five initializations and are checked with twice as many quadrature nodes.',
        'Initialization ranges and plotted lines are sensitivity diagnostics, NOT confidence intervals. No causal inference calibration is supplied.',
        'This is a DeepIV-style neural outcome stage with a kernel density first stage, not full mixture-density-network DeepIV.',
        'The additive-effects panel adaptation still requires valid panel IV moments and identification of the response function.','']
    for name,r in results.items():
        p=r['primary'];cv=r['network_selection'];old=r['comparison']
        lines += [b.OUTCOMES[name]+f' ({name})',
            f"  Chosen hidden units: {cv['selected_hidden']}; Adam steps: {cv['selected_steps']}; maximum-step boundary: {cv['selected_maximum_steps']}",
            f"  Average derivative: {p['average_derivative_trimmed']:.6g}; initialization range (not CI): {p['initialization_range_not_CI']}",
            f"  Held-out integrated MSE: neural {cv['selected_heldout_integrated_MSE']:.6g}; FE-only {cv['FE_only_heldout_MSE']:.6g}; no-penalty kernel {old['kernel_heldout_integrated_MSE']:.6g}",
            f"  Improvement over FE-only: {cv['heldout_loss_improvement_over_FE_only']:.6g}",
            f"  Ensemble training integrated MSE: {p['ensemble_training_integrated_MSE']:.6g}",
            f"  Common time-comparison profiles: {p['time_heterogeneity']['profile_count']}",
            f"  Largest quadrature-prediction RMSE on doubling nodes: {max(f['quadrature_prediction_RMSE_versus_double_nodes'] for f in p['initialization_fits']):.6g}",'']
    lines += [f'Kernel reference: {reference_directory}',
        'Files: results.json, summary.csv, OUTCOME/network_cv.csv, OUTCOME/cv_predictions.npz, OUTCOME/models_primary.npz, OUTCOME/cells_primary.csv, OUTCOME/time_heterogeneity.csv and .png.',
        f'Run with a Python environment containing torch: {python} project/npiv/kernel/run_density_neural.py','']
    (out/'report.txt').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--outcomes',nargs='+',default=['pc_19','pc_20','vict_agreg'],choices=list(b.OUTCOMES))
    parser.add_argument('--output',type=Path,default=HERE/'neural_outcome')
    parser.add_argument('--reference',type=Path,default=HERE/'no_penalty')
    parser.add_argument('--nodes',type=int,default=64)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);start=time.time()
    # Fail before training if the environment cannot produce required figures.
    import matplotlib
    matplotlib.use('Agg')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):
        b.save_json(out/'self_checks.json',self_checks());print('Numerical checks passed',flush=True)
        if args.check_only:return
        reference=json.loads((args.reference/'results.json').read_text())
        assert reference['source_sha256']==hashlib.sha256(b.DATA.read_bytes()).hexdigest()
        seed=reference['seed'];results={};summary=[]
        for outcome in args.outcomes:
            print(outcome,flush=True);od=out/outcome;od.mkdir(exist_ok=True)
            d,metadata=b.load_cells(outcome,'raw_shares');old=reference['outcomes'][outcome]
            selected,selection=tune_network(d,old,seed,args.nodes,od)
            b.save_json(od/'network_selection.json',selection)
            primary,cells=fit_final(d,old,selected,seed,args.nodes,od,args.reference)
            comparison={'kernel_average_derivative':old['primary']['average_derivative_trimmed'],
                        'kernel_heldout_integrated_MSE':old['structural_bandwidth_selection']['selected']['heldout_projected_integrated_MSE'],
                        'same_first_stage_and_validation_folds':True}
            results[outcome]={'data':metadata,'primary':primary,'network_selection':selection,'comparison':comparison,
                             'first_stage_diagnostics':old['first_stage_diagnostics'],
                             'density_bandwidth_selection':old['density_bandwidth_selection']}
            summary.append({'outcome':outcome,'average_derivative':primary['average_derivative_trimmed'],
                            'initialization_min_not_CI':primary['initialization_range_not_CI'][0],
                            'initialization_max_not_CI':primary['initialization_range_not_CI'][1],
                            'hidden_units':'x'.join(map(str,selected[0])),'steps':selected[1],
                            'neural_heldout_integrated_MSE':selection['selected_heldout_integrated_MSE'],
                            'FE_only_heldout_MSE':selection['FE_only_heldout_MSE'],
                            'kernel_heldout_integrated_MSE':comparison['kernel_heldout_integrated_MSE']})
            b.save_json(out/'results.json',{'method':'hybrid_kernel_density_neural_outcome_NPIV','specification':SPECIFICATION,
                'seed':seed,'quadrature_nodes':args.nodes,'elapsed_seconds':time.time()-start,
                'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'helper_sha256':{file:hashlib.sha256((HERE/file).read_bytes()).hexdigest() for file in ['run_kernel.py','run_density_no_penalty.py']},
                'reference_sha256':hashlib.sha256((args.reference/'results.json').read_bytes()).hexdigest(),
                'source_sha256':reference['source_sha256'],'environment':{'python':sys.executable,'torch':torch.__version__,'numpy':np.__version__},
                'outcomes':results})
            pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);report(results,out,args.reference,sys.executable)
        print(f'Saved {out}; elapsed {time.time()-start:.1f} seconds',flush=True)


if __name__=='__main__':main()
