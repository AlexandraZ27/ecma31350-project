"""Panel DeepIV: mixture-density first stage and neural structural response.

Uses the fixed panel sample, raw origin shares and commune-grouped folds.
Both networks are selected by their held-out losses. No L2 or dropout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
import panel as b
import outcome_network as h
DTYPE=torch.float64
DENSITY_ARCHITECTURES=[(16,),(32,16)]
MIXTURES=[1,3,10]
DENSITY_STEPS=[0,50,100,200,400]
DENSITY_SEEDS=[127,229]
DENSITY_LR=.005
SIGMA_FLOOR=.05  # In training-standardized treatment units; not an L2 penalty.
SPECIFICATION=dict(b.SPECIFICATION,
    first_stage='Gaussian mixture-density neural network on 11 raw shares, demographics and year indicators',
    first_stage_selection='nested commune-held-out negative log density; architecture, mixture count and Adam steps',
    first_stage_scale_floor_in_training_A_sd=SIGMA_FLOOR,
    integration='adaptive rowwise Gauss-Legendre quadrature checked against componentwise Gauss-Hermite; density network frozen in stage two',
    coefficient_penalty=0.,dropout=0.)


def continuous_inputs(d,with_shares):
    v=b.inputs(d)[1]
    return v[:,:-1] if with_shares else v[:,-3:-1]


def density_scaling(d,with_shares):
    x=continuous_inputs(d,with_shares)
    return {'x_mean':x.mean(axis=0),'x_sd':np.maximum(x.std(axis=0),1e-10),
            'years':np.sort(d.year.unique()).astype(float),'with_shares':np.array(with_shares),
            'A_mean':np.array(d.A.mean()),'A_sd':np.array(max(d.A.std(ddof=0),1e-10)),
            'train_communes':np.sort(d.cod_com.unique())}


def density_inputs(d,scale):
    x=(continuous_inputs(d,bool(scale['with_shares']))-scale['x_mean'])/scale['x_sd']
    year=(d.year.to_numpy()[:,None]==scale['years'][None,:]).astype(float)
    assert np.all(year.sum(axis=1)==1)
    return np.column_stack([x,year])


class DensityNetwork(nn.Module):
    def __init__(self,dimension,hidden,components):
        super().__init__();self.components=components
        dims=[dimension,*hidden,3*components]
        self.layers=nn.ModuleList([nn.Linear(a,c,dtype=DTYPE) for a,c in zip(dims[:-1],dims[1:])])
        nn.init.zeros_(self.layers[-1].weight)
        with torch.no_grad():
            self.layers[-1].bias.zero_()
            means=np.zeros(1) if components==1 else np.linspace(-1.2,1.2,components)
            self.layers[-1].bias[components:2*components]=torch.tensor(means,dtype=DTYPE)
            sd=1. if components==1 else .6
            self.layers[-1].bias[2*components:]=np.log(np.expm1(sd-SIGMA_FLOOR))

    def forward(self,x):
        for layer in self.layers[:-1]:x=torch.tanh(layer(x))
        logits,mu,raw=self.layers[-1](x).split(self.components,dim=-1)
        return torch.log_softmax(logits,dim=-1),mu,F.softplus(raw)+SIGMA_FLOOR


def torch_nll(network,x,a):
    logpi,mu,sigma=network(x)
    lp=logpi-.5*((a[:,None]-mu)/sigma)**2-torch.log(sigma)-.5*np.log(2*np.pi)
    return -torch.logsumexp(lp,dim=1)


def export_density(network,scale,hidden,components,seed,steps):
    model={**scale,'hidden':np.array(hidden),'components':np.array(components),
           'seed':np.array(seed),'steps':np.array(steps),'sigma_floor':np.array(SIGMA_FLOOR)}
    for j,layer in enumerate(network.layers):
        model[f'weight_{j}']=layer.weight.detach().numpy().copy()
        model[f'bias_{j}']=layer.bias.detach().numpy().copy()
    return model


def density_predict(model,d):
    x=density_inputs(d,model)
    for j in range(len(model['hidden'])+1):
        x=x@model[f'weight_{j}'].T+model[f'bias_{j}']
        if j<len(model['hidden']):x=np.tanh(x)
    logits,mu,raw=np.split(x,3,axis=1)
    pi=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
    sigma=(np.logaddexp(0,raw)+float(model['sigma_floor']))*float(model['A_sd'])
    mu=mu*float(model['A_sd'])+float(model['A_mean'])
    assert np.isfinite(mu).all() and np.isfinite(sigma).all() and np.all(sigma>0)
    return {'pi':pi,'mu':mu,'sigma':sigma}


def ensemble_density(predictions):
    return {key:np.concatenate([p[key] for p in predictions],axis=1)/(len(predictions) if key=='pi' else 1)
            for key in ['pi','mu','sigma']}


def log_density(density,a):
    lp=np.log(np.maximum(density['pi'],np.finfo(float).tiny))
    lp-=.5*((a[:,None]-density['mu'])/density['sigma'])**2+np.log(density['sigma'])+.5*np.log(2*np.pi)
    return logsumexp(lp,axis=1)


def train_density(d,hidden,components,seed,steps,with_shares,target=None,record_steps=None):
    scale=density_scaling(d,with_shares);torch.manual_seed(seed)
    x=torch.tensor(density_inputs(d,scale),dtype=DTYPE)
    a=torch.tensor((d.A.to_numpy()-float(scale['A_mean']))/float(scale['A_sd']),dtype=DTYPE)
    network=DensityNetwork(x.shape[1],hidden,components)
    optimizer=torch.optim.Adam(network.parameters(),lr=DENSITY_LR,weight_decay=0.)
    checkpoints=set(record_steps or [steps]);records={}
    for step in range(steps+1):
        if step in checkpoints:
            model=export_density(network,scale,hidden,components,seed,step)
            with torch.no_grad():train_nll=float(torch_nll(network,x,a).mean())+np.log(float(scale['A_sd']))
            records[step]={'model':model,'training_NLL':train_nll,
                           'density':density_predict(model,target) if target is not None else None}
        if step==steps:break
        optimizer.zero_grad();loss=torch_nll(network,x,a).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite MDN loss')
        loss.backward();optimizer.step()
    return export_density(network,scale,hidden,components,seed,steps),records


def save_models(path,models):
    np.savez_compressed(path,**{f'model_{j}_{key}':value for j,model in enumerate(models) for key,value in model.items()})


def load_models(path):
    z=np.load(path)
    indices=sorted({int(key.split('_')[1]) for key in z.files})
    return [{key[len(f'model_{j}_'):]:z[key] for key in z.files if key.startswith(f'model_{j}_')} for j in indices]


def tune_density(d,split,with_shares):
    candidates=[(hidden,c,step) for hidden in DENSITY_ARCHITECTURES for c in MIXTURES for step in DENSITY_STEPS]
    losses={c:0. for c in candidates};predictions={c:[] for c in candidates};models={c:[] for c in candidates}
    folds=[]
    for fold,held in enumerate(split):
        train,target=d.loc[~held],d.loc[held]
        records=[]
        for hidden in DENSITY_ARCHITECTURES:
            for components in MIXTURES:
                histories=[train_density(train,hidden,components,seed,max(DENSITY_STEPS),with_shares,
                    target=target,record_steps=DENSITY_STEPS)[1] for seed in DENSITY_SEEDS]
                for step in DENSITY_STEPS:
                    key=(hidden,components,step)
                    prediction=ensemble_density([history[step]['density'] for history in histories])
                    nll=float(-log_density(prediction,target.A.to_numpy()).sum())
                    losses[key]+=nll;predictions[key].append((np.flatnonzero(held),prediction))
                    models[key].append([history[step]['model'] for history in histories])
                    records.append({'hidden':list(hidden),'components':components,'steps':step,
                                    'heldout_density_NLL':nll/len(target)})
        folds.append({'fold':fold,'train_communes':sorted(train.cod_com.unique().tolist()),
                      'test_communes':sorted(target.cod_com.unique().tolist()),'test_n':len(target),'grid':records})
    chosen=min(candidates,key=lambda c:losses[c])
    grid=[{'hidden':list(c[0]),'components':c[1],'steps':c[2],'heldout_density_NLL':losses[c]/len(d)} for c in candidates]
    selected=next(row for row in grid if (tuple(row['hidden']),row['components'],row['steps'])==chosen)
    combined={key:np.empty((len(d),len(DENSITY_SEEDS)*chosen[1])) for key in ['pi','mu','sigma']}
    for idx,pred in predictions[chosen]:
        for key in combined:combined[key][idx]=pred[key]
    meta={'selected':selected,'grid':grid,'folds':folds,'with_shares':with_shares,
          'seeds':DENSITY_SEEDS,'weight_decay':0.,'dropout':0.,'sigma_floor_in_training_A_sd':SIGMA_FLOOR,
          'selection_loss':'negative log of the ensemble density; entire communes held out',
          'selected_maximum_steps':chosen[2]==max(DENSITY_STEPS)}
    return chosen,meta,combined,models[chosen]


def fit_density_ensemble(train,target,selected,with_shares):
    hidden,components,steps=selected
    models=[train_density(train,hidden,components,seed,steps,with_shares)[0] for seed in DENSITY_SEEDS]
    return ensemble_density([density_predict(model,target) for model in models]),models


def hermite_quadrature(density,nodes):
    x,w=np.polynomial.hermite.hermgauss(nodes)
    a=density['mu'][:,:,None]+np.sqrt(2)*density['sigma'][:,:,None]*x[None,None,:]
    q=density['pi'][:,:,None]*w[None,None,:]/np.sqrt(np.pi)
    a=a.reshape(len(a),-1);q=q.reshape(len(q),-1)
    assert np.allclose(q.sum(axis=1),1,atol=1e-12)
    return a,q


def quadrature(density,nodes):
    """A short shared grid per row integrates all mixture components together.

    Expand resolution until total mass and the first two analytic moments agree.
    The eight-SD bounds leave negligible normal tails outside the integration.
    """
    lower=np.min(density['mu']-8*density['sigma'],axis=1)
    upper=np.max(density['mu']+8*density['sigma'],axis=1)
    mean=np.sum(density['pi']*density['mu'],axis=1)
    second=np.sum(density['pi']*(density['mu']**2+density['sigma']**2),axis=1)
    remaining=np.arange(len(lower));accepted=[]
    while nodes<=2048 and len(remaining):
        lo,hi=lower[remaining],upper[remaining]
        pi,mu,sigma=[density[key][remaining] for key in ['pi','mu','sigma']]
        x,w=np.polynomial.legendre.leggauss(nodes)
        a=lo[:,None]+(x[None,:]+1)*(hi-lo)[:,None]/2
        weights=w[None,:]*(hi-lo)[:,None]/2
        pdf=np.exp(-.5*((a[:,None,:]-mu[:,:,None])/sigma[:,:,None])**2)
        pdf/=np.sqrt(2*np.pi)*sigma[:,:,None]
        q=np.sum(pi[:,:,None]*pdf,axis=1)*weights
        good=((np.abs(q.sum(axis=1)-1)<1e-9)
              &(np.abs(np.sum(q*a,axis=1)-mean[remaining])/(1+np.abs(mean[remaining]))<1e-9)
              &(np.abs(np.sum(q*a*a,axis=1)-second[remaining])/(1+second[remaining])<1e-9))
        if np.any(good):accepted.append((remaining[good],a[good],q[good]/q[good].sum(axis=1,keepdims=True)))
        remaining=remaining[~good]
        nodes*=2
    if len(remaining):raise ValueError('Mixture quadrature failed mass/moment accuracy checks')
    size=max(part[1].shape[1] for part in accepted)
    all_a=np.zeros((len(lower),size));all_q=np.zeros_like(all_a)
    for idx,a,q in accepted:all_a[idx,:a.shape[1]]=a;all_q[idx,:q.shape[1]]=q
    return all_a,all_q


def integration_groups(q,batch_size=128):
    lengths=np.max(np.where(q>0,np.arange(q.shape[1])[None,:]+1,0),axis=1)
    groups=[]
    for size in np.unique(lengths):
        idx=np.flatnonzero(lengths==size)
        groups.extend((idx[start:start+batch_size],int(size)) for start in range(0,len(idx),batch_size))
    return groups


def outcome_pack(d,a,q,scale,y_scale):
    pack=h.tensors(d,a,q,scale,y_scale);groups=[];order=[]
    for idx,size in integration_groups(q):
        groups.append((pack['a'][idx,:size],pack['context'][idx],pack['weights'][idx,:size]))
        order.extend(idx.tolist())
    pack['groups']=groups;pack['restore']=torch.tensor(np.argsort(order),dtype=torch.long)
    return pack


def expected_outcome(network,pack):
    # Each row keeps its own converged quadrature resolution. Evaluate in small
    # blocks to avoid allocating the largest grid for every observation.
    values=[(network.grid(a,ctx)*q).sum(dim=1) for a,ctx,q in pack['groups']]
    return torch.cat(values)[pack['restore']]


class IntegratedOutcomeNetwork(h.OutcomeNetwork):
    def grid(self,a,ctx):
        first=self.layers[0]
        value=torch.tanh((ctx@first.weight[:,1:].T+first.bias)[:,None,:]+a[:,:,None]*first.weight[:,0][None,None,:])
        for layer in self.layers[1:-1]:value=torch.tanh(layer(value))
        return self.layers[-1](value).squeeze(-1)


def integrate(model,s,a,q):
    result=np.empty(len(s))
    for idx,size in integration_groups(q):
        result[idx]=integrate_block(model,s[idx],a[idx,:size],q[idx,:size])
    return result


def integrate_block(model,s,a,q):
    ctx=h.context(s,model);w=model['weight_0']
    value=np.tanh((ctx@w[:,1:].T+model['bias_0'])[:,None,:]
                  +((a-model['mean'][0])/model['sd'][0])[:,:,None]*w[:,0][None,None,:])
    for j in range(1,len(model['hidden'])):value=np.tanh(value@model[f'weight_{j}'].T+model[f'bias_{j}'])
    last=len(model['hidden']);value=(value@model[f'weight_{last}'].T+model[f'bias_{last}'])[:,:,0]
    return np.sum(q*value,axis=1)*float(model['y_scale'])


def train_outcome(d,a,q,hidden,seed,steps,validation=None,record_steps=None):
    scale=h.scaling(d);p=b.Absorb(d);y_scale=max(float(np.std(p.project(d.y.to_numpy()))),1e-10)
    pack=outcome_pack(d,a,q,scale,y_scale)
    test=outcome_pack(*validation,scale,y_scale) if validation is not None else None
    torch.manual_seed(seed);network=IntegratedOutcomeNetwork(pack['context'].shape[1],hidden)
    optimizer=torch.optim.Adam(network.parameters(),lr=h.LEARNING_RATE,weight_decay=0.)
    records={};checkpoints=set(record_steps or [steps])
    for step in range(steps+1):
        if step in checkpoints:
            with torch.no_grad():
                mse=float(torch.mean(h.projected_residual(expected_outcome(network,pack),pack)**2))*y_scale**2
                row={'steps':step,'training_integrated_MSE':mse,
                     'model':h.export_model(network,scale,y_scale,hidden,seed,step)}
                if test is not None:row['prediction']=expected_outcome(network,test).numpy()*y_scale
                records[step]=row
        if step==steps:break
        optimizer.zero_grad();loss=torch.mean(h.projected_residual(expected_outcome(network,pack),pack)**2)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite integrated outcome loss')
        loss.backward();optimizer.step()
    return h.export_model(network,scale,y_scale,hidden,seed,steps),records


def density_diagnostics(d,pred,baseline_mean):
    mean=np.sum(pred['pi']*pred['mu'],axis=1)
    return {'density_NLL':float(-log_density(pred,d.A.to_numpy()).mean()),
            'mean_prediction_SSE':float(np.sum((d.A.to_numpy()-mean)**2)),
            'mean_baseline_SSE':float(np.sum((d.A.to_numpy()-baseline_mean)**2))}


def tune_pipeline(d,seed,nodes,out,reuse_density=False):
    candidates=[(hidden,steps) for hidden in h.ARCHITECTURES for steps in h.CHECKPOINTS]
    losses={c:0. for c in candidates};oof={c:np.empty(len(d)) for c in candidates}
    baseline_sse=0.;records=[];fsrecords=[]
    for fold,held in enumerate(b.folds(d,seed)):
        train,target=d.loc[~held].copy(),d.loc[held].copy();fd=out/f'fold_{fold}';fd.mkdir(exist_ok=True)
        inner=b.folds(train,seed+100+fold,3)
        densities={};fs={'fold':fold,'test_n':len(target),'train_communes':sorted(train.cod_com.unique().tolist()),
                       'test_communes':sorted(target.cod_com.unique().tolist())}
        for with_shares,label in [(True,'with_shares'),(False,'without_shares')]:
            selection_path=fd/f'density_{label}_selection.json'
            if reuse_density and selection_path.exists():
                tuning=json.loads(selection_path.read_text());chosen=tuning['selected']
                selected=(tuple(chosen['hidden']),chosen['components'],chosen['steps'])
                assert selected[0] in DENSITY_ARCHITECTURES and selected[1] in MIXTURES and selected[2] in DENSITY_STEPS
                models=load_models(fd/f'density_{label}_outer.npz')
                for model in models:
                    for key,value in density_scaling(train,with_shares).items():assert np.allclose(model[key],value)
                prediction=ensemble_density([density_predict(model,target) for model in models])
                inner_models=[];crossfit={key:np.empty((len(train),prediction[key].shape[1])) for key in prediction}
                for j,inside in enumerate(inner):
                    subset=load_models(fd/f'density_{label}_inner_{j}.npz');inner_models.append(subset)
                    for model in subset:
                        for key,value in density_scaling(train.loc[~inside],with_shares).items():assert np.allclose(model[key],value)
                    pred=ensemble_density([density_predict(model,train.loc[inside]) for model in subset])
                    for key in pred:crossfit[key][inside]=pred[key]
            else:
                selected,tuning,crossfit,inner_models=tune_density(train,inner,with_shares)
                prediction,models=fit_density_ensemble(train,target,selected,with_shares)
            save_models(fd/f'density_{label}_outer.npz',models)
            for j,subset in enumerate(inner_models):save_models(fd/f'density_{label}_inner_{j}.npz',subset)
            b.save_json(fd/f'density_{label}_selection.json',tuning)
            densities[label]=prediction
            fs[label]=dict(density_diagnostics(target,prediction,train.A.mean()),selection=tuning)
            if with_shares:training_density=crossfit
            print(f'  fold {fold+1}/5 MDN {label}: {selected}, held-out NLL {fs[label]["density_NLL"]:.5f}',flush=True)
        np.savez_compressed(fd/'density_predictions.npz',train_indices=np.flatnonzero(~held),test_indices=np.flatnonzero(held),
            **{f'train_{key}':value for key,value in training_density.items()},
            **{f'test_{label}_{key}':value for label,pred in densities.items() for key,value in pred.items()})
        a,q=quadrature(training_density,nodes);at,qt=quadrature(densities['with_shares'],nodes)
        fs['quadrature_nodes']={'training':a.shape[1],'validation':at.shape[1]}
        p=b.Absorb(target);y=target.y.to_numpy();baseline_sse+=float(np.sum(p.project(y)**2))
        fold_rows=[];checkpoint_models=[]
        for hidden in h.ARCHITECTURES:
            histories=[train_outcome(train,a,q,hidden,init,max(h.CHECKPOINTS),
                validation=(target,at,qt),record_steps=h.CHECKPOINTS)[1] for init in h.CV_INITIALIZATIONS]
            for steps in h.CHECKPOINTS:
                pred=np.mean([history[steps]['prediction'] for history in histories],axis=0)
                sse=float(np.sum(p.project(y-pred)**2));losses[(hidden,steps)]+=sse;oof[(hidden,steps)][held]=pred
                fold_rows.append({'hidden':list(hidden),'steps':steps,'heldout_integrated_MSE':sse/len(target)})
                checkpoint_models.extend([history[steps]['model'] for history in histories])
        save_models(fd/'outcome_cv_models.npz',checkpoint_models)
        fsrecords.append(fs);records.append({'fold':fold,'train_communes':fs['train_communes'],
            'test_communes':fs['test_communes'],'test_n':len(target),'grid':fold_rows})
        print(f'  fold {fold+1}/5 outcome validation complete',flush=True)
    chosen=min(candidates,key=lambda c:losses[c])
    grid=[{'hidden':list(hidden),'steps':steps,'heldout_integrated_MSE':losses[(hidden,steps)]/len(d)} for hidden,steps in candidates]
    selection={'selected_hidden':list(chosen[0]),'selected_steps':chosen[1],
        'selected_heldout_integrated_MSE':losses[chosen]/len(d),'FE_only_heldout_MSE':baseline_sse/len(d),
        'grid':grid,'folds':records,'weight_decay':0.,'dropout':0.,'cv_initializations':h.CV_INITIALIZATIONS,
        'selected_maximum_steps':chosen[1]==max(h.CHECKPOINTS),
        'selection_note':'Minimum CV score selects the network; it is not an untouched test-set score.'}
    fs={'folds':fsrecords,'nested_tuning':True}
    for label in ['with_shares','without_shares']:
        fs[label]={'density_NLL':sum(row['test_n']*row[label]['density_NLL'] for row in fsrecords)/len(d),
            'mean_prediction_R2':1-sum(row[label]['mean_prediction_SSE'] for row in fsrecords)/sum(row[label]['mean_baseline_SSE'] for row in fsrecords)}
    fs['density_loss_improvement']=fs['without_shares']['density_NLL']-fs['with_shares']['density_NLL']
    fs['folds_with_density_loss_improvement']=sum(row['with_shares']['density_NLL']<row['without_shares']['density_NLL'] for row in fsrecords)
    np.savez_compressed(out/'cv_predictions.npz',**{f"hidden_{'_'.join(map(str,c[0]))}_steps_{c[1]}":v for c,v in oof.items()})
    pd.DataFrame(grid).assign(hidden=lambda x:x.hidden.map(lambda v:'x'.join(map(str,v)))).to_csv(out/'network_cv.csv',index=False)
    b.save_json(out/'network_selection.json',selection);b.save_json(out/'first_stage_diagnostics.json',fs)
    return chosen,selection,fs


def fit_final(d,selected,seed,nodes,out,reference):
    config,tuning,density,fold_models=tune_density(d,b.folds(d,seed),True)
    b.save_json(out/'density_selection.json',tuning)
    for j,models in enumerate(fold_models):save_models(out/f'density_crossfit_{j}.npz',models)
    _,full_models=fit_density_ensemble(d,d,config,True)
    save_models(out/'density_full_models.npz',full_models)
    np.savez_compressed(out/'density_crossfit_predictions.npz',**density)
    a,q=quadrature(density,nodes);a2,q2=hermite_quadrature(density,64)
    s=b.inputs(d)[0];p=b.Absorb(d);scale_y=max(float(np.std(p.project(d.y.to_numpy()))),1e-10)
    models=[];values=[];derivatives=[];moments=[];fits=[]
    for init in h.FINAL_INITIALIZATIONS:
        model,history=train_outcome(d,a,q,selected[0],init,selected[1])
        value,derivative=h.evaluate(model,s);moment=integrate(model,s,a,q)
        error=float(np.sqrt(np.mean((moment-integrate(model,s,a2,q2))**2)))
        if error>1e-3*scale_y:raise ValueError(f'Quadrature error {error:g}; increase --nodes')
        models.append(model);values.append(value);derivatives.append(derivative);moments.append(moment)
        fits.append({'initialization':init,'average_derivative':float(derivative[d.trim_keep].mean()),
                     'training_integrated_MSE':history[selected[1]]['training_integrated_MSE'],
                     'quadrature_RMSE_versus_Hermite64':error})
        print(f'  final outcome initialization {init}: derivative {fits[-1]["average_derivative"]:.6g}',flush=True)
    save_models(out/'models_primary.npz',models)
    g=np.mean(values,axis=0);derivative=np.mean(derivatives,axis=0);moment=np.mean(moments,axis=0)
    nuisance=np.linalg.lstsq(p.D,d.y.to_numpy()-moment,rcond=None)[0]
    cells=d.copy();cells['g_component']=g;cells['marginal_effect']=derivative
    cells['initialization_min_not_CI']=np.min(derivatives,axis=0);cells['initialization_max_not_CI']=np.max(derivatives,axis=0)
    cells['commune_fixed_effect_component']=p.D[:,:p.n_communes]@nuisance[:p.n_communes]
    cells['year_main_effect_inside_f']=p.D[:,p.n_communes:]@nuisance[p.n_communes:]
    cells['f_component']=g+cells.year_main_effect_inside_f
    cells['structural_fitted']=cells.f_component+cells.commune_fixed_effect_component
    cells['integrated_g_component']=moment;cells['integrated_moment_fitted']=p.D@nuisance+moment
    cells.to_csv(out/'cells_primary.csv',index=False)
    assert np.isfinite(cells.select_dtypes(include=[np.number]).to_numpy()).all()
    means=[row['average_derivative'] for row in fits]
    primary={'average_derivative_trimmed':float(derivative[d.trim_keep].mean()),
        'average_derivative_all_cells':float(derivative.mean()),'initialization_range_not_CI':[min(means),max(means)],
        'initialization_fits':fits,'ensemble_training_integrated_MSE':float(np.mean(p.project(d.y.to_numpy()-moment)**2)),
        'quadrature_nodes':a.shape[1],'quadrature_check':'64 Gauss-Hermite nodes per component',
        'time_heterogeneity':h.time_results(d,models,out,reference)}
    return primary,tuning


def self_checks():
    rng=np.random.default_rng(71);torch.manual_seed(71)
    network=DensityNetwork(4,(8,),3)
    with torch.no_grad():network.layers[-1].weight.normal_(std=.1)
    x=torch.tensor(rng.normal(size=(9,4)),dtype=DTYPE);a=torch.tensor(rng.normal(size=9),dtype=DTYPE)
    logpi,mu,sigma=network(x)
    dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=logpi),torch.distributions.Normal(mu,sigma))
    assert torch.allclose(torch_nll(network,x,a),-dist.log_prob(a),atol=1e-12)
    pred={'pi':logpi.detach().exp().numpy(),'mu':mu.detach().numpy(),'sigma':sigma.detach().numpy()}
    assert np.allclose(log_density(pred,a.numpy()),dist.log_prob(a).detach().numpy(),atol=1e-12)
    grid,q=quadrature(pred,32)
    assert np.allclose(np.sum(q*grid,axis=1),np.sum(pred['pi']*pred['mu'],axis=1),atol=1e-12)
    assert np.allclose(np.sum(q*grid**2,axis=1),np.sum(pred['pi']*(pred['mu']**2+pred['sigma']**2),axis=1),atol=1e-12)
    target=np.column_stack([rng.normal(size=(9,3)),np.tile([2008,2009,2010],3)])
    scale={'mean':np.zeros(3),'sd':np.ones(3),'years':np.arange(2008,2011)}
    net=IntegratedOutcomeNetwork(5,(8,))
    with torch.no_grad():net.layers[-1].weight.normal_(std=.3)
    model=h.export_model(net,scale,2.,(8,),1,0)
    ct=torch.tensor(h.context(target,scale),dtype=DTYPE);at=torch.tensor(grid,dtype=DTYPE);qt=torch.tensor(q,dtype=DTYPE)
    value=(net.grid(at,ct)*qt).sum(dim=1)*2
    assert np.allclose(value.detach().numpy(),integrate(model,target,grid,q),atol=1e-12)
    groups=[];order=[]
    for idx in [np.array([7,2,4]),np.array([8,0,3]),np.array([1,5,6])]:
        groups.append((at[idx],ct[idx],qt[idx]));order.extend(idx.tolist())
    pack={'groups':groups,'restore':torch.tensor(np.argsort(order),dtype=torch.long)}
    assert torch.allclose(expected_outcome(net,pack)*2,value,atol=1e-12)
    changed=target.copy();changed[:,0]+=100
    assert np.allclose(integrate(model,target,grid,q),integrate(model,changed,grid,q))
    y=torch.tensor(rng.normal(size=9),dtype=DTYPE)
    def loss():return ((y-expected_outcome(net,pack))**2).mean()
    net.zero_grad();loss().backward();analytic=net.layers[0].weight.grad[0,0].item()
    with torch.no_grad():
        old=net.layers[0].weight[0,0].item();eps=1e-6
        net.layers[0].weight[0,0]=old+eps;plus=loss().item()
        net.layers[0].weight[0,0]=old-eps;minus=loss().item();net.layers[0].weight[0,0]=old
    assert np.isclose(analytic,(plus-minus)/(2*eps),atol=1e-8)
    return {'MDN_log_loss_matches_torch_distribution':True,'numpy_and_torch_density_agree':True,
        'quadrature_recovers_mixture_moments':True,'numpy_and_torch_integrated_predictions_agree':True,
        'integration_does_not_use_observed_target_treatment':True,'integrated_gradient_matches_finite_difference':True,
        'grouped_integration_preserves_row_order_and_predictions':True}


def write_report(results,out):
    lines=['FULL NEURAL PANEL NPIV / DEEPIV','',
        'First stage: neural Gaussian mixture density on 11 raw country shares, demographics and year indicators.',
        'Second stage: Y_mt = alpha_m + b_t + g_theta(A_mt, age_mt, male_share_mt, year indicators) + u_mt.',
        'Commune fixed effects are additive and separate from both networks; year enters both networks, with unpenalized outcome year main effects.',
        'Both stages have zero explicit L2 penalty and dropout. Compact architecture and early stopping restrict complexity.',
        f'MDN hidden units {DENSITY_ARCHITECTURES}; mixture counts {MIXTURES}; steps {DENSITY_STEPS}; seeds {DENSITY_SEEDS}.',
        f'MDN component standard deviations are bounded below by {SIGMA_FLOOR} times training SD(A), to prevent variance collapse.',
        'MDN weights, means and standard deviations all depend on its inputs. Stage one minimizes negative log density.',
        'First-stage tuning is nested inside five commune outer folds. The no-shares density baseline is tuned separately.',
        'Within each structural training set, first-stage predictions are cross-fitted by commune. All scaling uses only the relevant training sample.',
        'Outcome architecture and training duration minimize held-out integrated squared loss on the prespecified outer commune folds.',
        'Held-out commune/year effects are profiled for within-panel moment validation; these scores do not measure forecasting of new-commune levels.',
        'The minimum structural CV score is used for selection and is not an untouched-test-set assessment.',
        'Treatment is integrated on adaptive rowwise Gauss-Legendre grids, checked against componentwise Gauss-Hermite integration; the density network is frozen in stage two.',
        'The outcome objective squares the integrated residual; it does not regress the outcome on individual treatment draws.',
        'Final reported derivatives average five outcome-network initializations, holding the cross-fitted first-stage ensemble fixed.',
        'Initialization ranges are optimization diagnostics, not confidence intervals, and do not assess first-stage uncertainty.',
        'Common-profile time comparisons use fixed empirical overlap profiles in reference_profiles/.',
        'This is a panel adaptation of the two-neural-stage DeepIV method, with deterministic integration and the user-requested zero explicit penalties.',
        'Causal interpretation still requires the panel IV moment and identification assumptions. Conditional bootstrap inference is reported separately in inference/.','']
    for name,r in results.items():
        fs=r['first_stage_diagnostics'];p=r['primary'];cv=r['network_selection']
        lines += [b.OUTCOMES[name]+f' ({name})',
            f"  Final MDN selection: {r['density_selection']['selected']}",
            f"  Nested held-out first-stage density NLL: with shares {fs['with_shares']['density_NLL']:.6g}; without shares {fs['without_shares']['density_NLL']:.6g}; improvement {fs['density_loss_improvement']:.6g}.",
            f"  Treatment mean prediction R2: with shares {fs['with_shares']['mean_prediction_R2']:.6g}; without shares {fs['without_shares']['mean_prediction_R2']:.6g} (levels, not within-FE partial R2).",
            f"  Outcome hidden units {cv['selected_hidden']}; steps {cv['selected_steps']}.",
            f"  Held-out integrated MSE: full neural {cv['selected_heldout_integrated_MSE']:.6g}; FE-only {cv['FE_only_heldout_MSE']:.6g}.",
            f"  Average derivative {p['average_derivative_trimmed']:.6g}; initialization range (not CI) {p['initialization_range_not_CI']}.",
            f"  Training integrated MSE {p['ensemble_training_integrated_MSE']:.6g}.",'']
    lines += ['Run: python project/npiv/neural/run_full_neural.py',
        'Dependencies: project/npiv/neural/requirements-neural.txt',
        'Reference: Hartford et al. (2017), https://proceedings.mlr.press/v70/hartford17a.html.','']
    (out/'report.txt').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--outcomes',nargs='+',default=['pc_19','pc_20','vict_agreg'],choices=list(b.OUTCOMES))
    parser.add_argument('--output',type=Path,default=HERE/'full_neural')
    parser.add_argument('--nodes',type=int,default=64)
    parser.add_argument('--reuse-density',action='store_true',help='Reuse saved outer/inner density networks after checking training scales and commune IDs')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);started=time.time()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):
        b.save_json(out/'self_checks.json',self_checks());print('Full-neural numerical checks passed',flush=True)
        if args.check_only:return
        reference=json.loads((HERE/'reference_design.json').read_text())
        assert reference['source_sha256']==hashlib.sha256(b.DATA.read_bytes()).hexdigest()
        seed=reference['seed'];results={};summary=[]
        for outcome in args.outcomes:
            print(outcome,flush=True);od=out/outcome;od.mkdir(exist_ok=True)
            d,metadata=b.load_cells(outcome,'raw_shares')
            selected,selection,diagnostics=tune_pipeline(d,seed,args.nodes,od,args.reuse_density)
            primary,density_selection=fit_final(d,selected,seed,args.nodes,od,HERE/'reference_profiles')
            results[outcome]={'data':metadata,'primary':primary,'network_selection':selection,
                'first_stage_diagnostics':diagnostics,'density_selection':density_selection}
            summary.append({'outcome':outcome,'MDN_density_NLL':diagnostics['with_shares']['density_NLL'],
                'MDN_no_shares_density_NLL':diagnostics['without_shares']['density_NLL'],
                'full_neural_integrated_MSE':selection['selected_heldout_integrated_MSE'],
                'FE_only_integrated_MSE':selection['FE_only_heldout_MSE'],
                'average_derivative':primary['average_derivative_trimmed'],
                'initialization_min_not_CI':primary['initialization_range_not_CI'][0],
                'initialization_max_not_CI':primary['initialization_range_not_CI'][1]})
            b.save_json(out/'results.json',{'method':'full_neural_panel_DeepIV','specification':SPECIFICATION,
                'seed':seed,'quadrature_start_nodes':args.nodes,'elapsed_seconds':time.time()-started,
                'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'helper_sha256':{file:hashlib.sha256((HERE/file).read_bytes()).hexdigest() for file in ['panel.py','outcome_network.py']},
                'source_sha256':reference['source_sha256'],
                'reference_sha256':hashlib.sha256((HERE/'reference_design.json').read_bytes()).hexdigest(),
                'environment':{'python':sys.executable,'torch':torch.__version__,'numpy':np.__version__},'outcomes':results})
            pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);write_report(results,out)
        print(f'Saved {out}; elapsed {time.time()-started:.1f} seconds',flush=True)


if __name__=='__main__':main()
