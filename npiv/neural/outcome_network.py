"""Tanh structural response, analytic derivatives and common-profile time summaries."""
import numpy as np
import pandas as pd
import torch
from torch import nn
import panel as b

ARCHITECTURES=[(8,),(16,8)]
CHECKPOINTS=[0,50,100,200,400,600]
CV_INITIALIZATIONS=[17,43]
FINAL_INITIALIZATIONS=[101,211,307,401,503]
LEARNING_RATE=.005
DTYPE=torch.float64

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

def tensors(d,grid,q,scale,y_scale):
    p=b.Absorb(d)
    return {'context':torch.tensor(context(b.inputs(d)[0],scale),dtype=DTYPE),
            'a':torch.tensor((grid-scale['mean'][0])/scale['sd'][0],dtype=DTYPE),
            'weights':torch.tensor(q,dtype=DTYPE),'Q':torch.tensor(p.Q,dtype=DTYPE),
            'target':torch.tensor(p.transform(d.y.to_numpy(float))/y_scale,dtype=DTYPE)}

def projected_residual(moment,pack):
    return pack['target']-(moment-pack['Q']@(pack['Q'].T@moment))

def time_results(d,models,out,reference_directory):
    profiles=pd.read_csv(reference_directory/f'{out.name}.csv')
    # Fixed empirical reference profiles for the selected neural specification.
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
    ax.set_ylabel('Estimated marginal response\n(outcome points / log(1 + immigration rate))')
    ax.set_title(b.OUTCOMES[out.name]+' — neural outcome function\nAdditive commune effects; no confidence intervals',fontsize=11)
    ax.legend(fontsize=8);fig.savefig(out/'time_heterogeneity.png',dpi=150);plt.close(fig)
    return {'profile_count':len(profiles),'estimates':rows,
            'reference':'fixed empirical common-support profiles in reference_profiles/',
            'initialization_ranges_are_not_confidence_intervals':True}
