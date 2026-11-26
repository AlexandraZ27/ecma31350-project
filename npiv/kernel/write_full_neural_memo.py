"""Rebuild the full-neural memo without changing the original NPIV exposition."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=HERE/'full_neural'
WORK=SOURCE/'memo_work'
LABELS={'pc_19':'Crime concerns','pc_20':'Preventive behavior','vict_agreg':'Victimization'}


def table(caption,label,head,rows,numeric_columns,notes=''):
    body='\n'.join(' & '.join(row)+r'\\' for row in rows)
    return (r'\begin{table}[H]'+'\n'+r'\centering'+'\n'+
            '\\caption{'+caption+'}\n\\label{'+label+'}\n\\small\n'+
            '\\begin{tabularx}{\\linewidth}{@{}X'+'r'*numeric_columns+'@{}}\n'+
            '\\toprule\n'+' & '.join(head)+r'\\'+'\n\\midrule\n'+body+
            '\n\\bottomrule\n\\end{tabularx}\n'+
            (r'\smallskip'+'\n'+r'\parbox{\linewidth}{\footnotesize '+notes+'}\n' if notes else '')+
            '\\end{table}')


def diagnostics(results):
    fs=[results['outcomes'][o]['first_stage_diagnostics'] if 'first_stage_diagnostics' in results['outcomes'][o]
        else json.loads((SOURCE/o/'first_stage_diagnostics.json').read_text()) for o in LABELS]
    rows=[]
    for text,kind,key in [('$L_1$, with shares','with_shares','density_NLL'),
                          ('$L_1$, no shares','without_shares','density_NLL')]:
        rows.append([text]+[f'{s[kind][key]:.3f}' for s in fs])
    rows.append([r'$\Delta L_1$ (gain)']+[f'{s["density_loss_improvement"]:.3f}' for s in fs])
    for text,kind in [(r'$R^2_{\mathrm{mean}}$, with shares','with_shares'),
                      (r'$R^2_{\mathrm{mean}}$, no shares','without_shares')]:
        rows.append([text]+[f'{s[kind]["mean_prediction_R2"]:.3f}' for s in fs])
    rows.append(['Folds with density-loss gain']+[f'{s["folds_with_density_loss_improvement"]}/5' for s in fs])
    return table('Nested commune-held-out first-stage performance','tab:neural-first-stage',
                 ['Diagnostic','Crime concerns','Preventive','Victimization'],rows,3,
                 'Scores pool all 1,010 held-out cells. Densities with and without shares are tuned separately inside each outer training fold. $L_1$ uses natural logarithms and the original treatment units.')


def results_table(results,inf,draft):
    rows=[]
    for name,label in LABELS.items():
        p=results['outcomes'][name]['primary']['average_derivative_trimmed']
        if draft: rows.append([label,f'{p:.3f}','Pending','Pending','Pending'])
        else:
            r=inf['outcomes'][name]
            rows.append([label,f'{p:.3f}',f'{r["bootstrap_SE"]:.3f}',f'{r["centered_bootstrap_p"]:.4f}',
                         f'[{r["symmetric_CI_low"]:.3f}, {r["symmetric_CI_high"]:.3f}]'])
    return table('Average derivatives and conditional bootstrap inference','tab:neural-effects',
                 ['Outcome','Estimate','SE','$p$','95\% interval'],rows,4,
                 'Equal cell weights; 910 trimmed cells in the original sample. SEs resample commune histories and refit both stages. Two-sided centered-bootstrap $p$-values test a zero derivative; symmetric intervals invert the same absolute-error test. All are conditional approximations described below.')


def time_inference(results,B,draft):
    figure_dir=SOURCE/'inference';figure_dir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,1,figsize=(6.5,6.5),sharex=True,layout='constrained')
    contrasts=[];contrast_rows=[]
    colors=['#164b76','#17796d','#a05d20']
    for ax,(name,label),color in zip(axes,LABELS.items(),colors):
        entries=results['outcomes'][name]['primary']['time_heterogeneity']['estimates']
        years=np.array([e['year'] for e in entries]);point=np.array([e['common_profile_average_derivative'] for e in entries])
        delta=float(point[-1]-point[0])
        if not draft:
            records=[json.loads((figure_dir/'draws'/f'{name}_{b:04}.json').read_text()) for b in range(B)]
            draws=np.array([r['result']['time_common_profile_derivative'] for r in records])
            errors=draws-point[None,:]
            critical=np.sort(np.abs(errors),axis=0)[min(B-1,int(np.ceil(.95*(B+1)))-1)]
            lo=point-critical;hi=point+critical
            ax.fill_between(years,lo,hi,color=color,alpha=.17,linewidth=0,label='Pointwise 95% bootstrap interval')
            ds=draws[:,-1]-draws[:,0];se=float(ds.std(ddof=1))
            p=float((1+np.count_nonzero(np.abs(ds-delta)>=abs(delta)))/(B+1))
            low,high=delta-np.quantile(ds-delta,[.975,.025])
            c_delta=float(np.sort(np.abs(ds-delta))[min(B-1,int(np.ceil(.95*(B+1)))-1)])
            contrasts.append({'outcome':name,'contrast':'2017 minus 2008, common profiles',
                              'estimate':delta,'bootstrap_SE':se,'centered_bootstrap_p':p,
                              'basic_CI_low':float(low),'basic_CI_high':float(high),
                              'symmetric_CI_low':float(delta-c_delta),'symmetric_CI_high':float(delta+c_delta),'replicates':B})
            contrast_rows.append([label,f'{delta:.3f}',f'{se:.3f}',f'{p:.4f}'])
        else:contrast_rows.append([label,f'{delta:.3f}','Pending','Pending'])
        ax.plot(years,point,color=color,marker='o',markersize=3.8,linewidth=1.8,label='Estimated derivative')
        ax.axhline(0,color='#666666',linewidth=.7,linestyle='--')
        ax.set_title(label,loc='left',fontsize=10,fontweight='bold',pad=5)
        ax.set_ylabel('Points / log point',fontsize=9)
        ax.grid(axis='y',color='#dddddd',linewidth=.5)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].legend(loc='best',fontsize=7,frameon=False)
    axes[-1].set_xticks(np.arange(2008,2018));axes[-1].set_xlabel('Year',fontsize=9)
    suffix='_draft' if draft else ''
    fig.savefig(figure_dir/f'time_profiles_inference{suffix}.pdf')
    fig.savefig(figure_dir/f'time_profiles_inference{suffix}.png',dpi=180)
    plt.close(fig)
    if not draft:pd.DataFrame(contrasts).to_csv(figure_dir/'time_endpoint_contrasts.csv',index=False)
    figure=(r'\begin{figure}[H]'+'\n'+r'\centering'+'\n'+
            '\\includegraphics[width=.92\\linewidth]{npiv/kernel/full_neural/inference/time_profiles_inference'+suffix+'.pdf}\n'+
            '\\caption{Time-specific average derivatives at common treatment and demographic profiles. Shaded bands are pointwise 95\\% symmetric centered-bootstrap intervals, conditional on fixed reference profiles and selected fit settings.}\n'+
            '\\label{fig:neural-time}\n\\end{figure}')
    return figure,table('Change in the common-profile derivative: 2017 minus 2008',
                        'tab:neural-time-contrast',['Outcome','Change','SE','$p$'],contrast_rows,3,
                        'The two-sided centered-bootstrap test uses the paired year difference in each draw. These are separate endpoint tests, without multiplicity adjustment.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');args=parser.parse_args()
    results=json.loads((SOURCE/'results.json').read_text())
    inf=None if args.draft else json.loads((SOURCE/'inference'/'results.json').read_text())
    B=399 if args.draft else inf['replicates']
    text=(WORK/'memo_body.tex').read_text()
    figure,contrast=time_inference(results,B,args.draft)
    replacements={'@@FIRST_STAGE_TABLE@@':diagnostics(results),'@@RESULTS_TABLE@@':results_table(results,inf,args.draft),
                  '@@TIME_FIGURE@@':figure,'@@TIME_CONTRAST_TABLE@@':contrast,'@@B@@':str(B),
                  '@@P_RESOLUTION@@':f'{1/(B+1):.4f}'}
    if args.draft: replacements['@@BOOTSTRAP_SHIFT@@']='The centering diagnostics will be inserted when the bootstrap is complete.'
    else:
        shifts=[inf['outcomes'][o]['bootstrap_bias_diagnostic'] for o in LABELS]
        replacements['@@BOOTSTRAP_SHIFT@@']=(
            'The bootstrap means minus the original estimates are '
            +', '.join(f'${v:.3f}$' for v in shifts)
            +' for crime concerns, preventive behavior, and victimization, respectively. '
            'These shifts flag finite-sample centering sensitivity; they do not measure bias relative to the unknown causal response. '
            'The point estimates are not bias-corrected. The reported symmetric intervals invert the same absolute-error test as the displayed p-values; alternative equal-tailed basic intervals are retained in the saved results.')
    for key,value in replacements.items():text=text.replace(key,value)
    assert '@@' not in text
    original=(WORK/'NPIV.before_full_neural.tex').read_text()
    prefix=original.split('% BEGIN NPIV RESULTS')[0]
    current=(ROOT/'NPIV.tex').read_text()
    assert current.startswith(prefix),'The original exposition has changed; do not overwrite it.'
    (ROOT/'NPIV.tex').write_text(prefix+text+'\n\\end{document}\n')
    assert (ROOT/'NPIV.tex').read_text().split('% BEGIN FULL NEURAL MEMO')[0]==prefix
    print('Wrote NPIV.tex; original template and exposition unchanged; draft=',args.draft)


if __name__=='__main__':main()
