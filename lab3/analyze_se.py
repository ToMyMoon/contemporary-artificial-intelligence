"""SE 与 AFC 的验证集配对分析、同环境评价计时及已确认曲线。"""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import baseline as base
import augmentation_ablation as aug
import se_ablation as se


def history(run):
    with (run/'history.csv').open() as f:return list(csv.DictReader(f))


def main(args):
    if args.plot_only:
        return plot(args)
    assert se.read(args.out/'verification.json')['status']=='passed'
    records=[];per_class=[]
    with np.load(se.AFC/'validation_predictions.npz') as old, np.load(args.out/'validation_predictions.npz') as new:
        assert np.array_equal(old['indices'],new['indices']) and np.array_equal(old['targets'],new['targets'])
        y=old['targets']; a=old['predictions']; b=new['predictions']; ok0=a==y;ok1=b==y
        repaired=int((~ok0&ok1).sum());introduced=int((ok0&~ok1).sum())
        assert repaired-introduced==int(ok1.sum()-ok0.sum())
        for c,name in enumerate(base.CLASSES):
            mask=y==c
            per_class.append(dict(class_name=name,count=int(mask.sum()),afc_correct=int((ok0&mask).sum()),
                se_correct=int((ok1&mask).sum()),afc_accuracy=float(ok0[mask].mean()),se_accuracy=float(ok1[mask].mean()),
                delta_pp=100*float(ok1[mask].mean()-ok0[mask].mean()),
                repaired=int((~ok0&ok1&mask).sum()),introduced=int((ok0&~ok1&mask).sum())))
        paired=[dict(original_training_index=int(i),true_class=base.CLASSES[t],afc_prediction=base.CLASSES[p0],
            se_prediction=base.CLASSES[p1],change='repaired' if v1 else 'introduced_error')
            for i,t,p0,p1,v0,v1 in zip(old['indices'],y,a,b,ok0,ok1) if v0!=v1]
        se.csv_write(args.out/'paired_validation_changes.csv',paired)
        se.csv_write(args.out/'per_class_comparison.csv',per_class)
    for name,run in [('AFC',se.AFC),('SE-AFC',args.out)]:
        m=se.read(run/'metrics.json');cost=se.read(run/'training_summary.json')
        records.append(dict(group=name,parameters=288746 if name=='AFC' else 291672,
            selected_epoch=m['selected_epoch'],validation_accuracy=m['validation']['accuracy'],
            validation_correct=round(5000*m['validation']['accuracy']),validation_loss=m['validation']['loss'],
            clean_train_accuracy=m['train_eval']['accuracy'],generalization_gap_pp=100*m['generalization_gap'],
            training_seconds=cost['training_wall_seconds'],training_peak_allocated_mib=cost['peak_allocated_cuda_bytes']/2**20))
    # 两模型在同一进程交替测量，避免使用历史训练时差推断算子开销。
    cfg,device,dataset,indexed,tr,va=aug.load_protocol(args)
    models={name:se.load_model(run,device)[0] for name,run in [('AFC',se.AFC),('SE-AFC',args.out)]}
    loader=base.loader_for(dataset,va,128,device)
    state={k:aug.state_hash(m.state_dict()) for k,m in models.items()}
    for model in models.values():base.evaluate_loader(model,loader,device)
    times={k:[] for k in models}
    for repetition in range(5):
        names=list(models) if repetition%2==0 else list(reversed(models))
        for name in names:
            base.synchronize(device);start=time.perf_counter()
            base.evaluate_loader(models[name],loader,device)
            base.synchronize(device);times[name].append(time.perf_counter()-start)
    assert all(state[k]==aug.state_hash(m.state_dict()) for k,m in models.items())
    for r in records:
        values=times[r['group']]
        r.update(validation_eval_median_seconds=float(np.median(values)),validation_eval_min_seconds=min(values),
                 validation_eval_max_seconds=max(values))
    se.csv_write(args.out/'comparison.csv',records)
    se.base.write_json(args.out/'benchmark.json',dict(repetitions=5,warmup_passes_per_model=1,
        alternating_order=True,images_per_pass=5000,batch_size=128,times_seconds=times,
        scope='complete clean validation evaluation, CPU DataLoader + H2D + forward + CE/accuracy; synchronized; no softmax collection; not pure kernel latency',
        evaluation_preserves_both_models=True,environment=base.environment(device)))
    summary=dict(reference=records[0],se=records[1],validation_accuracy_delta_pp=100*(records[1]['validation_accuracy']-records[0]['validation_accuracy']),
        repaired_afc_errors=repaired,introduced_errors=introduced,net_correct=repaired-introduced,
        both_correct=int((ok0&ok1).sum()),both_wrong=int((~ok0&~ok1).sum()),per_class=per_class,
        limitations=['one training seed and fixed split; not stability evidence',
          'same validation set used for checkpoint/configuration selection',
          'SE includes extra parameters; no parameter-matched control',
          'historical training times are descriptive; controlled timing uses current same-process validation benchmark'],
        se_test_evaluated=False,formal_report_modified=False)
    base.write_json(args.out/'comparison_analysis.json',summary)
    plot(args)
    print(__import__('json').dumps(summary),flush=True)


def plot(args):
    summary=se.read(args.out/'comparison_analysis.json')
    records=[summary['reference'],summary['se']]
    figures=args.out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    handles=[]
    for color,(name,run) in zip(['#1769a8','#d35f13'],[('AFC',se.AFC),('SE-AFC',args.out)]):
        h=history(run);epochs=np.array([int(r['epoch']) for r in h]);chosen=se.read(run/'metrics.json')['selected_epoch']
        for ax,metric,scale,ylabel in zip(axes,['loss','accuracy'],[1,100],['Mean cross-entropy','Accuracy (%)']):
            for phase,style,label in [('train','--','training (online)'),('val','-','validation (epoch end)')]:
                line,=ax.plot(epochs,[float(r[f'{phase}_{metric}'])*scale for r in h],color=color,ls=style,lw=1.6,
                              label=f'{name}: {label}')
                if ax is axes[0]:handles.append(line)
            ax.axvline(chosen,color=color,ls=':',lw=1,alpha=.8)
            ax.set(xlim=(1,60),xlabel='Epoch\n'+('(a) Cross-entropy' if metric=='loss' else '(b) Accuracy'),ylabel=ylabel)
            ax.grid(alpha=.18)
    selected='Dotted lines: AFC '+str(records[0]['selected_epoch'])+'; SE-AFC '+str(records[1]['selected_epoch'])
    for ax in axes:ax.text(.5,1.015,selected,transform=ax.transAxes,ha='center',fontsize=9)
    axes[0].set_ylim(bottom=0);axes[1].set_ylim(0,100)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1),ncol=2,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.065,right=.99,bottom=.19,top=.83,wspace=.22)
    for suffix in ['png','svg']:fig.savefig(figures/f'se_training_comparison.{suffix}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    base.write_json(args.out/'figure_sources.json',dict(sources=[str(se.AFC/'history.csv'),str(args.out/'history.csv')],
        points_per_curve=60,smoothing=False,online_train_vs_epoch_end_val=True,
        selected_epochs={r['group']:r['selected_epoch'] for r in records}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=se.OUT)
    p.add_argument('--data-dir',type=Path,default=base.ROOT/'data');p.add_argument('--device',default='cuda',choices=['cuda'])
    p.add_argument('--plot-only',action='store_true')
    main(p.parse_args())
