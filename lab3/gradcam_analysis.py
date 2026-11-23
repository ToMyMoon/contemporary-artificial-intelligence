"""固定 A0/AFC 与样本名单的 Grad-CAM；不训练、不读取 SE 权重。"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import baseline as base
import augmentation_ablation as aug
import se_ablation as se

OUT=base.ROOT/'outputs/gradcam_a0_afc'
RUNS={'A0':aug.BASE_RUN,'AFC':se.AFC}
PRIMARY=[3094,262,58,6419]
SELECTION_RULE='Per class: AFC error nearest median error confidence; correct example nearest that error confidence; ties: lower original training index. Selected before heatmap computation.'


def digest_image(x):return hashlib.sha256(x.tobytes()).hexdigest()


def data_and_selection(args,prepare=False):
    train=base.CIFAR10(str(args.data_dir),train=True,download=False)
    test=base.CIFAR10(str(args.data_dir),train=False,download=False)
    arrays={name:{split:np.load((base.ROOT/'outputs/final_afc_epoch47' if name=='AFC' and split=='test' else run)/f'{split}_predictions.npz')
                  for split in ['test','validation']} for name,run in RUNS.items()}
    samples=[];images=[]
    def add(split,index,kind):
        raw=test if split=='test' else train
        image=raw.data[index].copy();target=int(raw.targets[index])
        rec=dict(key=f'{split}_{index}',split=split,index=int(index),true_class=target,
                 true_name=base.CLASSES[target],kind=kind,pixel_sha256=digest_image(image),saved_outputs={})
        for name in RUNS:
            source=arrays[name]['test' if split=='test' else 'validation']
            positions=np.flatnonzero(source['indices']==index);assert len(positions)==1
            pos=int(positions[0]);assert int(source['targets'][pos])==target
            probs=source['probabilities'][pos];prediction=int(probs.argmax())
            rec['saved_outputs'][name]=dict(prediction=prediction,confidence=float(probs[prediction]),probabilities=probs.tolist())
        samples.append(rec);images.append(image)
    for i in PRIMARY:add('test',i,'primary_fixed')
    source=arrays['AFC']['validation'];ids=source['indices'];y=source['targets'];p=source['predictions']
    confidence=source['probabilities'][np.arange(len(y)),p]
    for c in range(10):
        wrong=np.flatnonzero((y==c)&(p!=y));correct=np.flatnonzero((y==c)&(p==y))
        assert len(wrong)>0 and len(correct)>0
        median=float(np.median(confidence[wrong]))
        e=wrong[np.lexsort((ids[wrong],np.abs(confidence[wrong]-median)))[0]]
        k=correct[np.lexsort((ids[correct],np.abs(confidence[correct]-confidence[e])))[0]]
        add('validation',int(ids[e]),'validation_error')
        add('validation',int(ids[k]),'validation_correct')
    for item in arrays.values():
        for a in item.values():a.close()
    assert len(samples)==24 and len(set(s['key'] for s in samples))==24
    for i,expected in enumerate([2,5,4,9]):assert samples[i]['true_class']==expected
    selection=dict(rule=SELECTION_RULE,primary_indices=PRIMARY,samples=samples,
                   primary_examples_fixed_before_this_stage=True,validation_classes=10,
                   validation_examples_per_class=dict(correct=1,error=1),heatmaps_used_for_selection=False)
    if prepare:
        if (args.out/'selection.json').exists():raise FileExistsError('Refusing to replace the sample selection')
        args.out.mkdir(parents=True,exist_ok=True)
        base.write_json(args.out/'selection.json',selection)
    else:
        assert se.read(args.out/'selection.json')==selection
    return samples,np.asarray(images)


def load_models(device):
    models={};checkpoints={}
    for name,run in RUNS.items():
        model=base.BaselineCNN()
        cp=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
        assert cp['epoch']==(59 if name=='A0' else 47)
        model.load_state_dict(cp['state_dict']);models[name]=model.to(device).eval()
        checkpoints[name]=dict(path=str(run/'best.pt'),sha256=base.sha256(run/'best.pt'),epoch=cp['epoch'])
    assert checkpoints['AFC']['sha256']==base.sha256(base.ROOT/'outputs/final_afc_epoch47/best.pt')
    return models,checkpoints


def maps_for(model,x,targets):
    capture={}
    def hook(module,inputs,output):capture['a']=output
    model.zero_grad(set_to_none=True)
    with torch.no_grad():ordinary=model(x).detach().clone()
    handle=model.features[19].register_forward_hook(hook)
    try:
        logits=model(x);a=capture['a']
        assert tuple(a.shape)==(1,128,8,8)
        assert torch.equal(logits.detach(),ordinary), 'Hook changed model output'
        result=[]
        for k,target in enumerate(targets):
            gradient,=torch.autograd.grad(logits[0,target],a,retain_graph=k<len(targets)-1)
            assert torch.isfinite(gradient).all()
            weights=gradient.mean((2,3),keepdim=True)
            raw=(weights*a).sum(1).relu()[0]
            # 对本网络的 MaxPool -> GAP -> Linear 尾部可独立推导 alpha=w_c/64。
            analytic_weights=model.classifier.weight[target]/64
            weight_error=float((weights[0,:,0,0]-analytic_weights).abs().max())
            assert weight_error<1e-7
            analytic=(a[0]*analytic_weights[:,None,None]).sum(0).relu()
            raw_error=float((analytic-raw).abs().max())
            assert raw_error<1e-5
            maximum=float(raw.max());normalized=raw/maximum if maximum>0 else torch.zeros_like(raw)
            assert torch.isfinite(normalized).all()
            result.append(dict(target=int(target),raw=raw.detach().cpu().numpy(),
                normalized=normalized.detach().cpu().numpy(),activations=a[0].detach().cpu().numpy(),
                gradients=gradient[0].detach().cpu().numpy(),raw_max=maximum,zero_map=maximum==0,
                analytic_weight_max_error=weight_error,analytic_raw_max_error=raw_error))
        return logits.detach().cpu().numpy()[0],result
    finally:
        handle.remove()


def correlation(a,b):
    a=a.ravel().astype(float);b=b.ravel().astype(float)
    if a.std()==0 or b.std()==0:return None
    return float(np.corrcoef(a,b)[0,1])


def compute(args):
    if (args.out/'maps.npz').exists():raise FileExistsError('Existing heatmaps; use verify or a new output directory')
    started=time.perf_counter();base.seed_everything(42);device=base.get_device(args.device)
    samples,images=data_and_selection(args,prepare=True)
    base.write_json(args.out/'frozen_inputs.json',{str(p):base.sha256(p) for p in se.frozen_paths()})
    shutil.copy2(__file__,args.out/'gradcam_source.py')
    models,checkpoints=load_models(device)
    state={name:aug.state_hash(m.state_dict()) for name,m in models.items()}
    records=[];maps=[];raw=[];acts=[];grads=[];outputs=[];maximum_probability_difference=0
    x=base.normalize_images(images).to(device)
    for i,sample in enumerate(samples):
        targets=list(dict.fromkeys([sample['true_class'],sample['saved_outputs']['A0']['prediction']]))
        for name,model in models.items():
            logits,results=maps_for(model,x[i:i+1],targets)
            probs=torch.from_numpy(logits).softmax(0).numpy()
            saved=sample['saved_outputs'][name]
            diff=float(np.max(np.abs(probs-np.asarray(saved['probabilities']))))
            maximum_probability_difference=max(maximum_probability_difference,diff)
            assert int(probs.argmax())==saved['prediction'] and diff<2e-5
            outputs.append(dict(sample=i,model=name,logits=logits.tolist(),prediction=int(probs.argmax()),
                confidence=float(probs.max()),max_saved_batch128_probability_difference=diff))
            for result in results:
                j=len(records);target=result['target']
                roles=[r for r,c in [('true',sample['true_class']),('a0_prediction',sample['saved_outputs']['A0']['prediction'])] if c==target]
                records.append(dict(map_index=j,sample=i,sample_key=sample['key'],model=name,
                    target=target,target_name=base.CLASSES[target],target_roles=roles,
                    raw_max=result['raw_max'],zero_map=result['zero_map'],
                    analytic_weight_max_error=result['analytic_weight_max_error'],
                    analytic_raw_max_error=result['analytic_raw_max_error']))
                maps.append(result['normalized']);raw.append(result['raw']);acts.append(result['activations']);grads.append(result['gradients'])
    np.savez_compressed(args.out/'maps.npz',normalized=np.asarray(maps),raw=np.asarray(raw),
                        activations=np.asarray(acts),gradients=np.asarray(grads),images=images)
    base.write_json(args.out/'map_records.json',records);base.write_json(args.out/'model_outputs.json',outputs)
    # 分类头随机化只用于固定20张验证图的敏感性检查，副本不写回原权重。
    randomized=deepcopy(models['AFC'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5042);random_head=nn.Linear(128,10)
    randomized.classifier.load_state_dict(random_head.state_dict())
    randomized_before=aug.state_hash(randomized.state_dict())
    assert randomized_before!=state['AFC']
    index={(r['sample'],r['model'],r['target']):r['map_index'] for r in records}
    sanity=[];random_maps=[];random_raw=[]
    for i,sample in enumerate(samples[4:],4):
        target=sample['true_class'];logits,rr=maps_for(randomized,x[i:i+1],[target]);rr=rr[0]
        old=maps[index[(i,'AFC',target)]];new=rr['normalized']
        sanity.append(dict(sample=i,sample_key=sample['key'],target=target,
            pearson=correlation(old,new),mean_absolute_normalized_difference=float(np.abs(old-new).mean()),
            trained_zero=not bool(np.max(old)>0),randomized_zero=rr['zero_map'],
            exactly_equal=bool(np.array_equal(old,new)),randomized_prediction=int(logits.argmax()),
            analytic_weight_max_error=rr['analytic_weight_max_error']))
        random_maps.append(new);random_raw.append(rr['raw'])
    np.savez_compressed(args.out/'randomized_head_maps.npz',normalized=np.asarray(random_maps),raw=np.asarray(random_raw))
    base.write_json(args.out/'sanity_records.json',sanity)
    assert randomized_before==aug.state_hash(randomized.state_dict())
    assert all(aug.state_hash(m.state_dict())==state[name] for name,m in models.items())
    assert all(base.sha256(Path(c['path']))==c['sha256'] for c in checkpoints.values())
    # 以保存的 A、梯度在 NumPy 中独立复算，核对空间归约和维度。
    a=np.asarray(acts);g=np.asarray(grads)
    independent=np.maximum((a*g.mean(axis=(2,3),keepdims=True)).sum(axis=1),0)
    numerical_error=float(np.max(np.abs(independent-np.asarray(raw))))
    assert numerical_error<1e-5
    stats=dict(source_sha256=base.sha256(__file__),models=checkpoints,samples=24,validation_samples=20,
        primary_samples=4,heatmaps=len(records),target_layer='features[19], final ReLU before final maxpool',
        native_resolution=[8,8],target='pre-softmax class score',methods_identical_across_models=True,
        normalization='ReLU positive map divided by its own max; all-zero maps retained',
        overlay_alpha=.4,interpolation='bilinear 8x8 to32x32; original RGB nearest-neighbor display',
        max_saved_batch128_probability_difference=maximum_probability_difference,
        zero_maps=sum(r['zero_map'] for r in records),independent_numpy_raw_max_error=numerical_error,
        analytical_tail_gradient_verified=True,parameters_and_bn_buffers_unchanged=True,
        original_checkpoints_unchanged=True,randomized_head_seed=5042,
        randomized_head_sanity=dict(samples=20,exactly_equal=sum(r['exactly_equal'] for r in sanity),
            constant_pairs=sum(r['pearson'] is None for r in sanity),
            median_pearson=float(np.median([r['pearson'] for r in sanity if r['pearson'] is not None])),
            mean_normalized_absolute_difference=float(np.mean([r['mean_absolute_normalized_difference'] for r in sanity]))),
        elapsed_compute_seconds=time.perf_counter()-started,environment=base.environment(device),
        localization_accuracy_evaluated=False,training_performed=False,se_model_evaluated=False,
        limitations=['coarse8x8 map, upsampling adds no spatial evidence','no localization annotations or causal intervention',
          'own-max normalization compares spatial pattern, not cross-model strength',
          'randomization checks parameter sensitivity only; no universal correlation pass threshold',
          'illustrative fixed cases and selected class-balanced validation examples, not representative error-rate estimates'])
    base.write_json(args.out/'analysis.json',stats)
    print(json.dumps(stats),flush=True)


def verify(args):
    base.seed_everything(42);device=base.get_device(args.device)
    samples,images=data_and_selection(args)
    models,checkpoints=load_models(device)
    stats=se.read(args.out/'analysis.json');assert stats['source_sha256']==base.sha256(__file__)
    assert stats['models']==checkpoints
    records=se.read(args.out/'map_records.json');outputs=se.read(args.out/'model_outputs.json')
    lookup={(r['sample'],r['model'],r['target']):r['map_index'] for r in records}
    old_outputs={(r['sample'],r['model']):r for r in outputs}
    maximum=0.;maximum_logit=0.;state={n:aug.state_hash(m.state_dict()) for n,m in models.items()}
    x=base.normalize_images(images).to(device)
    with np.load(args.out/'maps.npz') as saved:
        assert np.array_equal(saved['images'],images)
        for i,sample in enumerate(samples):
            targets=list(dict.fromkeys([sample['true_class'],sample['saved_outputs']['A0']['prediction']]))
            for name,model in models.items():
                logits,results=maps_for(model,x[i:i+1],targets)
                maximum_logit=max(maximum_logit,float(np.abs(logits-np.asarray(old_outputs[(i,name)]['logits'])).max()))
                for r in results:
                    j=lookup[(i,name,r['target'])]
                    maximum=max(maximum,float(np.abs(r['raw']-saved['raw'][j]).max()))
                    assert np.array_equal(r['normalized'],saved['normalized'][j])
        assert maximum==0 and maximum_logit==0
    assert all(state[n]==aug.state_hash(m.state_dict()) for n,m in models.items())
    base.write_json(args.out/'verification.json',dict(status='passed',separate_process=True,
        samples=24,heatmaps=len(records),all_raw_and_normalized_maps_exact=True,
        max_raw_difference=maximum,max_logit_difference=maximum_logit,
        sample_selection_and_pixels_reproduced=True,model_state_unchanged=True,
        original_inputs_unchanged=se.check_frozen(args.out)))
    print(json.dumps(se.read(args.out/'verification.json')),flush=True)


def plot(args):
    samples=se.read(args.out/'selection.json')['samples'];records=se.read(args.out/'map_records.json')
    stats=se.read(args.out/'analysis.json');data=np.load(args.out/'maps.npz')
    lookup={(r['sample'],r['model'],r['target']):r['map_index'] for r in records}
    outputs={(r['sample'],r['model']):r for r in se.read(args.out/'model_outputs.json')}
    figures=args.out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    cmap=plt.get_cmap('magma')
    def overlay(i,name,target):
        j=lookup[(i,name,target)];small=data['normalized'][j]
        heat=nn.functional.interpolate(torch.from_numpy(small)[None,None],size=(32,32),mode='bilinear',align_corners=False)[0,0].numpy()
        image=data['images'][i].astype(float)/255
        return image if records[j]['zero_map'] else .6*image+.4*cmap(heat)[:,:,:3]
    def panel(ax,i,column,target,letter):
        s=samples[i]
        if column==0:
            ax.imshow(data['images'][i],interpolation='nearest')
            label=f'({letter}) Original\n{s["true_name"]} | {s["split"]} #{s["index"]}'
        else:
            name=['A0','AFC'][column-1];o=outputs[(i,name)]
            ax.imshow(overlay(i,name,target),interpolation='nearest')
            zero=records[lookup[(i,name,target)]]['zero_map']
            label=f'({letter}) {name}: {base.CLASSES[o["prediction"]]} ({o["confidence"]:.1%})\nTarget: {base.CLASSES[target]}'
            if zero:label+=' | no positive map'
        ax.set_xlabel(label,fontsize=10.5);ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    def save(fig,stem):
        for suffix in ['png','svg']:fig.savefig(figures/f'{stem}.{suffix}',dpi=220,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    def colorbar(fig,axes):
        bar=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cmap),ax=axes,fraction=.018,pad=.02)
        bar.set_label('Positive Grad-CAM / own maximum',fontsize=10)
    for target_role in ['true','a0_prediction']:
        get_target=lambda s:s['true_class'] if target_role=='true' else s['saved_outputs']['A0']['prediction']
        fig,axes=plt.subplots(2,6,figsize=(13.5,6.3),layout='constrained')
        for i in range(4):
            row=i//2;offset=3*(i%2);target=get_target(samples[i])
            for column in range(3):panel(axes[row,offset+column],i,column,target,chr(97+row*6+offset+column))
        colorbar(fig,axes);save(fig,f'primary_{target_role}_compact')
        fig,axes=plt.subplots(4,3,figsize=(9.5,13.5),layout='constrained')
        for i in range(4):
            for column in range(3):panel(axes[i,column],i,column,get_target(samples[i]),chr(97+i*3+column))
        colorbar(fig,axes);save(fig,f'primary_{target_role}_large')
        for page in range(4):
            ids=list(range(4+page*5,4+(page+1)*5))
            fig,axes=plt.subplots(5,3,figsize=(10,17),layout='constrained')
            for row,i in enumerate(ids):
                for column in range(3):panel(axes[row,column],i,column,get_target(samples[i]),chr(97+row*3+column))
            colorbar(fig,axes);save(fig,f'validation_{target_role}_{page+1}')
    data.close()
    base.write_json(args.out/'figure_sources.json',dict(approved_layouts=['2x2 sample groups, original/A0/AFC triplets','4x3 large view'],
        primary_samples=PRIMARY,selection_file=str(args.out/'selection.json'),raw_maps_file=str(args.out/'maps.npz'),
        shared_colorbar=[0,1],normalization='per-map own positive max; colors do not compare model evidence magnitude',
        overlay_alpha=.4,original_image_display='nearest-neighbor RGB; no input augmentation',
        heatmap_upsampling='bilinear8x8to32x32; no added smoothing',figures=sorted(p.name for p in figures.glob('*.png'))))
    print(json.dumps(dict(figures=len(list(figures.glob('*.png'))),directory=str(figures))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['compute','verify','plot']);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--data-dir',type=Path,default=base.ROOT/'data');p.add_argument('--device',choices=['cuda'],default='cuda')
    args=p.parse_args();globals()[args.command](args)
