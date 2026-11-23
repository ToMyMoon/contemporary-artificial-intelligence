"""固定 AFC 协议的 SE 对照：仅训练集/验证集，无测试集入口。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path, PureWindowsPath
import shutil
import time

import numpy as np
import torch
from torch import nn

import baseline as base
import augmentation_ablation as aug

ROOT = base.ROOT
AFC = aug.OUT / 'AFC'
OUT = ROOT / 'outputs' / 'se_seed42'
SE_SEED = 4042


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)

    def forward(self, x):
        gate = self.fc2(torch.relu(self.fc1(x.mean((2, 3))))).sigmoid()
        return x * gate[:, :, None, None]


class SEBaselineCNN(base.BaselineCNN):
    def __init__(self):
        super().__init__()
        # SE 参数使用独立初始化随机流，不影响原共有层初始化。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SE_SEED)
            self.se = nn.ModuleList(SEBlock(c) for c in (32, 64, 128))
        self.bypass_se = False

    def forward(self, x):
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in (5, 12, 19) and not self.bypass_se:
                x = self.se[(index - 5) // 7](x)
        return self.classifier(x.mean((2, 3)))


def initialized_pair():
    base.seed_everything(42)
    reference = base.BaselineCNN()
    rng_after_original = torch.get_rng_state().clone()
    model = SEBaselineCNN()
    incompatible = model.load_state_dict(reference.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(k.startswith('se.') for k in incompatible.missing_keys)
    common = {k: v for k, v in model.state_dict().items() if not k.startswith('se.')}
    assert aug.state_hash(common) == aug.state_hash(reference.state_dict())
    assert aug.state_hash(common) == read(aug.OUT / 'preflight.json')['initial_state_sha256']
    torch.set_rng_state(rng_after_original)
    return reference, model


def frozen_paths():
    paths = [ROOT / n for n in ['baseline.py', 'augmentation_ablation.py']]
    for run in [aug.BASE_RUN, AFC, ROOT / 'outputs/final_afc_epoch47']:
        paths += [p for p in run.rglob('*') if p.is_file()]
    return sorted(set(p for p in paths if p.is_file()))


def read_frozen(out):
    """Read old absolute or new project-relative experiment manifests."""
    saved = read(out / 'frozen_inputs.json')
    paths = {
        (PureWindowsPath(name) if '\\' in name else Path(name)): digest
        for name, digest in saved.items()
    }
    # Legacy manifests used their original checkout's absolute paths.
    original_root = next(
        (path.parent for path in paths if path.is_absolute() and path.name == 'baseline.py'),
        None,
    )
    inputs = {}
    for path, digest in paths.items():
        if path.is_absolute():
            if original_root is None:
                raise ValueError(f'Cannot locate the experiment root in {out}')
            path = path.relative_to(original_root)
        if path.parts[0] != 'report':
            inputs[path.as_posix()] = digest
    return inputs


def check_frozen(out):
    inputs = read_frozen(out)
    for relative_path, expected in inputs.items():
        path = ROOT / relative_path
        actual = base.sha256(path)
        if relative_path == 'final_test_afc.py' and actual != expected:
            # Historical runs recorded the launcher in addition to its saved copy.
            snapshot = ROOT / 'outputs/final_afc_epoch47/final_evaluation_source.py'
            actual = base.sha256(snapshot)
        assert actual == expected, f'Frozen input changed: {path}'
    return len(inputs)


def preflight(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'config.json').exists():
        raise FileExistsError('Existing formal run; choose a new output directory')
    base.write_json(args.out / 'frozen_inputs.json', {p.relative_to(ROOT).as_posix(): base.sha256(p) for p in frozen_paths()})
    cfg, device, dataset, indexed, tr, va = aug.load_protocol(args)
    original, model = initialized_pair()
    assert sum(p.numel() for p in original.parameters()) == 288746
    assert sum(p.numel() for p in model.parameters()) == 291672
    assert sum(p.numel() for p in model.se.parameters()) == 2926
    shared_hash = aug.state_hash(original.state_dict())
    full_hash = aug.state_hash(model.state_dict())
    original, model = original.to(device), model.to(device)
    x = dataset.tensors[0][tr[:8]].to(device)
    model.eval(); original.eval(); model.bypass_se = True
    with torch.no_grad():
        assert torch.equal(original(x), model(x))
    model.bypass_se = False
    with torch.no_grad():
        features = original.features[:6](x)
        block = model.se[0]
        independent_gate = torch.sigmoid(nn.functional.linear(
            torch.relu(nn.functional.linear(features.mean((2, 3)), block.fc1.weight, block.fc1.bias)),
            block.fc2.weight, block.fc2.bias))
        manual = features * independent_gate[:, :, None, None]
        assert torch.equal(block(features), manual)
        assert torch.all((independent_gate > 0) & (independent_gate < 1))
    model.train()
    y = dataset.tensors[1][tr[:8]].to(device)
    nn.functional.cross_entropy(model(x), y).backward()
    se_gradients = {n: float(p.grad.abs().sum()) for n, p in model.named_parameters() if n.startswith('se.')}
    assert all(np.isfinite(v) for v in se_gradients.values())
    assert all(sum(v for n, v in se_gradients.items() if n.startswith(f'se.{i}.')) > 0 for i in range(3))
    # 核查原 AFC 的实际增强参数，而非仅比较种子值。
    expected = [json.loads(s) for s in (AFC / 'augmentation_trace.jsonl').read_text().splitlines()]
    generator = aug.Augmentation((True, True, False), device)
    for i, old in enumerate(expected):
        _, trace = generator.epoch_parameters(45000)
        assert trace['flip_sha256'] == old['flip_sha256']
        assert trace['crop_sha256'] == old['crop_sha256']
    # 丢弃以上核查参数，以全新初始化模型检验64张小样本可拟合。
    del original, model
    _, tiny = initialized_pair(); tiny = tiny.to(device).train()
    optimizer = aug.optimizer_for(tiny, cfg)
    tx = dataset.tensors[0][tr[:64]].to(device); ty = dataset.tensors[1][tr[:64]].to(device)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(tiny(tx), ty)
        loss.backward(); optimizer.step()
    tiny.eval()
    with torch.no_grad():
        tiny_accuracy = float((tiny(tx).argmax(1) == ty).float().mean())
    assert tiny_accuracy >= .95
    base.write_json(args.out / 'preflight.json', dict(status='passed', source_sha256=base.sha256(__file__),
        common_initial_state_sha256=shared_hash, full_initial_state_sha256=full_hash,
        shared_layers_identical=True, bypass_bitwise_equal=True, independent_se_formula_equal=True,
        parameters=291672, added_parameters=2926, se_gradient_l1=se_gradients,
        all_60_augmentation_parameter_hashes_match_afc=True,
        tiny_fit=dict(samples=64, steps=100, eval_accuracy=tiny_accuracy, discarded=True),
        frozen_files=check_frozen(args.out), test_evaluated=False))
    print(json.dumps(read(args.out / 'preflight.json')), flush=True)


def train(args):
    pre = read(args.out / 'preflight.json')
    assert pre['status'] == 'passed' and pre['source_sha256'] == base.sha256(__file__)
    if (args.out / 'config.json').exists():
        raise FileExistsError('Refusing to overwrite an existing training run')
    cfg, device, dataset, indexed, tr, va = aug.load_protocol(args)
    reference, model = initialized_pair()
    del reference
    assert aug.state_hash(model.state_dict()) == pre['full_initial_state_sha256']
    model = model.to(device)
    conf = {**cfg, 'group': 'SE-AFC', 'model': 'BaselineCNN + 3 SE blocks before each maxpool',
        'augmentation': dict(flip=True, crop=True, brightness=False),
        'augmentation_settings': read(AFC / 'config.json')['augmentation_settings'],
        'se': dict(reduction=16, hidden_dimensions=[2, 4, 8], biases=True,
                   after_feature_indices=[5, 12, 19], initialization_seed=SE_SEED),
        'common_initial_state_sha256': pre['common_initial_state_sha256'],
        'initial_state_sha256': pre['full_initial_state_sha256'], 'parameters':291672,
        'source_sha256':base.sha256(__file__), 'augmentation_source_sha256':base.sha256(aug.__file__),
        'metric_semantics':read(AFC / 'config.json')['metric_semantics'], 'test_evaluated':False}
    base.write_json(args.out / 'config.json', conf)
    base.write_json(args.out / 'environment.json', base.environment(device))
    for path in [Path(__file__), Path(base.__file__), Path(aug.__file__), ROOT / 'requirements-lock.txt',
                 aug.BASE_RUN / 'split_indices.npz']:
        shutil.copy2(path, args.out / path.name)
    (args.out / 'model.txt').write_text(str(model), encoding='utf-8')
    loader = base.loader_for(indexed, tr, cfg['batch_size'], device, True, cfg['seed'])
    val_loader = base.loader_for(dataset, va, cfg['batch_size'], device)
    optimizer = aug.optimizer_for(model, cfg)
    augment = aug.Augmentation((True, True, False), device)
    old_trace = [json.loads(s) for s in (AFC / 'augmentation_trace.jsonl').read_text().splitlines()]
    fields = ['epoch','train_loss','train_accuracy','val_loss','val_accuracy','learning_rate',
              'train_seconds','val_seconds','elapsed_seconds','selected']
    best_key = (-float('inf'), -float('inf')); best_epoch = None
    base.synchronize(device); torch.cuda.reset_peak_memory_stats(device); start_all = time.perf_counter()
    with (args.out / 'history.csv').open('w', encoding='utf-8', newline='') as hf, \
         (args.out / 'augmentation_trace.jsonl').open('w', encoding='utf-8') as tf:
        writer = csv.DictWriter(hf, fieldnames=fields); writer.writeheader()
        for epoch in range(1, 61):
            base.synchronize(device); start = time.perf_counter()
            training, trace = aug.train_epoch(model, loader, optimizer, device, augment, len(tr))
            base.synchronize(device); training_seconds = time.perf_counter()-start
            for key in ['sample_order_sha256','flip_sha256','crop_sha256']:
                assert trace[key] == old_trace[epoch-1][key], f'{epoch}: {key} mismatch'
            start=time.perf_counter(); validation=base.evaluate_loader(model,val_loader,device)
            base.synchronize(device); validation_seconds=time.perf_counter()-start
            key=validation['accuracy'], -validation['loss']; selected=key>best_key
            if selected:
                best_key=key; best_epoch=epoch
                torch.save(dict(state_dict=model.state_dict(),epoch=epoch,
                    val_accuracy=validation['accuracy'],val_loss=validation['loss'],config=conf), args.out/'best.pt')
            row=dict(epoch=epoch,train_loss=training['loss'],train_accuracy=training['accuracy'],
                val_loss=validation['loss'],val_accuracy=validation['accuracy'],learning_rate=.001,
                train_seconds=training_seconds,val_seconds=validation_seconds,
                elapsed_seconds=time.perf_counter()-start_all,selected=int(selected))
            writer.writerow(row); hf.flush();tf.write(json.dumps(dict(epoch=epoch,**trace))+'\n');tf.flush()
            print(json.dumps(row),flush=True)
    base.synchronize(device)
    duration=time.perf_counter()-start_all
    peak=torch.cuda.max_memory_allocated(device)
    torch.save(dict(state_dict=model.state_dict(),epoch=60,config=conf),args.out/'last.pt')
    base.write_json(args.out/'training_summary.json',dict(completed_epochs=60,best_epoch=best_epoch,
        training_wall_seconds=duration,peak_allocated_cuda_bytes=peak,
        timing_scope='60 training epochs + validation + checkpoint/history/trace writes; excludes data load, final evaluations and plotting'))
    del model,optimizer
    torch.cuda.empty_cache()
    evaluate(args,cfg,device,dataset,tr,va)


def load_model(run, device):
    cp=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    model=(SEBaselineCNN() if run != AFC else base.BaselineCNN())
    model.load_state_dict(cp['state_dict']);return model.to(device).eval(),cp


def evaluate(args,cfg,device,dataset,tr,va):
    model,cp=load_model(args.out,device); before=aug.state_hash(model.state_dict())
    train_eval=base.evaluate_loader(model,base.loader_for(dataset,tr,128,device),device)
    value=base.evaluate_loader(model,base.loader_for(dataset,va,128,device),device,collect=True)
    prob=value.pop('probabilities');y=value.pop('targets');pred=prob.argmax(1)
    assert value['accuracy']==cp['val_accuracy']
    assert abs(value['loss']-cp['val_loss'])<1e-6
    assert before==aug.state_hash(model.state_dict())
    matrix=base.confusion_matrix(y,pred);assert np.array_equal(matrix.sum(1),np.full(10,500))
    np.savez_compressed(args.out/'validation_predictions.npz',indices=va,targets=y,predictions=pred,probabilities=prob)
    base.write_confusion(args.out/'validation_confusion_counts.csv',matrix)
    base.write_confusion(args.out/'validation_confusion_normalized.csv',matrix/500)
    value['correct']=int((pred==y).sum())
    metrics=dict(group='SE-AFC',selected_epoch=cp['epoch'],train_eval=train_eval,validation=value,
        generalization_gap=train_eval['accuracy']-value['accuracy'],per_class_accuracy=(matrix.diagonal()/500).tolist(),
        parameters=291672,added_parameters=2926,checkpoint_sha256=base.sha256(args.out/'best.pt'),
        model_state_unchanged_by_evaluation=True,test_evaluated=False)
    base.write_json(args.out/'metrics.json',metrics)
    print(json.dumps(dict(event='complete',**metrics)),flush=True)


def verify(args):
    cfg,device,dataset,indexed,tr,va=aug.load_protocol(args)
    metrics=read(args.out/'metrics.json')
    with (args.out/'history.csv').open() as f:history=list(csv.DictReader(f))
    assert len(history)==60 and [int(r['epoch']) for r in history]==list(range(1,61))
    chosen=max(history,key=lambda r:(float(r['val_accuracy']),-float(r['val_loss']),-int(r['epoch'])))
    assert int(chosen['epoch'])==metrics['selected_epoch']
    assert base.sha256(args.out/'best.pt')==metrics['checkpoint_sha256']
    with np.load(args.out/'validation_predictions.npz') as saved:
        model,_=load_model(args.out,device)
        before=aug.state_hash(model.state_dict())
        result=base.evaluate_loader(model,base.loader_for(dataset,va,128,device),device,collect=True)
        assert np.array_equal(saved['indices'],va) and np.array_equal(saved['targets'],result['targets'])
        diff=float(np.max(np.abs(saved['probabilities']-result['probabilities'])))
        assert diff==0 and before==aug.state_hash(model.state_dict())
        y=saved['targets']; p=saved['predictions']; n=len(y)
        matrix=np.zeros((10,10),dtype=np.int64)
        for a,b in zip(y,p):matrix[a,b]+=1
        ce=float(-np.log(saved['probabilities'][np.arange(n),y].astype(np.float64)).mean())
        assert abs(ce-metrics['validation']['loss'])<1e-5
        assert int(np.trace(matrix))==metrics['validation']['correct']
    base.write_json(args.out/'verification.json',dict(status='passed',all_5000_predictions_and_probabilities_exact=True,
        max_reload_probability_difference=diff,independent_ce=ce,correct=int(matrix.trace()),
        selected_epoch_verified=True,evaluation_preserves_state=True,
        frozen_files_unchanged=check_frozen(args.out),test_evaluated=False,
        note='Separate-process checkpoint inference reproduction; not an independent full training repeat'))
    print(json.dumps(read(args.out/'verification.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','train','verify'])
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--device',choices=['cuda'],default='cuda')
    args=p.parse_args();globals()[args.command](args)
