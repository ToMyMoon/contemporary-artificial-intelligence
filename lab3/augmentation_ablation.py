"""CIFAR-10 三因素完整消融；训练与验证，不读取测试集预测。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

import baseline as base

ROOT = base.ROOT
BASE_RUN = ROOT / "outputs" / "baseline_seed42"
OUT = ROOT / "outputs" / "augmentation_seed42"
GROUPS = {
    "A0": (False, False, False), "AF": (True, False, False),
    "AC": (False, True, False), "AB": (False, False, True),
    "AFC": (True, True, False), "AFB": (True, False, True),
    "ACB": (False, True, True), "AFCB": (True, True, True),
}
SEEDS = {"flip": 1042, "crop": 2042, "brightness": 3042}


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Augmentation:
    """按一轮样本呈现顺序生成参数，各因素使用独立 CPU Generator。"""
    def __init__(self, flags, device):
        self.flip, self.crop, self.brightness = flags
        self.device = device
        self.generators = {name: torch.Generator().manual_seed(seed) for name, seed in SEEDS.items()}

    def epoch_parameters(self, count):
        params, trace = {}, {"sample_presentations": count}
        if self.flip:
            x = torch.rand(count, generator=self.generators["flip"]) < 0.5
            trace.update(flip_sha256=tensor_hash(x), flipped_count=int(x.sum()))
            params["flip"] = x.to(self.device)
        if self.crop:
            x = torch.randint(0, 9, (count, 2), generator=self.generators["crop"])
            trace.update(crop_sha256=tensor_hash(x), crop_min=int(x.min()), crop_max=int(x.max()))
            params["crop"] = x.to(self.device)
        if self.brightness:
            x = torch.rand(count, generator=self.generators["brightness"]) * 0.4 + 0.8
            trace.update(brightness_sha256=tensor_hash(x), brightness_min=float(x.min()),
                         brightness_max=float(x.max()), brightness_mean=float(x.mean()))
            params["brightness"] = x.to(self.device)
        return params, trace

    @staticmethod
    def apply(x, params):
        # x 已按 mean=.5/std=.5 归一化；黑色像素对应 -1。
        if "crop" in params:
            padded = nn.functional.pad(x, (4, 4, 4, 4), value=-1)
            offsets = params["crop"]
            axis = torch.arange(32, device=x.device)
            rows = offsets[:, 0, None, None] + axis[None, :, None]
            cols = offsets[:, 1, None, None] + axis[None, None, :]
            indices = (rows * 40 + cols).flatten(1)[:, None, :].expand(-1, 3, -1)
            x = padded.flatten(2).gather(2, indices).reshape(-1, 3, 32, 32)
        if "flip" in params:
            x = torch.where(params["flip"][:, None, None, None], x.flip(-1), x)
        if "brightness" in params:
            # 等价于先在 [0,1] 浮点 RGB 上乘亮度并截断，再做原归一化。
            x = ((x * 0.5 + 0.5) * params["brightness"][:, None, None, None]).clamp(0, 1)
            x = (x - 0.5) / 0.5
        return x


def train_epoch(model, loader, optimizer, device, augment, count):
    model.train()
    parameters, trace = augment.epoch_parameters(count)
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.int64)
    n, order = 0, hashlib.sha256()
    for x, y, indices in loader:
        order.update(indices.numpy().astype("<i8", copy=False).tobytes())
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = augment.apply(x, {k: v[n:n + len(y)] for k, v in parameters.items()})
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        total_loss += loss.detach() * len(y)
        total_correct += (logits.detach().argmax(1) == y).sum()
        n += len(y)
    assert n == count
    trace["sample_order_sha256"] = order.hexdigest()
    return dict(loss=total_loss.item() / n, accuracy=total_correct.item() / n, count=n), trace


def load_protocol(args):
    cfg = json.loads((BASE_RUN / "config.json").read_text(encoding="utf-8"))
    assert base.sha256(ROOT / "baseline.py") == cfg["source_sha256"]
    assert base.sha256(BASE_RUN / "baseline_source.py") == cfg["source_sha256"]
    assert base.sha256(BASE_RUN / "split_indices.npz") == cfg["split_sha256"]
    assert cfg["epochs"] == 60 and cfg["seed"] == 42
    device = base.get_device(args.device)
    base.seed_everything(cfg["seed"])
    actual = base.environment(device)
    previous = json.loads((BASE_RUN / "environment.json").read_text(encoding="utf-8"))
    for name in ["torch", "torchvision", "numpy", "cuda_runtime", "cudnn", "gpu", "cpu_threads"]:
        if actual[name] != previous[name]:
            raise RuntimeError(f"Runtime changed for {name}: {actual[name]} != {previous[name]}")
    dataset, labels = base.load_training(args.data_dir)
    split = np.load(BASE_RUN / "split_indices.npz")
    tr, va = split["train_indices"].copy(), split["val_indices"].copy()
    expected_tr, expected_va = base.make_split(labels, cfg["seed"])
    assert np.array_equal(tr, expected_tr) and np.array_equal(va, expected_va)
    assert not np.intersect1d(tr, va).size
    indexed = TensorDataset(*dataset.tensors, torch.arange(len(dataset)))
    return cfg, device, dataset, indexed, tr, va


def optimizer_for(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"], betas=tuple(cfg["optimizer_betas"]),
                            eps=cfg["optimizer_eps"])


def preflight(args):
    cfg, device, dataset, indexed, tr, _ = load_protocol(args)
    # 与独立 NumPy 实现逐像素比较八种条件，包含黑边和亮度截断。
    images = np.arange(8 * 32 * 32 * 3, dtype=np.int64).reshape(8, 32, 32, 3) % 256
    images = images.astype(np.uint8)
    normalized = base.normalize_images(images).to(device)
    params = dict(crop=torch.tensor([[0, 0], [8, 8], [4, 4], [0, 8], [8, 0], [2, 6], [7, 1], [5, 3]], device=device),
                  flip=torch.tensor([True, False, True, False, True, False, True, False], device=device),
                  brightness=torch.tensor([0.8, 1.2, 0.9, 1.1, 1.0, 0.85, 1.15, 0.95], device=device))
    max_error = 0.0
    for group, (f, c, b) in GROUPS.items():
        enabled = {k: v for k, v in params.items() if {"flip": f, "crop": c, "brightness": b}[k]}
        output = Augmentation.apply(normalized, enabled).cpu().numpy()
        expected = []
        for i, image in enumerate(images):
            item = image.astype(np.float32) / np.float32(255)
            if c:
                item = np.pad(item, ((4, 4), (4, 4), (0, 0)), constant_values=0)
                top, left = params["crop"][i].cpu().tolist()
                item = item[top:top + 32, left:left + 32]
            if f and bool(params["flip"][i]):
                item = item[:, ::-1]
            if b:
                item = np.clip(item * np.float32(params["brightness"][i].item()), 0, 1)
            expected.append(((item - np.float32(.5)) / np.float32(.5)).transpose(2, 0, 1))
        difference = float(np.max(np.abs(output - np.stack(expected))))
        max_error = max(max_error, difference)
        assert difference <= 2e-7, (group, difference)
        assert output.shape == (8, 3, 32, 32) and np.isfinite(output).all()
        assert output.min() >= -1 and output.max() <= 1
    assert Augmentation.apply(normalized, {}) is normalized
    # 独立因素共享同一参数序列，且不消耗模型或 DataLoader 的默认 RNG。
    cpu_state, cuda_states = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    traces = {}
    for group, flags in GROUPS.items():
        _, trace = Augmentation(flags, device).epoch_parameters(45000)
        traces[group] = trace
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_states, torch.cuda.get_rng_state_all()))
    for factor, position in [("flip", 0), ("crop", 1), ("brightness", 2)]:
        assert len({t[f"{factor}_sha256"] for g, t in traces.items() if GROUPS[g][position]}) == 1
    # 原始 DataLoader 与含样本 ID 的 DataLoader，前三批输入完全相同。
    original = base.loader_for(dataset, tr, cfg["batch_size"], device, True, 42)
    augmented = base.loader_for(indexed, tr, cfg["batch_size"], device, True, 42)
    old_batches, new_batches = [], []
    for old, new in zip(itertools.islice(original, 3), itertools.islice(augmented, 3)):
        assert torch.equal(old[0], new[0]) and torch.equal(old[1], new[1])
        old_batches.append(old)
        new_batches.append(new)
    # 相同初始化下，无增强新入口与原入口两步更新逐位一致。
    base.seed_everything(42)
    first = base.BaselineCNN().to(device)
    initial_hash = state_hash(first.state_dict())
    first_result = base.train_epoch(first, old_batches[:2], optimizer_for(first, cfg), device)
    base.seed_everything(42)
    second = base.BaselineCNN().to(device)
    assert state_hash(second.state_dict()) == initial_hash
    second_result, _ = train_epoch(second, new_batches[:2], optimizer_for(second, cfg), device,
                                   Augmentation(GROUPS["A0"], device), 256)
    assert first_result == second_result
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    # 为基线原有采样协议生成完整 60 轮顺序哈希；没有重新训练 A0。
    only_ids = TensorDataset(torch.arange(len(dataset)))
    order_loader = base.loader_for(only_ids, tr, cfg["batch_size"], device, True, 42)
    orders = []
    for epoch in range(1, 61):
        h = hashlib.sha256()
        for (indices,) in order_loader:
            h.update(indices.numpy().astype("<i8", copy=False).tobytes())
        orders.append(dict(epoch=epoch, sample_order_sha256=h.hexdigest()))
    args.out.mkdir(parents=True, exist_ok=True)
    base.write_json(args.out / "expected_sample_orders.json", orders)
    result = dict(status="passed", transform_numpy_max_absolute_error=max_error,
                  all_eight_conditions_checked=True, black_padding_and_clipping_verified=True,
                  disabled_augmentation_is_identity=True, default_rng_unchanged=True,
                  factor_streams_paired=True, first_three_batches_equal=True,
                  disabled_two_step_updates_bitwise_equal=True, initial_state_sha256=initial_hash,
                  baseline_source_sha256=cfg["source_sha256"], split_sha256=cfg["split_sha256"],
                  ablation_source_sha256=base.sha256(__file__),
                  sample_order_note="Reconstructed 60 epochs using baseline loader/seed; original run did not log order hashes. Checked against actual order in each new run.",
                  note="Correctness-check weights discarded; no baseline retraining or test evaluation")
    base.write_json(args.out / "preflight.json", result)
    print(json.dumps(result), flush=True)


def run_group(args, group, cfg, device, dataset, indexed, tr, va, preflight_record, expected_orders):
    run = args.out / group
    if (run / "config.json").exists():
        raise FileExistsError(f"Refusing to overwrite run {run}")
    run.mkdir(parents=True, exist_ok=True)
    base.seed_everything(cfg["seed"])
    model = base.BaselineCNN().to(device)
    initial_hash = state_hash(model.state_dict())
    assert initial_hash == preflight_record["initial_state_sha256"]
    flags = GROUPS[group]
    conf = {**cfg, "group": group, "augmentation": dict(flip=flags[0], crop=flags[1], brightness=flags[2]),
            "augmentation_settings": dict(flip_probability=0.5, crop_padding=4, crop_size=32,
                padding_raw_pixel=0, padding_normalized_value=-1, brightness_range=[0.8, 1.2],
                brightness_precision="float32 on [0,1]; clamp then normalize; no uint8 requantization",
                order=["crop", "flip", "brightness", "normalize"],
                implementation="equivalent operations on normalized GPU tensors",
                parameter_sampling="independent CPU Generator per factor; fresh 45000 parameters per epoch, consumed in shared sample order",
                factor_seeds=SEEDS),
            "initial_state_sha256": initial_hash, "baseline_source_sha256": cfg["source_sha256"],
            "source_sha256": base.sha256(__file__), "test_evaluated": False,
            "metric_semantics": {**cfg["metric_semantics"],
                "train": "online predictions on augmented samples before each batch update; BN train mode",
                "train_eval": "selected checkpoint on unaugmented 45000 training images; BN eval mode"}}
    base.write_json(run / "config.json", conf)
    base.write_json(run / "environment.json", base.environment(device))
    shutil.copy2(__file__, run / "augmentation_source.py")
    shutil.copy2(ROOT / "baseline.py", run / "baseline_source.py")
    shutil.copy2(BASE_RUN / "split_indices.npz", run / "split_indices.npz")
    shutil.copy2(ROOT / "requirements-lock.txt", run / "requirements-lock.txt")
    train_loader = base.loader_for(indexed, tr, cfg["batch_size"], device, True, cfg["seed"])
    val_loader = base.loader_for(dataset, va, cfg["batch_size"], device)
    optimizer = optimizer_for(model, cfg)
    augment = Augmentation(flags, device)
    best_key, best_epoch = (-float("inf"), -float("inf")), None
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate",
              "train_seconds", "val_seconds", "elapsed_seconds", "selected"]
    base.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    print(json.dumps(dict(event="group_start", group=group, flags=flags)), flush=True)
    with (run / "history.csv").open("w", encoding="utf-8", newline="") as hf, \
            (run / "augmentation_trace.jsonl").open("w", encoding="utf-8") as tf:
        writer = csv.DictWriter(hf, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, cfg["epochs"] + 1):
            base.synchronize(device)
            start = time.perf_counter()
            training, trace = train_epoch(model, train_loader, optimizer, device, augment, len(tr))
            base.synchronize(device)
            train_seconds = time.perf_counter() - start
            assert trace["sample_order_sha256"] == expected_orders[epoch - 1]["sample_order_sha256"]
            start = time.perf_counter()
            validation = base.evaluate_loader(model, val_loader, device)
            base.synchronize(device)
            val_seconds = time.perf_counter() - start
            key = validation["accuracy"], -validation["loss"]
            selected = key > best_key
            if selected:
                best_key, best_epoch = key, epoch
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch,
                                val_accuracy=validation["accuracy"], val_loss=validation["loss"], config=conf), run / "best.pt")
            row = dict(epoch=epoch, train_loss=training["loss"], train_accuracy=training["accuracy"],
                       val_loss=validation["loss"], val_accuracy=validation["accuracy"],
                       learning_rate=optimizer.param_groups[0]["lr"], train_seconds=train_seconds,
                       val_seconds=val_seconds, elapsed_seconds=time.perf_counter() - wall_start,
                       selected=int(selected))
            writer.writerow(row)
            hf.flush()
            tf.write(json.dumps(dict(epoch=epoch, **trace)) + "\n")
            tf.flush()
            print(json.dumps(dict(group=group, **row)), flush=True)
    base.synchronize(device)
    duration = time.perf_counter() - wall_start
    peak_allocated = torch.cuda.max_memory_allocated(device)
    torch.save(dict(state_dict=model.state_dict(), epoch=cfg["epochs"], config=conf), run / "last.pt")
    base.write_json(run / "training_summary.json", dict(completed_epochs=cfg["epochs"], best_epoch=best_epoch,
            training_wall_seconds=duration, peak_allocated_cuda_bytes=peak_allocated,
            timing_scope="same baseline scope plus online augmentation parameter generation, device augmentation and trace/order checks; excludes dataset loading, final evaluation/export and plots"))
    # 仅对最佳权重进行干净训练集与验证集评价；没有测试集入口。
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    original_state = state_hash(model.state_dict())
    clean_train = base.evaluate_loader(model, base.loader_for(dataset, tr, cfg["batch_size"], device), device)
    val_result = base.evaluate_loader(model, val_loader, device, collect=True)
    probabilities, targets = val_result.pop("probabilities"), val_result.pop("targets")
    assert val_result["accuracy"] == checkpoint["val_accuracy"]
    assert original_state == state_hash(model.state_dict())
    predictions = probabilities.argmax(1)
    matrix = base.confusion_matrix(targets, predictions)
    assert np.array_equal(matrix.sum(1), np.full(10, 500))
    np.savez_compressed(run / "validation_predictions.npz", indices=va, targets=targets,
                        predictions=predictions, probabilities=probabilities)
    base.write_confusion(run / "validation_confusion_counts.csv", matrix)
    base.write_confusion(run / "validation_confusion_normalized.csv", matrix / 500)
    metrics = dict(group=group, selected_epoch=checkpoint["epoch"], train_eval=clean_train,
                   validation=val_result, generalization_gap=clean_train["accuracy"] - val_result["accuracy"],
                   validation_correct=int(np.trace(matrix)), per_class_accuracy=(np.diag(matrix) / 500).tolist(),
                   checkpoint_sha256=base.sha256(run / "best.pt"), test_evaluated=False,
                   model_state_unchanged_by_evaluation=True)
    base.write_json(run / "metrics.json", metrics)
    print(json.dumps(dict(event="group_complete", **metrics)), flush=True)
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()


def train(args):
    record = json.loads((args.out / "preflight.json").read_text(encoding="utf-8"))
    assert record["status"] == "passed" and record["ablation_source_sha256"] == base.sha256(__file__)
    cfg, device, dataset, indexed, tr, va = load_protocol(args)
    expected = json.loads((args.out / "expected_sample_orders.json").read_text(encoding="utf-8"))
    protocol = dict(groups={g: dict(zip(["flip", "crop", "brightness"], f)) for g, f in GROUPS.items()},
                    reused_baseline=str(BASE_RUN), seed=42, epochs=60, factor_seeds=SEEDS,
                    selection=cfg["selection"], test_evaluation="not performed in this stage")
    base.write_json(args.out / "protocol.json", protocol)
    for group in args.groups:
        if group == "A0":
            raise ValueError("A0 is reused, not retrained")
        run_group(args, group, cfg, device, dataset, indexed, tr, va, record, expected)
    print(json.dumps(dict(event="requested_groups_complete", groups=args.groups)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["preflight", "train"])
    p.add_argument("--device", choices=["cuda"], default="cuda")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS)[1:])
    args = p.parse_args()
    {"preflight": preflight, "train": train}[args.command](args)
