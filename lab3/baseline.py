"""CIFAR-10 基础实验：固定划分、从头训练 CNN、独立评价。"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parent
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
DEFAULTS = dict(seed=42, epochs=60, batch_size=128, learning_rate=0.001,
                weight_decay=0.0001, validation_per_class=500, num_workers=0,
                cpu_threads=4, augmentation="none", scheduler="none",
                early_stopping=False, pretrained=False, precision="float32",
                normalization_mean=[0.5] * 3, normalization_std=[0.5] * 3,
                selection="maximum validation accuracy; ties: minimum validation CE; exact ties: earlier epoch")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(DEFAULTS["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in (32, 64, 128):
            for _ in range(2):
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ])
                in_channels = out_channels
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        # 空间均值实现全局平均池化，使用确定性归约。
        return self.classifier(self.features(x).mean(dim=(2, 3)))


def normalize_images(images):
    # 等价于 ToTensor() 后 Normalize((.5,.5,.5), (.5,.5,.5))。
    x = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).float()
    return x.div_(255.0).sub_(0.5).div_(0.5)


def make_split(labels, seed=42, validation_per_class=500):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in range(10):
        indices = rng.permutation(np.flatnonzero(labels == c))
        if len(indices) <= validation_per_class:
            raise ValueError(f"Class {c}: insufficient examples")
        val.extend(indices[:validation_per_class])
        train.extend(indices[validation_per_class:])
    return rng.permutation(train).astype(np.int64), rng.permutation(val).astype(np.int64)


def load_training(data_dir):
    raw = CIFAR10(str(data_dir), train=True, download=False)
    assert raw.classes == CLASSES and len(raw) == 50000
    labels = np.asarray(raw.targets, dtype=np.int64)
    assert np.array_equal(np.bincount(labels), np.full(10, 5000))
    return TensorDataset(normalize_images(raw.data), torch.from_numpy(labels)), labels


def loader_for(dataset, indices, batch_size, device, shuffle=False, seed=42):
    subset = dataset if indices is None else Subset(dataset, np.asarray(indices).tolist())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=device.type == "cuda", generator=generator,
                      drop_last=False)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.int64)
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
    return dict(loss=total_loss.item() / n, accuracy=total_correct.item() / n, count=n)


@torch.inference_mode()
def evaluate_loader(model, loader, device, collect=False):
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.int64)
    probabilities, targets = [], []
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += nn.functional.cross_entropy(logits, y, reduction="sum")
        total_correct += (logits.argmax(1) == y).sum()
        n += len(y)
        if collect:
            probabilities.append(logits.softmax(1).cpu().numpy())
            targets.append(y.cpu().numpy())
    result = dict(loss=total_loss.item() / n, accuracy=total_correct.item() / n, count=n)
    if collect:
        result["probabilities"] = np.concatenate(probabilities)
        result["targets"] = np.concatenate(targets)
    return result


def confusion_matrix(targets, predictions):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if targets.shape != predictions.shape or targets.ndim != 1:
        raise ValueError("Targets/predictions must be equal-length vectors")
    if np.any((targets < 0) | (targets > 9) | (predictions < 0) | (predictions > 9)):
        raise ValueError("Class indices must be 0..9")
    return np.bincount(targets * 10 + predictions, minlength=100).reshape(10, 10)


def get_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; do not silently switch to CPU")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                        else "cpu" if requested == "auto" else requested)


def environment(device):
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__,
                cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                cpu_threads=torch.get_num_threads(), deterministic_algorithms=True,
                cudnn_benchmark=False, tf32=False,
                timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat())


def prepare(args):
    args.data_dir.mkdir(parents=True, exist_ok=True)
    dataset = CIFAR10(str(args.data_dir), train=True, download=True)
    # torchvision 校验官方文件 MD5；此处再保存 SHA-256 便于追溯。
    records = {}
    for path in sorted((args.data_dir / "cifar-10-batches-py").iterdir()):
        if path.is_file():
            records[path.name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    write_json(args.data_dir / "manifest.json", dict(source=dataset.url, files=records,
                                                    train_count=len(dataset), classes=dataset.classes))
    print(json.dumps(dict(event="data_ready", count=len(dataset), classes=dataset.classes)), flush=True)


def train(args):
    run = args.run_dir
    if (run / "config.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run}")
    run.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = get_device(args.device)
    dataset, labels = load_training(args.data_dir)
    train_indices, val_indices = make_split(labels, args.seed)
    np.savez_compressed(run / "split_indices.npz", train_indices=train_indices, val_indices=val_indices)
    config = {**DEFAULTS, "seed": args.seed, "epochs": args.epochs,
              "model": "6x Conv-BN-ReLU; channels 32/64/128; 3x MaxPool; GAP; Linear(128,10)",
              "optimizer": "AdamW", "optimizer_betas": [0.9, 0.999], "optimizer_eps": 1e-8,
              "weight_decay_applies_to": "all trainable parameters, including BatchNorm and classifier bias",
              "train_count": len(train_indices), "val_count": len(val_indices), "test_count": 10000,
              "split_sha256": sha256(run / "split_indices.npz"), "source_sha256": sha256(__file__),
              "metric_semantics": {"train": "online predictions before each batch update; BN train mode; no augmentation",
                                   "val": "fixed model after each epoch; BN eval mode",
                                   "loss": "sample-mean cross entropy; excludes decoupled AdamW weight decay",
                                   "train_eval": "selected checkpoint, fixed weights, BN eval mode"}}
    write_json(run / "config.json", config)
    shutil.copy2(__file__, run / "baseline_source.py")
    write_json(run / "environment.json", environment(device))
    model = BaselineCNN().to(device)
    (run / "model.txt").write_text(str(model), encoding="utf-8")
    write_json(run / "model_info.json", dict(parameters=sum(p.numel() for p in model.parameters()),
                                             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    train_loader = loader_for(dataset, train_indices, config["batch_size"], device, True, args.seed)
    val_loader = loader_for(dataset, val_indices, config["batch_size"], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    best_key = (-float("inf"), -float("inf"))
    best_epoch = None
    synchronize(device)
    wall_start = time.perf_counter()
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
              "learning_rate", "train_seconds", "val_seconds", "elapsed_seconds", "selected"]
    print(json.dumps(dict(event="training_start", device=str(device), epochs=args.epochs,
                          train=len(train_indices), validation=len(val_indices))), flush=True)
    with (run / "history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            synchronize(device)
            start = time.perf_counter()
            training = train_epoch(model, train_loader, optimizer, device)
            synchronize(device)
            train_seconds = time.perf_counter() - start
            start = time.perf_counter()
            validation = evaluate_loader(model, val_loader, device)
            synchronize(device)
            val_seconds = time.perf_counter() - start
            key = (validation["accuracy"], -validation["loss"])
            selected = key > best_key
            if selected:
                best_key, best_epoch = key, epoch
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch,
                                val_accuracy=validation["accuracy"], val_loss=validation["loss"],
                                config=config), run / "best.pt")
            row = dict(epoch=epoch, train_loss=training["loss"], train_accuracy=training["accuracy"],
                       val_loss=validation["loss"], val_accuracy=validation["accuracy"],
                       learning_rate=optimizer.param_groups[0]["lr"], train_seconds=train_seconds,
                       val_seconds=val_seconds, elapsed_seconds=time.perf_counter() - wall_start,
                       selected=int(selected))
            writer.writerow(row)
            f.flush()
            print(json.dumps(row), flush=True)
    synchronize(device)
    duration = time.perf_counter() - wall_start
    torch.save(dict(state_dict=model.state_dict(), epoch=args.epochs, config=config), run / "last.pt")
    write_json(run / "training_summary.json", dict(completed_epochs=args.epochs, best_epoch=best_epoch,
                                                 training_wall_seconds=duration,
                                                 timing_scope="training + per-epoch validation + best-checkpoint/history writes; excludes data loading, environment setup, final checkpoint save, final evaluations and plots"))
    print(json.dumps(dict(event="training_complete", best_epoch=best_epoch, seconds=duration)), flush=True)


def write_confusion(path, matrix):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/predicted"] + CLASSES)
        for name, row in zip(CLASSES, matrix):
            writer.writerow([name] + row.tolist())


def evaluate(args):
    run = args.run_dir
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    device = get_device(args.device)
    model = BaselineCNN().to(device)
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    dataset, _ = load_training(args.data_dir)
    split = np.load(run / "split_indices.npz")
    train_loader = loader_for(dataset, split["train_indices"], config["batch_size"], device)
    val_loader = loader_for(dataset, split["val_indices"], config["batch_size"], device)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_result = evaluate_loader(model, train_loader, device)
    val_result = evaluate_loader(model, val_loader, device, collect=True)
    val_prob, val_targets = val_result.pop("probabilities"), val_result.pop("targets")
    raw_test = CIFAR10(str(args.data_dir), train=False, download=False)
    assert raw_test.classes == CLASSES and len(raw_test) == 10000
    test_set = TensorDataset(normalize_images(raw_test.data), torch.tensor(raw_test.targets))
    test_loader = loader_for(test_set, None, config["batch_size"], device)
    synchronize(device)
    start = time.perf_counter()
    test_result = evaluate_loader(model, test_loader, device, collect=True)
    synchronize(device)
    test_seconds = time.perf_counter() - start
    probabilities, targets = test_result.pop("probabilities"), test_result.pop("targets")
    predictions = probabilities.argmax(1)
    matrix = confusion_matrix(targets, predictions)
    if not all(torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise AssertionError("Evaluation modified model weights or BatchNorm buffers")
    assert abs(val_result["accuracy"] - checkpoint["val_accuracy"]) < 1e-12
    np.savez_compressed(run / "test_predictions.npz", indices=np.arange(len(targets)), targets=targets,
                        predictions=predictions, probabilities=probabilities)
    np.savez_compressed(run / "validation_predictions.npz", indices=split["val_indices"],
                        targets=val_targets, predictions=val_prob.argmax(1), probabilities=val_prob)
    write_confusion(run / "confusion_counts.csv", matrix)
    write_confusion(run / "confusion_row_normalized.csv", matrix / matrix.sum(1, keepdims=True))
    with (run / "per_class.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class", "count", "correct", "accuracy"])
        for c, name in enumerate(CLASSES):
            writer.writerow([c, name, int(matrix[c].sum()), int(matrix[c, c]), matrix[c, c] / matrix[c].sum()])
    with (run / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_index", "true_class", "predicted_class", "confidence", "correct"])
        for i in range(len(targets)):
            writer.writerow([i, CLASSES[targets[i]], CLASSES[predictions[i]],
                             float(probabilities[i, predictions[i]]), int(targets[i] == predictions[i])])
    summary = dict(selected_epoch=checkpoint["epoch"], train_eval=train_result, validation=val_result,
                   test=test_result, test_inference_seconds=test_seconds,
                   inference_timing_scope="one pass, including CPU DataLoader, device transfers, CE/accuracy/softmax and probability collection; excludes dataset construction and export",
                   correct_test=int(np.trace(matrix)), errors_test=int(matrix.sum() - np.trace(matrix)),
                   per_class_accuracy=(np.diag(matrix) / matrix.sum(1)).tolist(),
                   weights_unchanged_during_evaluation=True, checkpoint_sha256=sha256(run / "best.pt"))
    write_json(run / "metrics.json", summary)
    print(json.dumps(summary), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "train", "evaluate"])
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "baseline_seed42")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.command](args)
