"""必要正确性检查；不比较模型或搜索超参数。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from baseline import (ROOT, CLASSES, BaselineCNN, confusion_matrix, evaluate_loader,
                      get_device, load_training, loader_for, make_split,
                      normalize_images, seed_everything, sha256, write_json)


def preflight(args):
    seed_everything(42)
    device = get_device(args.device)
    dataset, labels = load_training(args.data_dir)
    tr, va = make_split(labels)
    tr2, va2 = make_split(labels)
    assert len(tr) == 45000 and len(va) == 5000
    assert np.intersect1d(tr, va).size == 0
    assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(50000))
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2)
    assert np.array_equal(np.bincount(labels[tr]), np.full(10, 4500))
    assert np.array_equal(np.bincount(labels[va]), np.full(10, 500))
    extremes = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    extremes[1] = 255
    transformed = normalize_images(extremes)
    assert transformed.shape == (2, 3, 32, 32)
    assert transformed[0].eq(-1).all() and transformed[1].eq(1).all()
    # 由人工可核对的小例子检查矩阵方向与按真实类别统计。
    cm = confusion_matrix(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0]))
    assert cm[0, 0] == cm[0, 1] == cm[1, 1] == cm[2, 0] == 1
    assert cm.sum() == 4 and np.trace(cm) == 2
    model = BaselineCNN().to(device)
    indices = tr[:64]
    x, y = dataset.tensors[0][indices].to(device), dataset.tensors[1][indices].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    model.train()
    first_loss, last_loss = None, None
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        assert logits.shape == (64, 10)
        loss = nn.functional.cross_entropy(logits, y)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()
    model.eval()
    with torch.inference_mode():
        accuracy = (model(x).argmax(1) == y).float().mean().item()
    assert last_loss < first_loss * 0.2 and accuracy >= 0.95
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    result = evaluate_loader(model, loader_for(dataset, indices, 32, device), device)
    assert result["count"] == 64
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="passed", split_disjoint=True, split_complete=True,
                  split_reproducible=True, train_per_class=4500, validation_per_class=500,
                  normalization_extremes=[-1, 1], confusion_orientation="rows=true, columns=predicted",
                  tiny_fit=dict(samples=64, steps=100, initial_loss=first_loss,
                                final_online_loss=last_loss, eval_accuracy=accuracy),
                  evaluation_preserves_parameters_and_bn_buffers=True,
                  note="64-sample fit is an implementation check; its weights are discarded and never used in baseline training")
    write_json(args.output, report)
    print(json.dumps(report), flush=True)


def verify(args):
    run = args.run_dir
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    with (run / "history.csv").open(encoding="utf-8") as f:
        history = list(csv.DictReader(f))
    assert len(history) == config["epochs"]
    assert [int(r["epoch"]) for r in history] == list(range(1, config["epochs"] + 1))
    selected = max(history, key=lambda r: (float(r["val_accuracy"]), -float(r["val_loss"]), -int(r["epoch"])))
    assert int(selected["epoch"]) == metrics["selected_epoch"]
    assert all(float(r["learning_rate"]) == config["learning_rate"] for r in history)
    assert sha256(run / "best.pt") == metrics["checkpoint_sha256"]
    assert sha256(run / "split_indices.npz") == config["split_sha256"]
    with np.load(run / "test_predictions.npz") as pred:
        y, p, prob = pred["targets"], pred["predictions"], pred["probabilities"]
        assert np.array_equal(pred["indices"], np.arange(10000))
        assert prob.shape == (10000, 10) and np.isfinite(prob).all()
        assert np.allclose(prob.sum(1), 1, atol=1e-6)
        assert np.array_equal(p, prob.argmax(1))
        # 独立方式逐个累计，避免只重复调用被测的矩阵函数。
        independent_cm = np.zeros((10, 10), dtype=np.int64)
        for actual, predicted in zip(y, p):
            independent_cm[actual, predicted] += 1
        assert np.array_equal(independent_cm.sum(1), np.full(10, 1000))
        assert np.trace(independent_cm) == metrics["correct_test"]
        assert np.mean(y == p) == metrics["test"]["accuracy"]
        independent_ce = float(-np.log(prob[np.arange(len(y)), y].astype(np.float64)).mean())
        assert abs(independent_ce - metrics["test"]["loss"]) < 1e-5
    with (run / "confusion_counts.csv").open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    saved_cm = np.array([r[1:] for r in rows[1:]], dtype=np.int64)
    assert np.array_equal(saved_cm, independent_cm)
    with (run / "per_class.csv").open(encoding="utf-8") as f:
        per_class = list(csv.DictReader(f))
    assert len(per_class) == 10
    assert sum(int(r["correct"]) for r in per_class) == metrics["correct_test"]
    assert all(int(r["count"]) == 1000 for r in per_class)
    with (run / "test_predictions.csv").open(encoding="utf-8") as f:
        prediction_rows = list(csv.DictReader(f))
    assert len(prediction_rows) == 10000
    for i, row in enumerate(prediction_rows):
        assert int(row["test_index"]) == i
        assert row["true_class"] == CLASSES[y[i]]
        assert row["predicted_class"] == CLASSES[p[i]]
        assert int(row["correct"]) == int(y[i] == p[i])
        assert abs(float(row["confidence"]) - float(prob[i, p[i]])) < 1e-12
    # 新进程中重载检查点，并用真实测试图像复核全部概率。
    seed_everything(config["seed"])
    device = get_device(args.device)
    model = BaselineCNN().to(device)
    state = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    from torchvision.datasets import CIFAR10
    from torch.utils.data import TensorDataset
    raw = CIFAR10(str(args.data_dir), train=False, download=False)
    dataset = TensorDataset(normalize_images(raw.data), torch.tensor(raw.targets))
    rerun = evaluate_loader(model, loader_for(dataset, None, config["batch_size"], device), device, collect=True)
    with np.load(run / "test_predictions.npz") as pred:
        max_delta = float(np.max(np.abs(rerun["probabilities"] - pred["probabilities"])))
        assert np.array_equal(rerun["probabilities"].argmax(1), pred["predictions"])
        assert np.allclose(rerun["probabilities"], pred["probabilities"], rtol=0, atol=1e-6)
    result = dict(status="passed", history_complete=True, selection_rule_verified=True,
                  prediction_rows=10000, per_class_count=1000, confusion_recomputed=True,
                  cross_entropy_recomputed=independent_ce, checkpoint_reloaded_in_separate_process=True,
                  all_predictions_reproduced=True, maximum_probability_difference=max_delta,
                  limitation="Full 60-epoch training was run once; not repeated in a clean independent environment")
    write_json(run / "verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["preflight", "verify"])
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "baseline_seed42")
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "checks" / "preflight.json")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()
    {"preflight": preflight, "verify": verify}[args.command](args)
