"""Compare AFC with the same training protocol and label smoothing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import augmentation_ablation as ablation
import baseline as base


AFC = ablation.OUT / "AFC"
OUTPUT = base.ROOT / "outputs" / "loss_label_smoothing_seed42"
SMOOTHING = 0.1
HIGH_CONFIDENCE = 0.90
CALIBRATION_BINS = 15
TRACE_KEYS = ("sample_order_sha256", "flip_sha256", "crop_sha256")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def train_epoch(model, loader, optimizer, device, augment, count):
    model.train()
    parameters, trace = augment.epoch_parameters(count)
    loss_sum = torch.zeros((), device=device)
    order = hashlib.sha256()
    seen = 0
    for images, labels, indices in loader:
        order.update(indices.numpy().astype("<i8", copy=False).tobytes())
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        end = seen + len(labels)
        images = augment.apply(images, {key: value[seen:end] for key, value in parameters.items()})
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images), labels, label_smoothing=SMOOTHING)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach() * len(labels)
        seen = end
    assert seen == count and bool(torch.isfinite(loss_sum))
    trace["sample_order_sha256"] = order.hexdigest()
    return loss_sum.item() / count, trace


def summarize(probabilities, targets):
    probabilities = probabilities.astype(np.float64)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    high = confidence >= HIGH_CONFIDENCE
    bin_ids = np.minimum((confidence * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1)
    calibration_error = 0.0
    for bin_id in range(CALIBRATION_BINS):
        selected = bin_ids == bin_id
        if np.any(selected):
            calibration_error += abs(np.sum(correct[selected]) - np.sum(confidence[selected])) / len(targets)
    return {
        "samples": len(targets),
        "accuracy": float(correct.mean()),
        "hard_label_cross_entropy": float(-np.log(probabilities[np.arange(len(targets)), targets]).mean()),
        "expected_calibration_error": float(calibration_error),
        "mean_confidence": float(confidence.mean()),
        "wrong": int((~correct).sum()),
        "high_confidence_predictions": int(high.sum()),
        "high_confidence_wrong": int((high & ~correct).sum()),
        "high_confidence_correct": int((high & correct).sum()),
    }


def run(args):
    cfg, device, dataset, indexed, train_indices, val_indices = ablation.load_protocol(args)
    afc_config = read_json(AFC / "config.json")
    afc_metrics = read_json(AFC / "metrics.json")
    afc_predictions = AFC / "validation_predictions.npz"
    trace_file = AFC / "augmentation_trace.jsonl"
    expected_traces = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    assert len(expected_traces) == cfg["epochs"] and afc_metrics["selected_epoch"] == 47
    for field in ("seed", "epochs", "batch_size", "learning_rate", "weight_decay", "optimizer_betas",
                  "optimizer_eps", "split_sha256", "selection"):
        assert afc_config[field] == cfg[field]
    assert afc_config["source_sha256"] == base.sha256(Path(ablation.__file__))
    assert afc_metrics["checkpoint_sha256"] == base.sha256(AFC / "best.pt")
    base.seed_everything(cfg["seed"])
    model = base.BaselineCNN().to(device)
    assert ablation.state_hash(model.state_dict()) == afc_config["initial_state_sha256"]
    train_loader = base.loader_for(indexed, train_indices, cfg["batch_size"], device, True, cfg["seed"])
    val_loader = base.loader_for(dataset, val_indices, cfg["batch_size"], device)
    optimizer = ablation.optimizer_for(model, cfg)
    augment = ablation.Augmentation(ablation.GROUPS["AFC"], device)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    base.write_json(out / "config.json", {
        "control": "AFC epoch 47",
        "AFC_checkpoint_sha256": afc_metrics["checkpoint_sha256"],
        "AFC_validation_predictions_sha256": base.sha256(afc_predictions),
        "AFC_trace_sha256": base.sha256(trace_file),
        "training_loss_label_smoothing": SMOOTHING,
        "validation_loss_label_smoothing": 0.0,
        "high_confidence_threshold": HIGH_CONFIDENCE,
        "equal_width_calibration_bins": CALIBRATION_BINS,
        "training": afc_config,
        "loss_ablation_source_sha256": base.sha256(Path(__file__)),
        "test_evaluated": False,
    })
    base.write_json(out / "environment.json", base.environment(device))
    best_key = (-1.0, -float("inf"))
    start = time.perf_counter()
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as history:
        writer = csv.writer(history)
        writer.writerow(["epoch", "training_smoothed_loss", "validation_accuracy", "validation_hard_ce", *TRACE_KEYS])
        for epoch, expected in enumerate(expected_traces, start=1):
            cost, trace = train_epoch(model, train_loader, optimizer, device, augment, len(train_indices))
            assert expected["epoch"] == epoch and all(trace[key] == expected[key] for key in TRACE_KEYS)
            validation = base.evaluate_loader(model, val_loader, device)
            key = (validation["accuracy"], -validation["loss"])
            if key > best_key:
                best_key = key
                torch.save({"epoch": epoch, "state_dict": model.state_dict(), "validation": validation}, out / "best.pt")
            writer.writerow([epoch, cost, *[validation[k] for k in ("accuracy", "loss")], *[trace[k] for k in TRACE_KEYS]])
            history.flush()
            if epoch == 1 or epoch % 5 == 0:
                print(f"Epoch {epoch}/{cfg['epochs']} validation accuracy {key[0]:.2%}", flush=True)
    elapsed = time.perf_counter() - start
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    result = base.evaluate_loader(model, val_loader, device, collect=True)
    targets = result["targets"]
    probabilities = result["probabilities"]
    with np.load(afc_predictions) as saved:
        assert np.array_equal(saved["targets"], targets) and np.array_equal(saved["indices"], val_indices)
        afc = summarize(saved["probabilities"], targets)
    smoothed = summarize(probabilities, targets)
    assert checkpoint["validation"]["accuracy"] == result["accuracy"] == smoothed["accuracy"]
    assert abs(smoothed["hard_label_cross_entropy"] - result["loss"]) < 1e-6
    assert abs(afc["hard_label_cross_entropy"] - afc_metrics["validation"]["loss"]) < 1e-6
    np.savez_compressed(out / "validation_predictions.npz", indices=val_indices, targets=targets,
                        probabilities=probabilities, predictions=probabilities.argmax(1))
    comparison = {
        "AFC": {"epoch": afc_metrics["selected_epoch"], **afc},
        "AFC_label_smoothing": {"epoch": checkpoint["epoch"], **smoothed},
        "training_seconds": elapsed,
        "matching_sample_order_and_augmentation_epochs": len(expected_traces),
        "best_checkpoint_sha256": base.sha256(out / "best.pt"),
        "test_evaluated": False,
    }
    base.write_json(out / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--data-dir", type=Path, default=base.ROOT / "data")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    run(parser.parse_args())
