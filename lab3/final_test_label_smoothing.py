"""Test archived or validation-selected retrained label smoothing against locked AFC."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil

import numpy as np
import torch

import augmentation_ablation as aug
import baseline as base
import final_test_afc
import loss_ablation as loss


SMOOTH_SHA = "b30e6e7bce03a9329e8677b48069f4cc2a616cd23dc55c8a8dd13283476b4987"
DEST = base.ROOT / "outputs" / "final_label_smoothing_epoch43"
AFC = final_test_afc.DEST
SOURCE = loss.OUTPUT


def lock_weights(run, source, mode):
    stage = loss.read_json(source / "comparison.json")
    config = loss.read_json(source / "config.json")
    verification = loss.read_json(source / "verification.json")
    afc_lock = loss.read_json(AFC / "lock.json")
    assert not stage["test_evaluated"]
    assert verification["status"] == "passed" and verification["independent_reloaded_validation_predictions_exact"]
    assert config["training_loss_label_smoothing"] == 0.1
    epoch = stage["AFC_label_smoothing"]["epoch"]
    checkpoint_path = source / "best.pt"
    digest = base.sha256(checkpoint_path)
    assert digest == stage["best_checkpoint_sha256"]
    if mode == "archived":
        assert (epoch, digest) == (43, SMOOTH_SHA), "Archived label-smoothing checkpoint differs from the original lock"
    else:
        assert verification["checkpoint_sha256"] == digest
    assert afc_lock["epoch"] == 47
    assert base.sha256(AFC / "best.pt") == afc_lock["checkpoint_sha256"] == final_test_afc.LOCKED_SHA256
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert cp["epoch"] == epoch and cp["validation"]["accuracy"] == stage["AFC_label_smoothing"]["accuracy"]
    frozen = {f.relative_to(base.ROOT).as_posix(): base.sha256(f) for f in (
        checkpoint_path, source / "comparison.json", AFC / "best.pt", AFC / "lock.json",
        AFC / "test_predictions.npz", AFC / "metrics.json", AFC / "evaluation_environment.json",
    )}
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint_path, run / "best.pt")
    assert base.sha256(run / "best.pt") == digest
    shutil.copy2(source / "config.json", run / "training_config.json")
    shutil.copy2(source / "comparison.json", run / "validation_stage_metrics.json")
    shutil.copy2(__file__, run / "final_evaluation_source.py")
    base.write_json(run / "lock.json", {
        "epoch": epoch, "group": "AFC label smoothing 0.1", "mode": mode, "checkpoint_sha256": digest,
        "afc_epoch": 47, "afc_checkpoint_sha256": afc_lock["checkpoint_sha256"],
        "locked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "selection": "Validation-selected checkpoint locked before test evaluation",
        "confidence_threshold": loss.HIGH_CONFIDENCE, "equal_width_calibration_bins": loss.CALIBRATION_BINS,
        "training_performed": False, "frozen_inputs": frozen,
    })
    print(f"Locked {mode} label smoothing epoch {epoch}: {digest}", flush=True)


def infer(run, data_dir):
    lock = loss.read_json(run / "lock.json")
    mode = lock.get("mode", "archived")
    assert mode in ("archived", "reproduced")
    if mode == "archived":
        assert (lock["epoch"], lock["checkpoint_sha256"]) == (43, SMOOTH_SHA)
    assert lock["checkpoint_sha256"] == base.sha256(run / "best.pt")
    assert lock["afc_checkpoint_sha256"] == base.sha256(AFC / "best.pt") == final_test_afc.LOCKED_SHA256
    for path, digest in lock["frozen_inputs"].items():
        assert base.sha256(base.ROOT / path) == digest
    base.seed_everything(42)
    device = base.get_device("cuda")
    env = base.environment(device)
    afc_env = loss.read_json(AFC / "evaluation_environment.json")
    for key in ("torch", "torchvision", "numpy", "cuda_runtime", "cudnn", "gpu", "cpu_threads"):
        assert env[key] == afc_env[key], key
    file = data_dir / "cifar-10-batches-py" / "test_batch"
    assert base.sha256(file) == loss.read_json(data_dir / "manifest.json")["files"]["test_batch"]["sha256"]
    raw = base.CIFAR10(str(data_dir), train=False, download=False)
    targets = np.asarray(raw.targets, dtype=np.int64)
    assert raw.classes == base.CLASSES and len(raw) == 10000
    assert np.array_equal(np.bincount(targets), np.full(10, 1000))
    dataset = base.TensorDataset(base.normalize_images(raw.data), torch.from_numpy(targets))
    model = base.BaselineCNN().to(device)
    cp = torch.load(run / "best.pt", weights_only=True, map_location=device)
    assert cp["epoch"] == lock["epoch"]
    model.load_state_dict(cp["state_dict"])
    state = aug.state_hash(model.state_dict())
    result = base.evaluate_loader(model, base.loader_for(dataset, None, 128, device), device, collect=True)
    assert state == aug.state_hash(model.state_dict())
    assert result["count"] == len(targets) and np.array_equal(result["targets"], targets)
    assert np.isfinite(result["probabilities"]).all() and np.allclose(result["probabilities"].sum(1), 1.0)
    return targets, result["probabilities"], env


def compare(old, new, targets, epoch):
    afc, smooth = loss.summarize(old, targets), loss.summarize(new, targets)
    masks = []
    for probabilities, metrics in ((old, afc), (new, smooth)):
        ok = probabilities.argmax(1) == targets
        confidence = probabilities.max(1).astype(np.float64)
        high = confidence >= loss.HIGH_CONFIDENCE
        edges = np.linspace(0, 1, loss.CALIBRATION_BINS + 1)
        real, _ = np.histogram(confidence[ok], bins=edges)
        estimated, _ = np.histogram(confidence, bins=edges, weights=confidence)
        assert np.isclose(np.abs(real - estimated).sum() / len(ok), metrics["expected_calibration_error"], atol=1e-12)
        masks.append((ok, high))
    old_ok, old_high = masks[0]
    new_ok, new_high = masks[1]
    old_high_wrong = old_high & ~old_ok
    return {"AFC_epoch47": afc, f"label_smoothing_epoch{epoch}": smooth, "paired_changes": {
        "all_corrected": int((~old_ok & new_ok).sum()),
        "all_introduced_errors": int((old_ok & ~new_ok).sum()),
        "old_high_wrong_now_correct": int((old_high_wrong & new_ok).sum()),
        "old_high_wrong_still_wrong_below_90": int((old_high_wrong & ~new_ok & ~new_high).sum()),
        "old_high_wrong_still_wrong_at_least_90": int((old_high_wrong & ~new_ok & new_high).sum()),
        "new_high_wrong": int((~old_high_wrong & ~new_ok & new_high).sum()),
    }}


def run(args):
    output = args.run_dir.resolve()
    if args.command == "evaluate":
        lock_weights(output, args.training_run.resolve(), args.mode)
    targets, new, env = infer(output, args.data_dir)
    predictions = new.argmax(1)
    path = output / "test_predictions.npz"
    with np.load(AFC / "test_predictions.npz") as saved:
        assert np.array_equal(saved["indices"], np.arange(len(targets))) and np.array_equal(saved["targets"], targets)
        old = saved["probabilities"]
    epoch = loss.read_json(output / "lock.json")["epoch"]
    comparison = compare(old, new, targets, epoch)
    afc = comparison["AFC_epoch47"]
    afc_saved = loss.read_json(AFC / "metrics.json")["test"]
    assert afc["accuracy"] == afc_saved["accuracy"] and abs(afc["hard_label_cross_entropy"] - afc_saved["loss"]) < 1e-6
    if args.command == "evaluate":
        np.savez_compressed(path, indices=np.arange(len(targets)), targets=targets, predictions=predictions, probabilities=new)
        base.write_json(output / "comparison.json", comparison)
        base.write_json(output / "evaluation_environment.json", env)
        print(comparison, flush=True)
    else:
        with np.load(path) as saved:
            assert np.array_equal(saved["probabilities"], new) and np.array_equal(saved["predictions"], predictions)
            assert np.array_equal(saved["targets"], targets) and np.array_equal(saved["indices"], np.arange(len(targets)))
        assert comparison == loss.read_json(output / "comparison.json")
        base.write_json(output / "verification.json", {
            "status": "passed", "test_count": len(targets), "matched_original_afc_indices_and_labels": True,
            "reloaded_predictions_and_probabilities_exact": True, "calibration_error_independently_recomputed": True,
        })
        print("Verified all 10,000 predictions and probabilities.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "verify"])
    parser.add_argument("--mode", choices=["archived", "reproduced"], default="archived",
                        help="Evaluate the historical checkpoint or this run's validation winner; verify uses the recorded mode.")
    parser.add_argument("--training-run", type=Path, default=SOURCE,
                        help="Label-smoothing training outputs inside this checkout (evaluate only).")
    parser.add_argument("--data-dir", type=Path, default=base.ROOT / "data")
    parser.add_argument("--run-dir", type=Path, default=DEST)
    run(parser.parse_args())
