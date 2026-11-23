"""Evaluate archived or validation-selected retrained SE-AFC against locked AFC."""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import augmentation_ablation as aug
import baseline as base
import final_test_afc
import se_ablation as se


DEST = base.ROOT / "outputs" / "final_se_afc_epoch48"
SE_EPOCH = 48
SE_SHA256 = "d91594f0d1ea7b8f1459cb315e127b580b78e08611e8ea040c075b1d0582a21c"
TEST_COUNT = 10000
AFC = final_test_afc.DEST


def artifact_hashes(source):
    frozen = se.read_frozen(source)
    for run in (source, AFC):
        for name in ("best.pt", "config.json", "metrics.json"):
            path = run / name
            frozen[path.relative_to(base.ROOT).as_posix()] = base.sha256(path)
    for name in ("test_predictions.npz", "lock.json", "evaluation_environment.json"):
        path = AFC / name
        frozen[path.relative_to(base.ROOT).as_posix()] = base.sha256(path)
    return frozen


def lock_weights(run, source, mode):
    prior = se.read(source / "metrics.json")
    training = se.read(source / "config.json")
    verification = se.read(source / "verification.json")
    assert verification["status"] == "passed" and verification["selected_epoch_verified"]
    assert training["group"] == "SE-AFC" and training["batch_size"] == 128
    assert not prior["test_evaluated"]
    epoch = prior["selected_epoch"]
    checkpoint_path = source / "best.pt"
    digest = base.sha256(checkpoint_path)
    assert digest == prior["checkpoint_sha256"]
    if mode == "archived":
        assert (epoch, digest) == (SE_EPOCH, SE_SHA256), "Archived SE checkpoint differs from the original lock"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == epoch and checkpoint["config"] == training
    assert base.sha256(source / "se_ablation.py") == training["source_sha256"]
    assert checkpoint["val_accuracy"] == prior["validation"]["accuracy"]
    assert checkpoint["val_loss"] == prior["validation"]["loss"]
    assert base.sha256(base.__file__) == se.read(se.AFC / "config.json")["baseline_source_sha256"]
    afc_lock = se.read(AFC / "lock.json")
    assert afc_lock["epoch"] == 47 and afc_lock["group"] == "AFC"
    assert base.sha256(AFC / "best.pt") == afc_lock["checkpoint_sha256"] == final_test_afc.LOCKED_SHA256
    se.check_frozen(source)
    frozen = artifact_hashes(source)
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint_path, run / "best.pt")
    assert base.sha256(run / "best.pt") == digest
    shutil.copy2(source / "config.json", run / "training_config.json")
    shutil.copy2(source / "metrics.json", run / "validation_stage_metrics.json")
    shutil.copy2(__file__, run / "final_evaluation_source.py")
    base.write_json(run / "frozen_inputs.json", frozen)
    base.write_json(run / "lock.json", dict(
        group="SE-AFC", mode=mode, epoch=epoch, checkpoint_sha256=digest,
        locked_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        selection="Validation-selected checkpoint locked before test evaluation",
        validation_accuracy=prior["validation"]["accuracy"], validation_loss=prior["validation"]["loss"],
        comparison_checkpoint_sha256=afc_lock["checkpoint_sha256"],
        evaluation_source_sha256=base.sha256(__file__), training_performed=False,
    ))
    print(f"Locked {mode} SE-AFC epoch {epoch} before evaluation: {digest}", flush=True)


def evaluate_locked(run, data_dir):
    lock = se.read(run / "lock.json")
    mode = lock.get("mode", "archived")
    assert mode in ("archived", "reproduced")
    if mode == "archived":
        assert (lock["epoch"], lock["checkpoint_sha256"]) == (SE_EPOCH, SE_SHA256)
    assert base.sha256(run / "best.pt") == lock["checkpoint_sha256"]
    assert base.sha256(AFC / "best.pt") == lock["comparison_checkpoint_sha256"] == final_test_afc.LOCKED_SHA256
    base.seed_everything(42)
    device = base.get_device("cuda")
    env = base.environment(device)
    afc_env = se.read(AFC / "evaluation_environment.json")
    for key in ("torch", "torchvision", "numpy", "cuda_runtime", "cudnn", "gpu", "cpu_threads"):
        assert env[key] == afc_env[key], f"Evaluation environment differs in {key}"

    model, checkpoint = se.load_model(run, device)
    assert checkpoint["epoch"] == lock["epoch"]
    assert checkpoint["val_accuracy"] == lock["validation_accuracy"]
    assert checkpoint["val_loss"] == lock["validation_loss"]
    initial_state = aug.state_hash(model.state_dict())
    manifest = se.read(data_dir / "manifest.json")
    test_hash = base.sha256(data_dir / "cifar-10-batches-py" / "test_batch")
    assert test_hash == manifest["files"]["test_batch"]["sha256"]
    raw = base.CIFAR10(str(data_dir), train=False, download=False)
    assert raw.classes == base.CLASSES and len(raw) == TEST_COUNT
    targets = np.asarray(raw.targets, dtype=np.int64)
    assert np.array_equal(np.bincount(targets), np.full(10, 1000))
    dataset = base.TensorDataset(base.normalize_images(raw.data), torch.from_numpy(targets))
    loader = base.loader_for(dataset, None, 128, device)
    base.synchronize(device)
    started = time.perf_counter()
    result = base.evaluate_loader(model, loader, device, collect=True)
    base.synchronize(device)
    seconds = time.perf_counter() - started
    probabilities = result.pop("probabilities")
    assert np.array_equal(result.pop("targets"), targets)
    assert result["count"] == TEST_COUNT
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1, atol=1e-6)
    assert initial_state == aug.state_hash(model.state_dict())
    assert base.sha256(run / "best.pt") == lock["checkpoint_sha256"]
    return result, targets, probabilities, seconds, env, test_hash


def evaluate(args):
    run = args.run_dir.resolve()
    lock_weights(run, args.training_run.resolve(), args.mode)
    lock = se.read(run / "lock.json")
    result, targets, probabilities, seconds, env, test_hash = evaluate_locked(run, args.data_dir)
    predictions = probabilities.argmax(1)
    indices = np.arange(TEST_COUNT)
    matrix = base.confusion_matrix(targets, predictions)
    correct = int(np.trace(matrix))
    assert correct / TEST_COUNT == result["accuracy"]
    np.savez_compressed(run / "test_predictions.npz", indices=indices, targets=targets,
                        predictions=predictions, probabilities=probabilities)
    se.csv_write(run / "test_predictions.csv", [dict(
        test_index=int(i), true_class=base.CLASSES[y], predicted_class=base.CLASSES[p],
        confidence=float(probabilities[i, p]), correct=int(y == p),
    ) for i, y, p in zip(indices, targets, predictions)])
    base.write_confusion(run / "confusion_counts.csv", matrix)
    base.write_confusion(run / "confusion_row_normalized.csv", matrix / 1000)
    metrics = dict(group="SE-AFC", selected_epoch=lock["epoch"], test=result, correct_test=correct,
                   errors_test=TEST_COUNT - correct, per_class_accuracy=(matrix.diagonal() / 1000).tolist(),
                   checkpoint_sha256=lock["checkpoint_sha256"], test_batch_sha256=test_hash, test_evaluated=True,
                   test_inference_seconds=seconds, inference_timing_scope="Single full test pass including loading, transfer, CE/accuracy and probability collection",
                   evaluation_preserves_parameters_and_bn_buffers=True)
    base.write_json(run / "evaluation_environment.json", env)
    base.write_json(run / "metrics.json", metrics)
    se.check_frozen(run)
    print(metrics, flush=True)


def cross_entropy(probabilities, targets):
    return -np.log(probabilities[np.arange(len(targets)), targets].astype(np.float64))


def verify_and_compare(args):
    run = args.run_dir.resolve()
    metrics = se.read(run / "metrics.json")
    lock = se.read(run / "lock.json")
    assert metrics["selected_epoch"] == lock["epoch"] and metrics["checkpoint_sha256"] == lock["checkpoint_sha256"]
    afc = se.read(AFC / "metrics.json")
    result, targets, reloaded, _, _, test_hash = evaluate_locked(run, args.data_dir)
    assert test_hash == metrics["test_batch_sha256"] and result == metrics["test"]
    with np.load(run / "test_predictions.npz") as saved, np.load(AFC / "test_predictions.npz") as old:
        assert np.array_equal(saved["indices"], old["indices"]) and np.array_equal(saved["indices"], np.arange(TEST_COUNT))
        assert np.array_equal(saved["targets"], targets) and np.array_equal(old["targets"], targets)
        assert np.array_equal(saved["probabilities"], reloaded)
        p_new, p_old = saved["predictions"], old["predictions"]
        assert np.array_equal(p_new, reloaded.argmax(1)) and np.array_equal(p_old, old["probabilities"].argmax(1))
        ce_new = cross_entropy(reloaded, targets)
        ce_old = cross_entropy(old["probabilities"], targets)
    assert abs(float(ce_new.mean()) - metrics["test"]["loss"]) < 1e-5
    assert abs(float(ce_old.mean()) - afc["test"]["loss"]) < 1e-5
    old_ok, new_ok = p_old == targets, p_new == targets
    repaired, introduced = ~old_ok & new_ok, old_ok & ~new_ok
    assert int(old_ok.sum()) == afc["correct_test"]
    assert int(new_ok.sum()) == metrics["correct_test"]
    net = int(repaired.sum() - introduced.sum())
    assert net == int(new_ok.sum() - old_ok.sum())
    old_cm = base.confusion_matrix(targets, p_old)
    new_cm = base.confusion_matrix(targets, p_new)
    per_class = []
    for c, label in enumerate(base.CLASSES):
        actual = targets == c
        a, b = int(old_cm[c, c]), int(new_cm[c, c])
        per_class.append(dict(class_name=label, count=1000, afc_correct=a, se_correct=b,
            afc_accuracy=a / 1000, se_accuracy=b / 1000, delta_pp=(b - a) / 10,
            repaired=int((actual & repaired).sum()), introduced=int((actual & introduced).sum())))
    se.csv_write(run / "per_class_comparison.csv", per_class)
    changed = p_old != p_new
    if changed.any():
        se.csv_write(run / "paired_test_changes.csv", [dict(
            test_index=int(i), true_class=base.CLASSES[targets[i]], afc_prediction=base.CLASSES[p_old[i]],
            se_prediction=base.CLASSES[p_new[i]], change=("repaired" if repaired[i] else
                "introduced_error" if introduced[i] else "both_wrong_different_class"),
        ) for i in np.flatnonzero(changed)])
    strata = {"both_correct": old_ok & new_ok, "repaired": repaired,
              "introduced": introduced, "both_wrong": ~old_ok & ~new_ok}
    loss_by_stratum = {}
    for name, mask in strata.items():
        if mask.any():
            loss_by_stratum[name] = dict(count=int(mask.sum()), afc_mean_ce=float(ce_old[mask].mean()),
                se_mean_ce=float(ce_new[mask].mean()), contribution_to_mean_ce_change=float((ce_new[mask] - ce_old[mask]).sum() / TEST_COUNT))
    comparison = dict(afc=afc["test"], se_afc=metrics["test"], accuracy_delta_pp=net / 100,
        ce_delta=metrics["test"]["loss"] - afc["test"]["loss"], net_correct=net,
        repaired=int(repaired.sum()), introduced=int(introduced.sum()),
        both_correct=int((old_ok & new_ok).sum()), both_wrong=int((~old_ok & ~new_ok).sum()),
        changed_predictions=int(changed.sum()), per_class=per_class, loss_by_stratum=loss_by_stratum,
        cat_as_dog=dict(afc=int(old_cm[3, 5]), se_afc=int(new_cm[3, 5])),
        dog_as_cat=dict(afc=int(old_cm[5, 3]), se_afc=int(new_cm[5, 3])))
    base.write_json(run / "comparison_analysis.json", comparison)
    check = dict(status="passed", reloaded_probabilities_and_predictions_identical=True,
                 test_indices_and_labels_match_afc=True, independent_se_ce=float(ce_new.mean()),
                 independent_afc_ce=float(ce_old.mean()), original_files_unchanged=se.check_frozen(run),
                 checkpoint_sha256=lock["checkpoint_sha256"], test_count=TEST_COUNT)
    base.write_json(run / "verification.json", check)
    print(comparison, flush=True)
    print(check, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "verify"])
    parser.add_argument("--mode", choices=["archived", "reproduced"], default="archived",
                        help="Evaluate the historical checkpoint or this run's validation winner; verify uses the recorded mode.")
    parser.add_argument("--training-run", type=Path, default=se.OUT,
                        help="SE training outputs inside this checkout (evaluate only).")
    parser.add_argument("--run-dir", type=Path, default=DEST)
    parser.add_argument("--data-dir", type=Path, default=base.ROOT / "data")
    args = parser.parse_args()
    {"evaluate": evaluate, "verify": verify_and_compare}[args.command](args)
