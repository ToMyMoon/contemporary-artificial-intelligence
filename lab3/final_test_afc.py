"""锁定 AFC 第 47 轮，进行一次最终测试评价；不训练或选择其他模型。"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

import baseline as base

SOURCE = base.ROOT / "outputs" / "augmentation_seed42" / "AFC"
DEST = base.ROOT / "outputs" / "final_afc_epoch47"
LOCKED_SHA256 = "688fa02161b1f78345ba783b7c4766df26b7e30739af0eaf17757f9af8fd59a8"


def prepare_and_evaluate(args):
    if args.run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite final evaluation: {args.run_dir}")
    training = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    prior = json.loads((SOURCE / "metrics.json").read_text(encoding="utf-8"))
    assert training["group"] == "AFC"
    assert training["augmentation"] == dict(flip=True, crop=True, brightness=False)
    assert prior["selected_epoch"] == 47 and prior["validation"]["accuracy"] == .8806
    assert base.sha256(SOURCE / "best.pt") == prior["checkpoint_sha256"] == LOCKED_SHA256
    assert base.sha256(base.ROOT / "baseline.py") == training["baseline_source_sha256"]
    checkpoint = torch.load(SOURCE / "best.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 47 and checkpoint["val_accuracy"] == .8806
    args.run_dir.mkdir(parents=True)
    for name in ["best.pt", "config.json", "history.csv", "split_indices.npz", "training_summary.json",
                 "environment.json", "baseline_source.py", "augmentation_source.py", "requirements-lock.txt"]:
        shutil.copy2(SOURCE / name, args.run_dir / name)
    shutil.copy2(SOURCE / "metrics.json", args.run_dir / "validation_stage_metrics.json")
    shutil.copy2(SOURCE / "config.json", args.run_dir / "training_config.json")
    shutil.copy2(__file__, args.run_dir / "final_evaluation_source.py")
    lock = dict(group="AFC", epoch=47, checkpoint_sha256=LOCKED_SHA256,
                validation_accuracy_used_for_selection=.8806,
                selection="model and epoch fixed by validation stage and explicitly confirmed by user before final test",
                timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                test_preprocessing="unaltered RGB 32x32; ToTensor-equivalent and mean/std=.5; no augmentation or test-time averaging",
                training_performed=False, full_data_retraining=False,
                evaluation_source_sha256=base.sha256(__file__))
    base.write_json(args.run_dir / "lock.json", lock)
    assert base.sha256(args.run_dir / "best.pt") == LOCKED_SHA256
    base.evaluate(args)
    assert base.sha256(SOURCE / "best.pt") == base.sha256(args.run_dir / "best.pt") == LOCKED_SHA256
    metrics = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["selected_epoch"] == 47
    for part in ["train_eval", "validation"]:
        assert metrics[part]["accuracy"] == prior[part]["accuracy"]
        assert abs(metrics[part]["loss"] - prior[part]["loss"]) < 1e-6
    config = {**training, "test_evaluated": True, "phase": "locked_final_test", "test_lock": lock}
    base.write_json(args.run_dir / "config.json", config)


def compare(args):
    run, reference = args.run_dir, base.ROOT / "outputs" / "baseline_seed42"
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    old_metrics = json.loads((reference / "metrics.json").read_text(encoding="utf-8"))
    assert base.sha256(run / "best.pt") == base.sha256(SOURCE / "best.pt") == LOCKED_SHA256
    with np.load(run / "test_predictions.npz") as new, np.load(reference / "test_predictions.npz") as old:
        assert np.array_equal(new["indices"], old["indices"])
        assert np.array_equal(new["targets"], old["targets"])
        y, p, old_p = new["targets"], new["predictions"], old["predictions"]
        correct, old_correct = p == y, old_p == y
        repaired = ~old_correct & correct
        regressed = old_correct & ~correct
        assert int(repaired.sum() - regressed.sum()) == metrics["correct_test"] - old_metrics["correct_test"]
        by_class = []
        for c, name in enumerate(base.CLASSES):
            mask = y == c
            old_count, new_count = int(old_correct[mask].sum()), int(correct[mask].sum())
            by_class.append(dict(class_name=name, count=int(mask.sum()), baseline_correct=old_count,
                                 afc_correct=new_count, baseline_accuracy=old_count / int(mask.sum()),
                                 afc_accuracy=new_count / int(mask.sum()),
                                 delta_pp=100 * (new_count - old_count) / int(mask.sum()),
                                 repaired=int((repaired & mask).sum()), introduced=int((regressed & mask).sum())))
        old_cm = base.confusion_matrix(y, old_p)
        new_cm = base.confusion_matrix(y, p)
        examples = []
        with (reference / "example_selection.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["group"] != "error":
                    continue
                i = int(row["test_index"])
                examples.append(dict(test_index=i, true_class=base.CLASSES[y[i]],
                    baseline_prediction=base.CLASSES[old_p[i]], afc_prediction=base.CLASSES[p[i]],
                    afc_confidence=float(new["probabilities"][i, p[i]]), corrected=bool(correct[i])))
        with (run / "paired_test_changes.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["test_index", "true_class", "baseline_prediction", "afc_prediction", "change"])
            for i in np.flatnonzero(repaired | regressed):
                writer.writerow([int(i), base.CLASSES[y[i]], base.CLASSES[old_p[i]], base.CLASSES[p[i]],
                                 "repaired" if repaired[i] else "introduced_error"])
        result = dict(baseline_test_accuracy=old_metrics["test"]["accuracy"], afc_test_accuracy=metrics["test"]["accuracy"],
            accuracy_delta_pp=100 * (metrics["test"]["accuracy"] - old_metrics["test"]["accuracy"]),
            repaired_baseline_errors=int(repaired.sum()), introduced_errors=int(regressed.sum()),
            both_correct=int((old_correct & correct).sum()), both_wrong=int((~old_correct & ~correct).sum()),
            net_correct_gain=int(correct.sum() - old_correct.sum()),
            baseline_errors=int((~old_correct).sum()), afc_errors=int((~correct).sum()),
            relative_error_reduction=float(((~old_correct).sum() - (~correct).sum()) / (~old_correct).sum()),
            baseline_cat_dog_confusions=int(old_cm[3, 5] + old_cm[5, 3]),
            afc_cat_dog_confusions=int(new_cm[3, 5] + new_cm[5, 3]),
            per_class=by_class, previously_selected_baseline_errors=examples,
            checkpoint_unchanged=True, checkpoint_sha256=LOCKED_SHA256,
            limitation="one trained model per configuration and one standard test set; no repeated-seed stability estimate")
    with (run / "per_class_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_class[0]))
        writer.writeheader()
        writer.writerows(by_class)
    base.write_json(run / "baseline_comparison.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["evaluate", "compare"])
    p.add_argument("--run-dir", type=Path, default=DEST)
    p.add_argument("--data-dir", type=Path, default=base.ROOT / "data")
    p.add_argument("--device", choices=["cuda"], default="cuda")
    args = p.parse_args()
    {"evaluate": prepare_and_evaluate, "compare": compare}[args.command](args)
