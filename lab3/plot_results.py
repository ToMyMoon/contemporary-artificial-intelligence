"""按已确认形式绘制原始曲线、混淆矩阵和规则选样网格。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10

from baseline import CLASSES, ROOT, write_json


def save_figure(fig, out, stem, vector=True):
    fig.savefig(out / f"{stem}.png", dpi=180, facecolor="white", bbox_inches="tight")
    if vector:
        fig.savefig(out / f"{stem}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def curves(run, out, metrics):
    with (run / "history.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = np.array([int(r["epoch"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    for ax, metric, ylabel, scale in zip(axes, ["loss", "accuracy"],
                                       ["Mean cross-entropy", "Accuracy (%)"], [1, 100]):
        train = np.array([float(r[f"train_{metric}"]) for r in rows]) * scale
        val = np.array([float(r[f"val_{metric}"]) for r in rows]) * scale
        ax.plot(epochs, train, color="#1769a8", linestyle="--", linewidth=1.6, label="Training (online)")
        ax.plot(epochs, val, color="#d35f13", linewidth=1.7, label="Validation (epoch end)")
        ax.axvline(metrics["selected_epoch"], color="#666666", linestyle=":", linewidth=1.2,
                   label=f"Selected epoch: {metrics['selected_epoch']}")
        ax.set(xlabel="Epoch", ylabel=ylabel, xlim=(1, epochs[-1]))
        ax.set_ylim(bottom=0, top=100 if metric == "accuracy" else max(train.max(), val.max()) * 1.06)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    save_figure(fig, out, "training_curves")


def confusion(run, out):
    with (run / "confusion_counts.csv").open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    matrix = np.array([r[1:] for r in rows[1:]], dtype=int)
    percentages = matrix / matrix.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(10, 8.4), constrained_layout=True)
    im = ax.imshow(percentages, vmin=0, vmax=100, cmap="Blues", interpolation="nearest")
    ax.set_xticks(np.arange(10), CLASSES, rotation=40, ha="right")
    ax.set_yticks(np.arange(10), [f"{name} (n={matrix[i].sum()})" for i, name in enumerate(CLASSES)])
    ax.set(xlabel="Predicted class", ylabel="True class")
    for r in range(10):
        for c in range(10):
            ax.text(c, r, f"{percentages[r, c]:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if percentages[r, c] > 55 else "#172333")
    fig.colorbar(im, ax=ax, shrink=0.88, label="Share of true class (%)")
    save_figure(fig, out, "confusion_matrix")
    return matrix


def sample_grids(run, out, data_dir, pred):
    targets, predictions, prob = pred["targets"], pred["predictions"], pred["probabilities"]
    confidence = prob[np.arange(len(targets)), predictions]
    data = CIFAR10(str(data_dir), train=False, download=False).data
    records = []
    for c in range(10):
        wrong = np.flatnonzero((targets == c) & (predictions != c))
        correct = np.flatnonzero((targets == c) & (predictions == c))
        if not len(wrong) or not len(correct):
            raise ValueError(f"Class {c}: no error/correct pair available")
        median_conf = float(np.median(confidence[wrong]))
        # np.flatnonzero 为原始索引顺序；argmin 并列时取较小索引。
        error_index = int(wrong[np.argmin(np.abs(confidence[wrong] - median_conf))])
        correct_index = int(correct[np.argmin(np.abs(confidence[correct] - confidence[error_index]))])
        for group, index in [("error", error_index), ("correct", correct_index)]:
            records.append(dict(group=group, true_class=CLASSES[c], test_index=index,
                                predicted_class=CLASSES[predictions[index]], confidence=float(confidence[index]),
                                class_error_median_confidence=median_conf))
    for group, stem in [("error", "misclassified_examples"), ("correct", "correct_examples")]:
        fig, axes = plt.subplots(2, 5, figsize=(14, 6.1), constrained_layout=True)
        selected = [r for r in records if r["group"] == group]
        for ax, record in zip(axes.flat, selected):
            ax.imshow(data[record["test_index"]], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"True: {record['true_class']}\nPred: {record['predicted_class']}\n"
                          f"p(pred): {record['confidence']:.1%} | index: {record['test_index']}",
                          fontsize=9, color="#9f2a20" if group == "error" else "#175a75")
        save_figure(fig, out, stem, vector=False)
    with (run / "example_selection.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def main(args):
    run = args.run_dir
    out = run / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    curves(run, out, metrics)
    matrix = confusion(run, out)
    with np.load(run / "test_predictions.npz") as pred:
        records = sample_grids(run, out, args.data_dir, pred)
        p, y = pred["predictions"], pred["targets"]
        confidence = pred["probabilities"][np.arange(len(y)), p]
        wrong = p != y
        high_conf = confidence >= 0.9
        directed = [(int(matrix[r, c]), r, c) for r in range(10) for c in range(10) if r != c]
        directed.sort(key=lambda t: (-t[0], t[1], t[2]))
        pairs = [(int(matrix[r, c] + matrix[c, r]), r, c) for r in range(10) for c in range(r + 1, 10)]
        pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
        summary = dict(
            most_frequent_directed_errors=[dict(count=n, true_class=CLASSES[r], predicted_class=CLASSES[c],
                                                fraction_of_true_class=n / matrix[r].sum()) for n, r, c in directed[:10]],
            most_frequent_confusion_pairs=[dict(count=n, classes=[CLASSES[r], CLASSES[c]],
                                                fraction_of_all_errors=n / wrong.sum()) for n, r, c in pairs[:10]],
            error_confidence_quantiles=dict(zip(["min", "q25", "median", "q75", "max"],
                                               np.quantile(confidence[wrong], [0, .25, .5, .75, 1]).tolist())),
            high_confidence_errors=int((wrong & high_conf).sum()),
            high_confidence_predictions=int(high_conf.sum()),
            incorrect_fraction_among_high_confidence_predictions=float(wrong[high_conf].mean()),
            examples=records,
            example_selection_rule="One error per true class, nearest to that class's median error confidence; ties: smallest test index. One correctly classified example per same class, nearest predicted confidence to the selected error; ties: smallest index. Neither is a causal control.",
            image_display="Original 32x32 RGB pixels enlarged with nearest-neighbor display; no enhancement or crop",
            curve_display="All 60 raw epoch records; no smoothing/interpolation; training online, validation fixed model after epoch; CE excludes AdamW weight decay",
        )
        write_json(run / "descriptive_analysis.json", summary)
    print(json.dumps(dict(event="figures_created", directory=str(out), files=sorted(p.name for p in out.iterdir()))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "baseline_seed42")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    main(p.parse_args())
