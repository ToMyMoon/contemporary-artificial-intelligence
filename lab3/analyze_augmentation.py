"""核对八组验证结果并绘制已确认的消融图；不加载测试数据。"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import baseline as base
from augmentation_ablation import BASE_RUN, OUT, GROUPS, state_hash


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_history(run):
    with (run / "history.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(out):
    reference = np.load(BASE_RUN / "validation_predictions.npz")
    y, ids, old_predictions = reference["targets"], reference["indices"], reference["predictions"]
    old_correct = old_predictions == y
    records, classes, matrices = [], [], {}
    traces = {}
    expected_orders = read_json(out / "expected_sample_orders.json")
    preflight = read_json(out / "preflight.json")
    baseline_cfg = read_json(BASE_RUN / "config.json")
    for group, flags in GROUPS.items():
        run = BASE_RUN if group == "A0" else out / group
        metrics, cfg = read_json(run / "metrics.json"), read_json(run / "config.json")
        history = read_history(run)
        assert len(history) == 60
        assert [int(r["epoch"]) for r in history] == list(range(1, 61))
        chosen = max(history, key=lambda r: (float(r["val_accuracy"]), -float(r["val_loss"]), -int(r["epoch"])))
        assert int(chosen["epoch"]) == metrics["selected_epoch"]
        assert all(float(r["learning_rate"]) == .001 for r in history)
        assert base.sha256(run / "best.pt") == metrics["checkpoint_sha256"]
        assert base.sha256(run / "split_indices.npz") == baseline_cfg["split_sha256"]
        for name in ["seed", "epochs", "batch_size", "learning_rate", "weight_decay", "optimizer",
                     "optimizer_betas", "optimizer_eps", "cpu_threads", "scheduler", "precision", "selection"]:
            assert cfg[name] == baseline_cfg[name]
        if group != "A0":
            assert cfg["initial_state_sha256"] == preflight["initial_state_sha256"]
            assert not metrics["test_evaluated"]
            assert base.sha256(run / "augmentation_source.py") == cfg["source_sha256"]
            assert base.sha256(run / "baseline_source.py") == baseline_cfg["source_sha256"]
            with (run / "augmentation_trace.jsonl").open(encoding="utf-8") as f:
                trace = [json.loads(line) for line in f]
            assert len(trace) == 60
            for i, t in enumerate(trace):
                assert t["epoch"] == i + 1 and t["sample_presentations"] == 45000
                assert t["sample_order_sha256"] == expected_orders[i]["sample_order_sha256"]
            traces[group] = trace
        with np.load(run / "validation_predictions.npz") as pred:
            assert np.array_equal(pred["indices"], ids) and np.array_equal(pred["targets"], y)
            probabilities, predictions = pred["probabilities"], pred["predictions"]
            assert probabilities.shape == (5000, 10) and np.allclose(probabilities.sum(1), 1, atol=1e-6)
            assert np.array_equal(predictions, probabilities.argmax(1))
            correct = predictions == y
            matrix = np.zeros((10, 10), dtype=np.int64)
            for actual, predicted in zip(y, predictions):
                matrix[actual, predicted] += 1
            assert np.array_equal(matrix.sum(1), np.full(10, 500))
            assert float(correct.mean()) == metrics["validation"]["accuracy"]
            ce = float(-np.log(probabilities[np.arange(5000), y].astype(np.float64)).mean())
            assert abs(ce - metrics["validation"]["loss"]) < 1e-5
            matrices[group] = matrix
            repaired = int((~old_correct & correct).sum())
            regressed = int((old_correct & ~correct).sum())
            assert repaired - regressed == int(correct.sum()) - int(old_correct.sum())
        duration = read_json(run / "training_summary.json")["training_wall_seconds"]
        record = dict(group=group, flip=flags[0], crop=flags[1], brightness=flags[2],
                      selected_epoch=metrics["selected_epoch"], validation_correct=int(matrix.trace()),
                      validation_accuracy=metrics["validation"]["accuracy"],
                      validation_loss=metrics["validation"]["loss"],
                      clean_train_accuracy=metrics["train_eval"]["accuracy"],
                      generalization_gap_pp=100 * (metrics["train_eval"]["accuracy"] - metrics["validation"]["accuracy"]),
                      delta_from_baseline_pp=100 * (metrics["validation"]["accuracy"] - old_correct.mean()),
                      repaired_baseline_errors=repaired, introduced_errors=regressed,
                      cat_accuracy=matrix[3, 3] / 500, dog_accuracy=matrix[5, 5] / 500,
                      bird_accuracy=matrix[2, 2] / 500, cat_dog_confusions=int(matrix[3, 5] + matrix[5, 3]),
                      training_seconds=duration,
                      timing_note="previous baseline run" if group == "A0" else "includes augmentation and trace/order recording")
        records.append(record)
        for i, name in enumerate(base.CLASSES):
            classes.append(dict(group=group, class_name=name, count=500, correct=int(matrix[i, i]),
                                accuracy=matrix[i, i] / 500))
    for factor, index in [("flip", 0), ("crop", 1), ("brightness", 2)]:
        selected = [g for g, f in GROUPS.items() if f[index]]
        for epoch in range(60):
            assert len({traces[g][epoch][factor + "_sha256"] for g in selected}) == 1
    write_csv(out / "comparison.csv", records)
    write_csv(out / "validation_per_class.csv", classes)
    for group, matrix in matrices.items():
        base.write_confusion(out / f"validation_confusion_{group}.csv", matrix)
    reference.close()
    return records


def effects(out, records):
    accuracy = {r["group"]: r["validation_accuracy"] for r in records}
    specifications = {
        "flip": [("A0", "AF"), ("AC", "AFC"), ("AB", "AFB"), ("ACB", "AFCB")],
        "crop": [("A0", "AC"), ("AF", "AFC"), ("AB", "ACB"), ("AFB", "AFCB")],
        "brightness": [("A0", "AB"), ("AF", "AFB"), ("AC", "ACB"), ("AFC", "AFCB")],
    }
    paired, main = [], []
    for factor, pairs in specifications.items():
        differences = []
        for off, on in pairs:
            delta = 100 * (accuracy[on] - accuracy[off])
            differences.append(delta)
            paired.append(dict(factor=factor, off_group=off, on_group=on, delta_pp=delta,
                               net_correct_images=int(round(delta * 50))))
        main.append(dict(factor=factor, average_effect_pp=float(np.mean(differences)),
                         minimum_conditional_effect_pp=min(differences), maximum_conditional_effect_pp=max(differences)))
    a = accuracy
    interactions = {
        "flip_crop_without_brightness_pp": 100 * (a["AFC"] - a["AF"] - a["AC"] + a["A0"]),
        "flip_crop_with_brightness_pp": 100 * (a["AFCB"] - a["AFB"] - a["ACB"] + a["AB"]),
        "flip_brightness_without_crop_pp": 100 * (a["AFB"] - a["AF"] - a["AB"] + a["A0"]),
        "flip_brightness_with_crop_pp": 100 * (a["AFCB"] - a["AFC"] - a["ACB"] + a["AC"]),
        "crop_brightness_without_flip_pp": 100 * (a["ACB"] - a["AC"] - a["AB"] + a["A0"]),
        "crop_brightness_with_flip_pp": 100 * (a["AFCB"] - a["AFC"] - a["AFB"] + a["AF"]),
        "three_way_difference_pp": 100 * (a["AFCB"] - a["AFC"] - a["AFB"] - a["ACB"] + a["AF"] + a["AC"] + a["AB"] - a["A0"]),
    }
    best = max(records, key=lambda r: (r["validation_accuracy"], -r["validation_loss"]))
    leaders = [r["group"] for r in records if r["validation_accuracy"] == best["validation_accuracy"]
               and r["validation_loss"] == best["validation_loss"]]
    summary = dict(validation_leaders=leaders, primary_leader=best, main_effects=main,
                   conditional_effects=paired, interactions=interactions,
                   added_training_seconds=sum(r["training_seconds"] for r in records if r["group"] != "A0"),
                   new_test_evaluations=0,
                   limitations=["one split and one seed; conditional effects are not independent seed replications",
                                "all 60 checkpoints and 8 configurations selected/evaluated on the same validation set",
                                "results describe unaugmented validation performance under equal 60-epoch budget",
                                "new-run times include trace/order checks; baseline came from an earlier run"])
    write_csv(out / "conditional_effects.csv", paired)
    write_csv(out / "main_effects.csv", main)
    base.write_json(out / "factorial_analysis.json", summary)
    return summary


def verify_reloaded(args):
    # 新进程重新加载所有新模型，仅评价同一验证集。
    base.seed_everything(42)
    device = base.get_device(args.device)
    dataset, _ = base.load_training(args.data_dir)
    split = np.load(BASE_RUN / "split_indices.npz")
    val_loader = base.loader_for(dataset, split["val_indices"], 128, device)
    results = []
    for group in list(GROUPS)[1:]:
        run = args.out / group
        model = base.BaselineCNN().to(device)
        state = torch.load(run / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["state_dict"])
        before = state_hash(model.state_dict())
        result = base.evaluate_loader(model, val_loader, device, collect=True)
        assert before == state_hash(model.state_dict())
        with np.load(run / "validation_predictions.npz") as saved:
            difference = float(np.max(np.abs(result["probabilities"] - saved["probabilities"])))
            assert np.array_equal(result["probabilities"].argmax(1), saved["predictions"])
            assert np.allclose(result["probabilities"], saved["probabilities"], rtol=0, atol=1e-6)
        assert result["accuracy"] == state["val_accuracy"]
        results.append(dict(group=group, validation_count=5000, predictions_identical=True,
                            maximum_probability_difference=difference, evaluation_preserved_state=True))
        del model
    return results


def figures(out, records):
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    def save(fig, name):
        fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
        fig.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    histories = {g: read_history(BASE_RUN if g == "A0" else out / g) for g in GROUPS}
    for metric in ["loss", "accuracy"]:
        fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.8), sharex=True, sharey=True)
        fig.subplots_adjust(left=.055, right=.995, top=.94, bottom=.085, hspace=.30, wspace=.08)
        ymax = 100 if metric == "accuracy" else max(float(r[k]) for rows in histories.values()
                         for r in rows for k in ["train_loss", "val_loss"]) * 1.05
        handles = None
        for i, (group, rows) in enumerate(histories.items()):
            ax = axes.flat[i]
            scale = 100 if metric == "accuracy" else 1
            epochs = np.array([int(r["epoch"]) for r in rows])
            first, = ax.plot(epochs, [float(r["train_" + metric]) * scale for r in rows],
                            "--", color="#1769a8", linewidth=1.3, label="Training (online)")
            second, = ax.plot(epochs, [float(r["val_" + metric]) * scale for r in rows],
                             color="#d35f13", linewidth=1.4, label="Validation (epoch end)")
            record = next(r for r in records if r["group"] == group)
            third = ax.axvline(record["selected_epoch"], linestyle=":", color="#666666", linewidth=1,
                               label="Selected epoch")
            handles = [first, second, third]
            ax.set(xlim=(1, 60), ylim=(0, ymax), xlabel=f"Epoch\n({chr(97+i)}) {group}")
            ax.grid(alpha=.2)
        fig.supylabel("Accuracy (%)" if metric == "accuracy" else "Mean cross-entropy", x=.005)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.0), ncol=3, frameon=False)
        save(fig, f"{metric}_curves_8groups")
    rmap = {r["group"]: r for r in records}
    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    pairs = [("A0", "AB"), ("AF", "AFB"), ("AC", "ACB"), ("AFC", "AFCB")]
    all_values = []
    for i, (off, on) in enumerate(pairs):
        values = [rmap[off]["validation_accuracy"] * 100, rmap[on]["validation_accuracy"] * 100]
        all_values.extend(values)
        ax.plot([i-.07, i+.07], values, color="#8a8a8a", linewidth=1.3, zorder=1)
        for j, (group, value) in enumerate(zip([off, on], values)):
            x = i + (-.07 if j == 0 else .07)
            color = "#1769a8" if j == 0 else "#d35f13"
            ax.scatter(x, value, s=60, color=color, zorder=3,
                       label=("Brightness off" if j == 0 else "Brightness on") if i == 0 else None)
            above = value >= values[1-j]
            ax.annotate(f"{group}: {value:.2f}%", (x, value), xytext=(0, 11 if above else -19),
                        textcoords="offset points", ha="center", fontsize=9, color=color)
    ax.set_xticks(range(4), ["None", "Flip", "Crop", "Flip + crop"])
    ax.set(xlabel="Other enabled factors", ylabel="Best validation accuracy (%)", xlim=(-.5, 3.5),
           ylim=(min(all_values)-1.1, max(all_values)+1.1))
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, loc="best")
    save(fig, "brightness_paired_accuracy")


def notes(out, records, summary):
    lines = ["# 数据增强消融阶段记录", "",
             "本文件为已确认补充实验的结果记录，尚不是正式报告。本轮复用 A0，新增七组各 60 轮训练，仅使用训练集和验证集，没有新增测试集评价。", "",
             "F：水平翻转 p=0.5；C：补 4 像素黑边后随机裁剪 32×32；B：亮度系数 Uniform[0.8,1.2)，在浮点 RGB 上乘系数并截断。组合按 C→F→B 执行。", "",
             "## 控制条件与核查", "",
             "所有组复用 45,000／5,000 的实际划分索引、相同 CNN、种子 42、初始权重、样本顺序、AdamW（lr=0.001、weight_decay=0.0001）、batch=128 和 60 轮预算。验证准确率最高者入选，同分选择交叉熵更低者，再同分保留更早轮次。", "",
             "F/C/B 随机种子分别为 1042/2042/3042，与模型及采样器 RNG 分离；相同因素在对应组中每轮使用相同参数序列。原基线未保存顺序哈希，本轮以原 DataLoader 协议重建参照，新七组均核对了实际 60 轮顺序。", "",
             "关闭增强时，前三批输入及两步模型更新与原入口逐位一致；八种变换与独立 NumPy 实现的最大绝对误差约 1.19e-7。全部历史、选择规则、混淆矩阵、概率交叉熵和成对错误变化已核查；另进程重载七个最佳模型复核全部验证预测，详见 verification.json。", "",
             "## 八组验证结果", "",
             "下表训练准确率是最佳模型在无增强训练图像上的固定权重评价，用于计算可比的泛化差；不等于增强训练曲线的在线指标。每组验证样本数都是 5,000。", "",
             "| 组别 | 最佳轮次 | 干净训练准确率 | 验证准确率 | 验证交叉熵 | 比 A0 变化／百分点 | 泛化差／百分点 | 训练秒数 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['group']} | {r['selected_epoch']} | {r['clean_train_accuracy']:.3%} | {r['validation_accuracy']:.2%} | {r['validation_loss']:.4f} | {r['delta_from_baseline_pp']:+.2f} | {r['generalization_gap_pp']:.2f} | {r['training_seconds']:.1f} |")
    leader = summary["primary_leader"]
    baseline_correct = next(r["validation_correct"] for r in records if r["group"] == "A0")
    lines += ["", f"按预定规则，本轮验证领先组为 **{', '.join(summary['validation_leaders'])}**，准确率 **{leader['validation_accuracy']:.2%}**，比 A0 高 **{leader['delta_from_baseline_pp']:.2f} 个百分点**，对应净多判断正确 **{leader['validation_correct']-baseline_correct}** 张验证图像。该排名是验证结果，不是新增测试成绩。", "",
              "![八组准确率曲线](outputs/augmentation_seed42/figures/accuracy_curves_8groups.png)", "",
              "![八组损失曲线](outputs/augmentation_seed42/figures/loss_curves_8groups.png)", "",
              "两图均为原始 60 轮记录，未平滑。所有子图采用统一坐标范围，虚线表示在线训练、实线表示轮末验证、竖点线表示所选轮次。增强组训练输入与验证输入不同，不能直接用在线曲线之差作为可比泛化差。", "",
              "## 三个因素及亮度的条件效应", "",
              "平均效应是固定另外两个因素的四个条件差值的平均，单位为百分点；这四个条件不是四次独立种子重复，不提供误差条或显著性结论。", "",
              "| 因素 | 平均效应／百分点 | 最小条件效应 | 最大条件效应 |", "|---|---:|---:|---:|"]
    names = {"flip": "水平翻转", "crop": "随机裁剪", "brightness": "随机亮度"}
    for r in summary["main_effects"]:
        lines.append(f"| {names[r['factor']]} | {r['average_effect_pp']:+.3f} | {r['minimum_conditional_effect_pp']:+.2f} | {r['maximum_conditional_effect_pp']:+.2f} |")
    lines += ["", "亮度的四个配对对照：", "", "| 亮度关闭→开启 | 准确率变化／百分点 | 净正确图像变化 |", "|---|---:|---:|"]
    brightness = [r for r in summary["conditional_effects"] if r["factor"] == "brightness"]
    for r in brightness:
        lines.append(f"| {r['off_group']}→{r['on_group']} | {r['delta_pp']:+.2f} | {r['net_correct_images']:+d} |")
    positive = sum(r["delta_pp"] > 0 for r in brightness)
    negative = sum(r["delta_pp"] < 0 for r in brightness)
    lines += ["", f"在本次四个配对条件中，加入亮度后有 {positive} 个提高、{negative} 个降低，其余不变。差值的大小仍需逐项判断；不能将单个种子上的差值当作跨种子的稳定效应。", "",
              "![亮度配对准确率图](outputs/augmentation_seed42/figures/brightness_paired_accuracy.png)", "",
              "三种两因素交互的条件差分和三因素差分保存在 factorial_analysis.json。它们描述已选最佳模型上的验证准确率组合关系，不直接证明视觉特征互补的机制，也不验证平移或光照扰动下的鲁棒性。", "",
              "## 修正旧错误、引入新错误", "",
              "对同一 5,000 张验证图像逐项配对；净修正=修正 A0 错误−引入 A0 没有的新错误。", "",
              "| 组别 | 修正 A0 错误 | 引入新错误 | 净修正 |", "|---|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['group']} | {r['repaired_baseline_errors']} | {r['introduced_errors']} | {r['repaired_baseline_errors']-r['introduced_errors']:+d} |")
    lines += ["", "## 关注类别", "", "以下均为验证集结果，每类 500 张，不能与基础阶段每类 1,000 张测试数据的准确率混用。", "",
              "| 组别 | 猫准确率 | 狗准确率 | 鸟准确率 | 猫狗双向混淆张数 |", "|---|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['group']} | {r['cat_accuracy']:.1%} | {r['dog_accuracy']:.1%} | {r['bird_accuracy']:.1%} | {r['cat_dog_confusions']} |")
    by_group = {r["group"]: r for r in records}
    fc, fcb, reference = by_group["AFC"], by_group["AFCB"], by_group["A0"]
    lines += ["", "## 主要观察与有限结论", "",
              f"1. 翻转＋裁剪在本轮取得最高验证准确率。其干净训练／验证差距从 A0 的 {reference['generalization_gap_pp']:.2f} 降为 {fc['generalization_gap_pp']:.2f} 个百分点，验证交叉熵从 {reference['validation_loss']:.4f} 降为 {fc['validation_loss']:.4f}。这支持该方案在相同预算和选择规则下改善了本次泛化表现；不证明具体鲁棒性机制。",
              f"2. AFC 修正了 {fc['repaired_baseline_errors']} 个 A0 原有错误，同时引入 {fc['introduced_errors']} 个新错误，净修正 {fc['repaired_baseline_errors']-fc['introduced_errors']} 个。增强并非保留全部原有正确结果后再额外修正，而是改变了分类决策。",
              f"3. AFC→AFCB 的总体变化为 {fcb['validation_accuracy']*100-fc['validation_accuracy']*100:+.2f} 个百分点，但猫准确率从 {fc['cat_accuracy']:.1%} 变为 {fcb['cat_accuracy']:.1%}，狗从 {fc['dog_accuracy']:.1%} 变为 {fcb['dog_accuracy']:.1%}。同一验证集上，接近的总准确率可以掩盖较大的类别间变化；这里只陈述本次所选模型的差异，不推广到其他种子。",
              "4. 按事先指定的准确率主指标，本轮没有证据支持把亮度加入默认方案。AF→AFB、AC→ACB 的准确率分别仅少 1 张、2 张，但交叉熵反而下降，因此也不能概括成亮度在所有评价方面都有害。",
              "5. 不含亮度时，翻转＋裁剪的交互差分为 +0.94 个百分点，组合也高于任一单项；这是本次结果中的组合收益，不能仅凭一次运行断言稳定协同。"]
    lines += ["", "全部类别的样本数、正确数及准确率见 validation_per_class.csv，各组原始混淆矩阵另存 CSV。", "",
              "## 成本、限制与下一阶段", "",
              f"新增七组训练合计 **{summary['added_training_seconds']:.1f} 秒（{summary['added_training_seconds']/60:.2f} 分钟）**。计时含训练、在线增强、逐轮验证、最佳权重和核查记录写入；不含准备、最终评价、重载核对和绘图。原 A0 计时来自较早的一次运行，新组额外包含少量顺序与参数校验开销，不能把全部时间差归因于增强。", "",
              "每个配置只训练一次，同一验证集用于选择 60 轮中的权重及八种配置。因此当前结论限于本次划分、种子和预算，不能当作无偏测试性能估计或稳定优势。小幅差距需结合样本数解读，0.1 个百分点对应 5 张验证图像。", "",
              "下一阶段是确认本轮结果和拟锁定方案，再决定最终测试评价。本轮没有新增测试成绩，没有多种子重复，也没有修改增强强度、学习率或模型结构；尚未撰写正式报告。", "",
              "原始证据：[outputs/augmentation_seed42](outputs/augmentation_seed42)。运行方法见 [README](README.md)。"]
    boundary = [r["group"] for r in records if r["selected_epoch"] == 60]
    if boundary:
        lines += ["", f"最佳轮次落在第 60 轮的组为 {', '.join(boundary)}。这不意味着已达到各自更长训练下的性能上限；本轮不自动增加轮数，所有结论限定于共同 60 轮预算。"]
    target = base.ROOT / "AUGMENTATION_RESULTS.md" if out.resolve() == OUT.resolve() else out / "RESULTS.md"
    relative = Path(os.path.relpath(out, target.parent)).as_posix()
    content = ("\n".join(lines) + "\n").replace("outputs/augmentation_seed42", relative)
    target.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--data-dir", type=Path, default=base.ROOT / "data")
    p.add_argument("--device", choices=["cuda"], default="cuda")
    p.add_argument("--skip-reload", action="store_true", help="Only for regenerating figures after a recorded successful reload check")
    args = p.parse_args()
    records = collect(args.out)
    summary = effects(args.out, records)
    if args.skip_reload:
        assert read_json(args.out / "verification.json")["status"] == "passed"
    else:
        reloads = verify_reloaded(args)
        base.write_json(args.out / "verification.json", dict(status="passed", groups=reloads,
                histories_and_selection_verified=True, sample_orders_verified_all_epochs=True,
                factor_random_streams_matched_all_epochs=True, metrics_independently_recomputed=True,
                no_new_test_evaluations=True, full_training_repeated=False))
    figures(args.out, records)
    notes(args.out, records, summary)
    print(json.dumps(summary), flush=True)
