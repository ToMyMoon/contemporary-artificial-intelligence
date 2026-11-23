"""从既有 SE/Grad-CAM 输出生成阶段记录；不训练、不修改正式报告。"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np

import baseline as base
import augmentation_ablation as aug
import se_ablation as se


def main(args):
    root=base.ROOT;s=args.se_out;g=args.cam_out
    a=se.read(s/'comparison_analysis.json');cam=se.read(g/'analysis.json')
    assert se.read(s/'verification.json')['status']==se.read(g/'verification.json')['status']=='passed'
    cfg=se.read(s/'config.json')
    assert base.sha256(s/'se_ablation.py')==base.sha256(root/'se_ablation.py')==cfg['source_sha256']
    assert base.sha256(root/'gradcam_analysis.py')==base.sha256(g/'gradcam_source.py')==cam['source_sha256']
    oldtrace=[json.loads(t) for t in (se.AFC/'augmentation_trace.jsonl').read_text().splitlines()]
    newtrace=[json.loads(t) for t in (s/'augmentation_trace.jsonl').read_text().splitlines()]
    assert len(oldtrace)==len(newtrace)==60
    for old,new in zip(oldtrace,newtrace):
        for field in ['epoch','sample_order_sha256','flip_sha256','crop_sha256']:assert old[field]==new[field]
    with np.load(se.AFC/'validation_predictions.npz') as old,np.load(s/'validation_predictions.npz') as new:
        y=old['targets'];oa=old['predictions']==y;ob=new['predictions']==y
        la=-np.log(old['probabilities'][np.arange(5000),y].astype(float))
        lb=-np.log(new['probabilities'][np.arange(5000),y].astype(float))
        groups=[dict(group=n,count=int(m.sum()),afc_ce=float(la[m].mean()),se_ce=float(lb[m].mean()),
            weighted_delta=float((lb[m]-la[m]).sum()/5000)) for n,m in [
            ('both_correct',oa&ob),('repaired',~oa&ob),('introduced',oa&~ob),('both_wrong',~oa&~ob)]]
        probability=dict(changed_predictions=int((old['predictions']!=new['predictions']).sum()),
            ce_change_by_correctness_group=groups,delta_ce=float((lb-la).mean()))
    base.write_json(s/'probability_analysis.json',probability)
    ref,new=a['reference'],a['se']
    relative_loss=100*(ref['validation_loss']-new['validation_loss'])/ref['validation_loss']
    evaluation_overhead=100*(new['validation_eval_median_seconds']/ref['validation_eval_median_seconds']-1)
    parameter_overhead=100*(new['parameters']/ref['parameters']-1)
    source_frozen=se.check_frozen(s);cam_frozen=se.check_frozen(g)
    assert not (s/'test_predictions.npz').exists() and not cfg['test_evaluated']
    for script in ['analyze_se.py','extension_results.py']:shutil.copy2(root/script,s/script)
    (g/'supplement_index.md').write_text('# Grad-CAM 补充图索引\n\n全部样本先按规则固定，再生成热图；不是随机抽取的准确率估计样本。\n\n'+
        '\n'.join(f'- [{f.name}](figures/{f.name})' for f in sorted((g/'figures').glob('*.png')))+'\n',encoding='utf-8')
    table='\n'.join(f'| {r["class_name"]} | {r["afc_accuracy"]:.1%} | {r["se_accuracy"]:.1%} | {r["delta_pp"]:+.1f} | {r["repaired"]} | {r["introduced"]} |' for r in a['per_class'])
    main_table='\n'.join(f'| {r["group"]} | {r["selected_epoch"]} | {r["validation_accuracy"]:.2%} | {r["validation_correct"]} | {r["validation_loss"]:.6f} | {r["clean_train_accuracy"]:.4%} | {r["generalization_gap_pp"]:.2f} | {r["parameters"]:,} |' for r in [ref,new])
    text=f'''# SE 对照阶段结果

本阶段已完成1组新增 SE-AFC 的60轮训练。只评价训练/验证集；没有 SE 测试集成绩，未修改正式报告或原 AFC 最终测试记录。

## 核心结果

| 模型 | 最佳轮次 | 验证准确率 | 正确数/5000 | 验证交叉熵 | 干净训练准确率 | 泛化差/百分点 | 参数量 |
|---|---:|---:|---:|---:|---:|---:|---:|
{main_table}

SE 未提高验证准确率；交叉熵降低{relative_loss:.2f}%。按预设的“准确率同分时比较交叉熵”规则，SE 在本次验证排序中优先，但这不等于测试性能更好，也不支持全面优于 AFC。泛化差没有缩小。

![SE与AFC原始训练对照](outputs/se_seed42/figures/se_training_comparison.png)

左右分别为平均交叉熵与准确率。颜色区分模型，虚线为在线训练、实线为轮末验证；竖点线标记第47/48轮。保留全部60轮、未平滑。在线曲线不能直接作为固定权重的泛化差。

## 控制条件与实现检查

三个 SE 均放在每块第二次 ReLU 之后、最大池化之前；缩减比16，隐藏维数2/4/8，两层全连接均带偏置，初始化种子4042。增加2,926个参数（{parameter_overhead:.2f}%）。共有层重建原种子42的初始权重并逐张量核对；没有从训练好的 AFC 微调。绕过 SE 时，前向输出与原 CNN 逐位一致。SE 前向公式、有效梯度和64样本100步拟合通过检查，核查权重全部丢弃。

数据划分、初始化、45,000张样本的逐轮顺序、翻转/裁剪随机参数、AdamW、学习率0.001、衰减0.0001、batch128、60轮及确定性FP32均按原协议。60轮实际顺序和两个增强参数哈希全部与 AFC 一致。

## 相同总分下的类别得失

SE 修正{a['repaired_afc_errors']}个 AFC 错误，同时引入{a['introduced_errors']}个新错误，净修正0。两者均正确{a['both_correct']}张、均错误{a['both_wrong']}张；共有{probability['changed_predictions']}张图像的预测类别不同，包括错误类别之间的变化。

每类验证图像均为500张。修正/新增均相对于 AFC：

| 类别 | AFC | SE-AFC | 变化/百分点 | 修正 | 新增错误 |
|---|---:|---:|---:|---:|---:|
{table}

狗、鸟、青蛙改善，但猫从83.0%降至71.8%，汽车也下降3.0个百分点；总准确率持平掩盖了分类行为变化。不能只保留有利类别来说明 SE 有效。

## 交叉熵为何可以改善而准确率不变

准确率只取最高分对应的类别，交叉熵还取决于分配给真实类别的概率。双方都错的309张中，平均交叉熵从{groups[3]['afc_ce']:.4f}降至{groups[3]['se_ce']:.4f}，即部分错误的真实类别概率不再被压得那么低；双方都对的4,115张中则从{groups[0]['afc_ce']:.4f}升至{groups[0]['se_ce']:.4f}。连同修正/新增两组的变化，加权后总差为{probability['delta_ce']:.6f}。因此是概率分配的整体取舍，不能直接称为置信度校准改善。

## 成本

历史 AFC 训练及验证{ref['training_seconds']:.2f}秒，新增 SE {new['training_seconds']:.2f}秒；峰值CUDA已分配张量内存分别{ref['training_peak_allocated_mib']:.2f}/{new['training_peak_allocated_mib']:.2f}MiB，不含全部驱动/进程显存。历史训练计时不是同一时段的严格算子开销对照。

同一进程、同一5,000张验证图像、batch128，分别预热一遍后交替测5遍，完整评价中位时间分别{ref['validation_eval_median_seconds']*1000:.2f}/{new['validation_eval_median_seconds']*1000:.2f}毫秒，本机 SE 增加约{evaluation_overhead:.1f}%。范围分别{ref['validation_eval_min_seconds']*1000:.2f}–{ref['validation_eval_max_seconds']*1000:.2f}、{new['validation_eval_min_seconds']*1000:.2f}–{new['validation_eval_max_seconds']*1000:.2f}毫秒。计时含CPU DataLoader、传输、前向及损失/准确率，不是纯模型内核时延。

## 复核与结论范围

新进程重载后5,000张验证预测及概率完全一致，独立交叉熵和最佳轮次复算通过。{source_frozen}个固定实验输入文件（含原权重）哈希不变。

本轮支持“验证准确率持平、交叉熵改善，同时存在类别与效率代价”。只有一个训练种子，同一验证集用于选择权重；也未增加参数量匹配的第三组，因此不声称稳定优势或独立证明注意力机制。没有根据结果追加种子、调整SE比例或查看SE测试集。

原始记录：`outputs/se_seed42/`；`comparison.csv`、`per_class_comparison.csv`、`probability_analysis.json`、`benchmark.json` 和 `verification.json`。
'''
    (root/'SE_RESULTS.md').write_text(text,encoding='utf-8')
    text=f'''# Grad-CAM 阶段结果

本阶段解释已有 A0 第59轮与 AFC 第47轮。没有训练新模型、评价SE，未修改正式报告。主图固定原先四例，补充20张验证图像：每类一张AFC错例与一张正确例。按预定置信度规则选择，名单先于热图生成；不是无偏准确率估计样本。

## 主图

![真实类别目标的Grad-CAM](outputs/gradcam_a0_afc/figures/primary_true_compact.png)

2×2个样本组，每组横排原图/A0/AFC。两模型对同一图像解释同一真实类别。Target表示热图目标，预测及括号内softmax置信度是实际输出，二者可能不同。

![A0预测类别共同目标的补充图](outputs/gradcam_a0_afc/figures/primary_a0_prediction_compact.png)

此图把同一行的目标同时换为A0预测类别，仍保持跨模型目标一致。两种目标的大图、20张验证图分成每页5张的补充图，见[完整图索引](outputs/gradcam_a0_afc/supplement_index.md)。

## 可直接观察的现象与有限解释

- 鸟#3094：AFC已把飞机修正为鸟；在共同“鸟”目标下，A0高响应分布较分散，AFC更集中于图像中上部。这是空间分布变化，不足以证明修正原因。
- 狗#262：A0预测猫，AFC预测狗；共同“狗”目标的热点由左下侧转向头部附近及身体右下侧。它提示两模型对同一类别的响应不同，不能据此排除背景等其他因素。
- 鹿#58：两模型都预测猫。解释“鹿”与解释“猫”时热点位置不同；即使对真实类别有局部正响应，也不代表真实类别得分最高。
- 卡车#6419：AFC仍以98.7%置信度预测船，热图变化并未对应分类修正。局部亮区或看起来贴近物体，都不能保证预测正确。

20张验证样本同时保留正确和错误情况，并未按热图是否“好看”筛选。没有定位标注，不能给出定位准确率或证明模型关注区域更正确；没有扰动/遮挡干预，不能把热点当错误因果证据。

## 方法与核查

目标层为第三块最终ReLU输出、最后最大池化之前，原生128×8×8。按目标logit求梯度，对通道梯度作空间平均，按权重求和并ReLU。8×8热图双线性插值到32×32，与原始RGB按0.4透明度叠加；图像最近邻放大，无输入增强。逐图除以自身最大值，仅比较空间模式，不跨模型比较颜色强弱；原始幅度另存。

共24张图像、{cam['heatmaps']}张主分析热图；当真实类别与A0预测类别相同时复用同一图，故少于96张上限。全零热图{cam['zero_maps']}张。另对固定20张验证图在AFC副本上以种子5042随机化分类头，得到20张敏感性核查图；原参数不变。

原网络尾部为MaxPool→GAP→Linear，可独立推导目标层Grad-CAM通道权重为分类器权重除以64，数值核对通过。用保存激活/梯度在NumPy复算，原始热图最大误差{cam['independent_numpy_raw_max_error']:.3g}；独立进程重载后全部原始热图、归一化图及logit完全一致。

分类头随机化后20张图均发生变化，空间Pearson相关中位数为{cam['randomized_head_sanity']['median_pearson']:.4f}，归一化热图平均绝对差为{cam['randomized_head_sanity']['mean_normalized_absolute_difference']:.4f}。这支持热图对模型参数敏感，不是解释真实性或因果正确性的充分证明，不设任意相关阈值作为通过标准。

单图求梯度使用batch1，与历史batch128概率的最大差为{cam['max_saved_batch128_probability_difference']:.3g}；类别预测一致。挂接激活前后同批logit逐位一致，参数与BN统计未改变。计算部分约{cam['elapsed_compute_seconds']:.2f}秒，不含数据加载、启动和绘图。

## 保存内容

`selection.json`保存24张名单与像素哈希；`maps.npz`保存原图、原生热图、激活及梯度；`map_records.json`保存目标与原始幅度；`sanity_records.json`和`randomized_head_maps.npz`保存随机化核查；`verification.json`记录独立进程复核。

方法参考：[SE](https://arxiv.org/abs/1709.01507)、[Grad-CAM](https://arxiv.org/abs/1610.02391)、[参数随机化检查](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html)。
'''
    (root/'GRADCAM_RESULTS.md').write_text(text,encoding='utf-8')
    audit=dict(status='passed',se_training_runs_added=1,se_epochs=60,se_test_evaluated=False,
        se_trace_epochs_matched=60,se_all_5000_validation_predictions_reloaded_exact=True,
        gradcam_samples=24,gradcam_maps=cam['heatmaps'],randomized_head_maps=20,
        gradcam_maps_reloaded_exact=True,gradcam_figures=len(list((g/'figures').glob('*.png'))),
        original_files_unchanged=source_frozen,gradcam_frozen_files_unchanged=cam_frozen,
        formal_report_modified=False,se_checkpoint=se.read(s/'metrics.json')['checkpoint_sha256'],
        source_hashes={n:base.sha256(root/n) for n in ['se_ablation.py','analyze_se.py','gradcam_analysis.py','extension_results.py']})
    base.write_json(s/'extension_verification.json',audit)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--se-out',type=Path,default=se.OUT)
    p.add_argument('--cam-out',type=Path,default=base.ROOT/'outputs/gradcam_a0_afc')
    main(p.parse_args())
