# 实验三：CIFAR-10 CNN 与拓展实验

本实验完成了基础 CNN、翻转／裁剪／亮度三因素消融、SE 通道注意力对照、Grad-CAM 分析及标签平滑对照。先由验证集选定 AFC（翻转＋裁剪）第47轮，再用锁定权重取得 87.76% 的官方测试准确率。后续分别锁定 SE-AFC 第48轮及标签平滑组第43轮，用同一官方测试集与 AFC 对照，最终保留 AFC 第47轮。测试数据与核查记录见[测试对照与复核](#测试对照与复核)。训练采用固定学习率及60轮预算，从头初始化。

从仓库根目录进入 `lab3` 后运行下文命令。仓库随代码附上 `evidence/` 中的指标和复核记录。运行产生的数据、权重和图保存在本地，供后续复核。

## 环境

推荐 64 位 Python 3.11、NVIDIA CUDA GPU。代码使用 PyTorch 2.7.1 + CUDA 12.6、torchvision 0.22.1。CUDA 运行库随 PyTorch 安装，不要求另装 CUDA Toolkit；显卡驱动需要兼容。安装版本依据 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。

在本目录打开 PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果系统没有 `py` 命令，请用已安装的 Python 3.11 执行同一条 `-m venv .venv` 命令。本机运行时为复用已有 NumPy、Matplotlib、Pillow，以 `--system-site-packages` 创建了环境。实测详细版本见结果的 `environment.json` 和项目的 `requirements-lock.txt`。从新结果目录复跑时使用了同一台机器的依赖环境。

本次实测 Python 3.11.9、NumPy 2.4.0、Matplotlib 3.10.8、Pillow 12.0.0；GPU 为 8 GB RTX 4060 Laptop GPU。需要锁定本次全部 21 个运行依赖的版本时，用 `requirements-lock.txt` 替代上面的 `requirements.txt`，该文件针对本次 Windows CUDA 环境生成。

CPU 环境可改装相同版本的 CPU PyTorch/torchvision 并以 `--device cpu` 运行；CPU 性能及跨设备的逐位一致性未验证。

`download_assets.py` 是本机网络较慢时使用的官方文件分段续传辅助，不参与模型或训练方法。下载的运行库安装包放在 `.downloads/`，以官方索引中的 SHA-256 核验；该目录不属于提交源码。

## 数据和运行顺序

CIFAR-10 原始数据来自官方多伦多大学地址，不随源码提交。下载脚本验证官方压缩包 MD5；`prepare` 通过 torchvision 再次校验并解压，另存各文件 SHA-256。不要使用来历不明的 pickle 数据文件。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 download_data.py
.\.venv\Scripts\python.exe -B -X utf8 baseline.py prepare
.\.venv\Scripts\python.exe -B -X utf8 checks.py preflight --device cuda
.\.venv\Scripts\python.exe -B -X utf8 baseline.py train --device cuda
.\.venv\Scripts\python.exe -B -X utf8 baseline.py evaluate --device cuda
.\.venv\Scripts\python.exe -B -X utf8 checks.py verify --device cuda
.\.venv\Scripts\python.exe -B -X utf8 plot_results.py
```

下载、数据解压、模型训练、模型评价、验证和绘图分别执行，失败时不自动跳过检查。`train` 拒绝覆盖已有 `config.json` 的结果目录。需要完整重跑时使用另一个目录，例如 `--run-dir outputs/reproduction_seed42`，其后的 `evaluate`、`verify`、绘图均传入相同目录。未提供中途恢复功能。

## 固定基础配置

- 数据：官方 50,000 张训练图像按类别分层，训练 45,000、验证 5,000；每类分别 4,500、500 张。官方测试集 10,000 张不参与选择。
- 划分与训练种子：42。划分使用 NumPy `default_rng`，索引保存于 `split_indices.npz`；训练打乱使用独立的 PyTorch Generator。
- 预处理：RGB、32×32，像素除以 255 后按 mean=(0.5,0.5,0.5)、std=(0.5,0.5,0.5) 归一化。为避免重复 PIL 转换，预先在 CPU 批量转成等价的 NCHW 浮点张量，通过 DataLoader 供给模型。
- 模型：三个块，通道 32/64/128；每块两层 3×3 无偏置卷积、BatchNorm、ReLU，末尾 2×2 最大池化；全局平均池化后 Linear(128,10)。从头训练。
- 优化：AdamW，lr=0.001、betas=(0.9,0.999)、eps=1e-8、weight_decay=0.0001。权重衰减应用于全部可训练参数，包括 BN 和分类头偏置。
- 批量大小 128，60 轮，FP32，无 AMP、TF32；固定学习率；交叉熵不含额外类别权重。
- 模型选择：验证准确率最高；相同时验证交叉熵最低；仍相同时保留更早轮次。保存最佳及末轮权重，评价使用最佳权重。
- 未将验证数据加入训练，未全量重训；不以测试结果修改配置。

可覆盖参数只有 `--data-dir`、`--run-dir`、`--device`、`--epochs`、`--seed`。60 轮及种子 42 为本次正式基础运行；调整参数需在结果中说明，不覆盖原记录。

## 指标、图形和计时口径

- 每轮训练指标：该轮各批次更新前的预测汇总，模型处于训练模式；每轮验证指标：该轮结束后固定模型、BN 评价模式。两条曲线的计算时点不同，不能视为同一固定模型的直接泛化差。
- loss 为按样本加权平均的交叉熵，不包括 AdamW 的解耦权重衰减。绘图使用全部原始逐轮记录，不平滑、不插值。
- 另对最佳模型在训练／验证／测试集进行固定权重、BN 评价模式评估，用于可比的泛化差分析。
- 总准确率=预测正确数/样本数。每类准确率=该真实类别中预测正确数/该类样本数；在标准指标术语中等价于该类召回率。
- 混淆矩阵行是真实类别、列是预测类别；CSV 同时给原始计数和按真实类别归一化比例，图中显示百分比。
- 错例每个真实类别选一个：预测置信度最接近该类全部错例置信度中位数，平局取较小测试索引。正确样本取相同真实类别、置信度最接近该错例者，平局同样取较小索引。这些是规则选出的示例，不是因果对照。图像直接显示原始 32×32 像素，以最近邻放大。
- 总训练用时含 60 轮训练、逐轮验证和最佳模型／历史记录写入；不含环境配置、数据下载／加载、最后权重保存、最终评价和绘图。每轮分项时间另存。推理用时含一个测试集遍历、CPU 数据加载、设备传输、损失与 softmax 计算及概率收集，不是纯模型计算时间。

## 输出目录

```text
data/                              官方数据与校验清单（自动创建）
outputs/checks/preflight.json       划分、归一化、矩阵、64 样本拟合等检查
outputs/baseline_seed42/
  config.json                      完整配置及源文件、划分哈希
  baseline_source.py               本次训练入口的源码快照
  environment.json                  实测环境
  model.txt, model_info.json        结构与参数量
  split_indices.npz                训练和验证索引
  history.csv                      逐轮原始记录
  best.pt, last.pt                 最佳及末轮权重
  training_summary.json            训练用时与最佳轮次
  metrics.json                     最佳模型三份数据的指标
  test_predictions.npz/.csv         按官方测试顺序保存预测
  validation_predictions.npz       验证预测及原训练集索引
  confusion_counts.csv             10×10 原始计数
  confusion_row_normalized.csv     逐行归一化比例
  per_class.csv                    每类样本数、正确数、准确率
  example_selection.csv            示例索引与选择依据
  descriptive_analysis.json        混淆方向、错例置信度等描述性统计
  verification.json                指标独立核算、检查点重载复核
  figures/                         曲线、混淆矩阵、错例和正确样本网格
```

预测 CSV 每行包含 `test_index,true_class,predicted_class,confidence,correct`，共 10,000 行；置信度为预测类别 softmax 值，不代表已经校准的正确概率。NPZ 另外保存所有十类概率和标签编号。曲线和混淆矩阵同时导出 PNG、SVG。

## 已实施的核查及限制

`preflight` 验证划分互斥、覆盖完整、类别平衡且可重复，检查归一化边界和混淆矩阵方向，并用 64 个真实训练样本进行 100 步拟合。检查权重被丢弃，正式运行重新设种子并创建模型。

`verify` 独立逐样本累计混淆矩阵，用保存概率重新计算交叉熵，核对 60 轮历史、选择规则、样本数、CSV、检查点哈希及类别统计，并在新的进程重新加载模型、复核全部 10,000 张测试图像的预测。重复推理仅验证保存模型，不用于选择参数。

正式结果使用单一种子。已在同机隔离目录复跑 A0、AFC、SE 和标签平滑的训练及测试，尚未估计跨种子波动。启用了确定性算法，跨版本、跨硬件的一致性仍需另行核对，相关限制见 [PyTorch 官方复现说明](https://docs.pytorch.org/docs/2.7/notes/randomness.html)。

## 三因素数据增强消融

此阶段复用原无增强基线 A0，新增 AF、AC、AB、AFC、AFB、ACB、AFCB 七组。F 为水平翻转（p=0.5），C 为补 4 像素黑边后随机裁剪 32×32，B 为随机亮度系数 Uniform[0.8,1.2)。每组 60 轮，模型、划分、初始化、样本顺序及优化器沿用基础实验；以验证集选择模型，本阶段不新增测试集评价。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 augmentation_ablation.py preflight
.\.venv\Scripts\python.exe -B -X utf8 augmentation_ablation.py train
.\.venv\Scripts\python.exe -B -X utf8 analyze_augmentation.py
```

原结果目录为 `outputs/augmentation_seed42/`。训练入口默认顺序完成七组；`--groups AF AC` 可指定尚未运行的组，但不允许重训 A0 或覆盖已有组。需要在全新结果目录复现时，为三个命令均传入同一个 `--out outputs/augmentation_reproduction`。`preflight` 和 `train` 支持 `--data-dir`、`--device cuda`；分析入口同样支持这些参数。没有中途恢复功能，失败时不覆盖既有配置。

实现沿用基础实验的 CPU 预归一化张量和 DataLoader。补边在归一化空间填 -1，等价于原图黑像素；翻转在宽度维操作；亮度先反归一化到 [0,1] 浮点 RGB，乘系数并截断，再归一化，不作 uint8 重新量化。先裁剪、再翻转、再调亮度，数值等价性已通过独立 NumPy 逐像素检查（最大绝对误差约 1.19e-7）。

增强参数使用独立 CPU Generator：F=1042、C=2042、B=3042。每轮为每个训练样本呈现生成新参数，再按全组一致的样本顺序分批应用于 GPU。因每个样本每轮只呈现一次，这等价于访问时独立采样。相同因素在所有启用组中共享同一参数序列，不消耗模型默认 RNG。

预检查比较原入口与关闭增强的新入口：前三批数据相同、两步权重及 BatchNorm 更新逐位相同；另根据基础 DataLoader 与种子重建 60 轮样本顺序哈希。原基础运行没有保存顺序哈希，因此它是由原协议重建的参照；新增各组会核对实际顺序，并保存各因素逐轮参数校验值。

各组子目录包含源码快照、配置、环境、划分、`history.csv`、`augmentation_trace.jsonl`、最佳及末轮模型、干净训练集／验证指标、验证预测与混淆矩阵。根目录汇总 `comparison.csv`、`validation_per_class.csv`、`conditional_effects.csv`、`main_effects.csv`、`factorial_analysis.json`、`verification.json` 及确认布局的图形。A0 验证预测取自已有基础记录，不将其测试指标混入本轮比较。

分析入口独立复算样本数、交叉熵、混淆矩阵、原错例修正／新引入错误，并核对所有轮次的样本顺序、对应因素的随机序列及选择规则；在新的进程重载七个最佳模型，复核全部验证预测。曲线全部使用原始记录、不平滑；增强组在线训练曲线来自增强输入，各组泛化差距则统一用干净训练集固定评价计算。

计时范围沿用基础训练与逐轮验证，并包含在线增强参数生成、设备运算和校验记录；相对原基线多出的少量核查开销也包含在内，不把所有耗时差都归因于增强运算。七组依次运行。每个条件仅一个种子，不将四个条件下的差值视作四次独立随机重复，也不以验证成绩证明独立测试集上的提升。

## 已锁定的 AFC 最终测试

八组验证比较完成后，用户确认使用 AFC（翻转＋裁剪）第 47 轮。`final_test_afc.py` 在评价前校验固定权重 SHA-256、组别和轮次，写入锁定记录，再复用基础评价流程。最终阶段没有新增训练或调参。结果与分析见 [AFC 测试指标](evidence/final_afc_epoch47/metrics.json)。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 final_test_afc.py evaluate
.\.venv\Scripts\python.exe -B -X utf8 checks.py verify --run-dir outputs/final_afc_epoch47 --device cuda
.\.venv\Scripts\python.exe -B -X utf8 final_test_afc.py compare
.\.venv\Scripts\python.exe -B -X utf8 plot_results.py --run-dir outputs/final_afc_epoch47
```

若本地默认目录 `outputs/final_afc_epoch47/` 已存在，评价入口会拒绝覆盖。需要复核评价时，第一条命令指定新的 `--run-dir outputs/final_afc_epoch47_recheck`，其余三条均传入相同新目录。比较使用已有基础模型测试预测，`checks.py verify` 重新加载锁定模型复核预测，不据此改变选择。

最终目录包含 `lock.json`、锁定权重及训练源码／配置快照、最终 `metrics.json`、10,000 行测试预测、每类结果和混淆矩阵、`baseline_comparison.json`、`per_class_comparison.csv`、逐样本变化表、核查结果和图形。AFC 的 `evaluate` 命令写入 `evaluation_environment.json`，供 SE 和标签平滑的测试对照核对运行环境。`training_config.json`、`validation_stage_metrics.json` 保留测试前阶段记录，最终 `config.json` 标记本次已完成测试。

## SE 通道注意力对照

与已有 AFC 比较，只新增一组 SE-AFC 的60轮正式训练。每块第二次 ReLU 之后、最大池化之前加入 SE，缩减比例16，两层全连接带偏置。共有层从原种子42的初始 CNN 复制，SE 参数使用独立种子4042；没有加载训练好的 AFC 微调。全部60轮的样本顺序、翻转与裁剪参数哈希均与 AFC 实际记录一致。其余训练与验证协议保持不变。第48轮由验证结果选出并锁定，随后在官方测试集上与 AFC 比较。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 se_ablation.py preflight
.\.venv\Scripts\python.exe -B -X utf8 se_ablation.py train
.\.venv\Scripts\python.exe -B -X utf8 se_ablation.py verify
.\.venv\Scripts\python.exe -B -X utf8 analyze_se.py
```

默认结果目录为 `outputs/se_seed42/`。`preflight`、`train` 拒绝覆盖已有正式配置；需要重新训练时，为四条命令传入同一个新目录，例如 `--out outputs/se_reproduction`。引用的原 AFC 目录保持不变。`verify` 是新进程重载推理复核，不是完整训练重跑。

该目录包含源码快照、共享/新增初始化哈希、预检查、完整配置、每轮记录与增强追踪、最佳和末轮权重、干净训练/验证指标、5,000张验证预测、配对错误变化、每类对照及成本测量。`analyze_se.py --plot-only` 仅按保存结果重绘左右双图，不重新测量或训练。[SE 验证阶段记录](evidence/se_seed42/metrics.json)对应选权重阶段，锁定后的测试结果与逐图变化见 [SE 测试对照](evidence/final_se_afc_epoch48/comparison_analysis.json)。

成本对照在同一进程对两模型分别预热一遍，再以交替次序对相同5,000张验证图评价5遍，记录完整范围和中位数。包含CPU DataLoader、传输、前向、CE与准确率，不能当作纯模型内核时延。历史训练时间只是另行保留的描述性记录。

## Grad-CAM 与参数敏感性分析

使用 A0 第59轮和 AFC 第47轮。主图沿用生成热图前固定的四张 A0 测试错例（#3094、#262、#58、#6419），展示 AFC 修正和仍判错的案例各两张。补充分析另在验证集按类选样。每类取一张置信度最接近该类 AFC 错例置信度中位数的错例，再取置信度最接近它的同类 AFC 正确例，平局取较小的原始索引，共20张。选样在计算热图前完成，规则与索引见[选样记录](evidence/gradcam_a0_afc/selection.json)。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 gradcam_analysis.py compute
.\.venv\Scripts\python.exe -B -X utf8 gradcam_analysis.py verify
.\.venv\Scripts\python.exe -B -X utf8 gradcam_analysis.py plot
.\.venv\Scripts\python.exe -B -X utf8 extension_results.py
```

默认目录为 `outputs/gradcam_a0_afc/`，计算入口拒绝覆盖既有热图。复现时前三条命令传入同一个新的 `--out`；最后一条用 `--cam-out` 指向该目录、用 `--se-out` 指向对应 SE 结果目录，以生成阶段记录。默认命令复核和汇总当前保留的原始运行。

原生目标层为最终池化前的128×8×8特征。主图比较相同真实类别，补充图比较相同 A0 预测类别；解释目标与实际预测分开标注。逐图按自身正贡献最大值归一化，不能跨模型比较颜色强弱；原始8×8幅度、激活和梯度保存在 `maps.npz`。20张验证图另在 AFC 副本的随机分类头上做参数敏感性核查，不训练、不写回原模型。

`verify` 在独立进程重新选样和加载权重，复算全部主分析原生/归一化热图及logit，核对模型状态和原始文件哈希。原网络尾部还允许独立推导通道梯度权重，另以NumPy复算热图。参数随机化只检验敏感性，没有定位标注，不将图面或相关系数当作解释真实性的证明。

主图采用2×2样本组布局，每组原图/A0/AFC三列。另存4行×3列大图及全部20张验证图的补充页，均为PNG/SVG。两模型分别对真实类别和 A0 预测类别计算热图。两个目标相同时复用对应热图，共得到72张主分析热图。复核见[热图复核记录](evidence/gradcam_a0_afc/verification.json)，完整图形由上面的 `plot` 命令写入本地 `outputs/gradcam_a0_afc/figures/`。

## 标签平滑对照

以 AFC 第47轮为对照，按相同协议从头训练60轮，只将训练交叉熵的标签平滑系数从0改为0.1。程序逐轮核对样本顺序及翻转、裁剪参数。轮次仍按验证准确率优先、真实标签交叉熵次优先选取。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 loss_ablation.py train
.\.venv\Scripts\python.exe -B -X utf8 loss_ablation.py verify
```

[标签平滑验证对照](evidence/loss_label_smoothing_seed42/comparison.json)给出同一批5,000张验证图像上的准确率、真实标签交叉熵、置信度至少90%的正确／错误预测计数及15等宽分箱的校准误差。第43轮按验证结果锁定后，另行完成[官方测试集对照](evidence/final_label_smoothing_epoch43/comparison.json)。

## 测试对照与复核

AFC 第47轮先由八组增强的验证结果选出，首次官方测试准确率为 87.76%。SE-AFC 与标签平滑的轮次也分别根据验证结果在测试前锁定。后续探索性对照复用同一批10,000张官方测试图像，逐图对应 AFC 的已保存预测。实验最终保留 AFC 第47轮。

| 已锁定权重 | 测试准确率 | 结果入口 | 独立重载复核 |
|---|---:|---|---|
| AFC 第47轮 | 87.76% | [测试指标](evidence/final_afc_epoch47/metrics.json) | [复核记录](evidence/final_afc_epoch47/verification.json) |
| SE-AFC 第48轮 | 86.58% | [对照指标与配对统计](evidence/final_se_afc_epoch48/comparison_analysis.json)、[混淆计数](evidence/final_se_afc_epoch48/confusion_counts.csv) | [复核记录](evidence/final_se_afc_epoch48/verification.json) |
| 标签平滑组第43轮 | 87.17% | [准确率、高置信度错例、校准误差与配对统计](evidence/final_label_smoothing_epoch43/comparison.json) | [复核记录](evidence/final_label_smoothing_epoch43/verification.json) |

三组权重独立重载后均复现了各自的测试预测和概率。与 AFC 对照时，SE 与标签平滑的测试样本顺序及真实标签均逐项一致。

### 复核已保存的 SE 与 Grad-CAM 结果

在本地已生成该阶段的实验输出后，从实验目录运行对应命令。SE 验证和 Grad-CAM 复核分别使用训练与分析阶段的输出。SE 测试复核需要该次锁定测试生成的完整结果目录和原权重。仓库中的摘要记录可直接浏览。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 se_ablation.py verify
.\.venv\Scripts\python.exe -B -X utf8 gradcam_analysis.py verify
.\.venv\Scripts\python.exe -B -X utf8 final_test_se.py verify
```

SE、SE 测试和 Grad-CAM 的输入校验覆盖代码与实验产物。旧清单中的 `report/` 条目会被排除，Word 报告可以独立修订。

### 从新目录复现 SE 和标签平滑的测试

在新的检出目录进入本实验，先用前文的基础运行命令完成 A0 的数据准备、训练与评价。只复现这两个扩展时，增强消融的训练命令可以只运行 AFC。以下命令使用与原实验相同的 CUDA 环境和默认训练输出目录。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 augmentation_ablation.py preflight
.\.venv\Scripts\python.exe -B -X utf8 augmentation_ablation.py train --groups AFC
.\.venv\Scripts\python.exe -B -X utf8 final_test_afc.py evaluate

.\.venv\Scripts\python.exe -B -X utf8 se_ablation.py preflight
.\.venv\Scripts\python.exe -B -X utf8 se_ablation.py train
.\.venv\Scripts\python.exe -B -X utf8 se_ablation.py verify
.\.venv\Scripts\python.exe -B -X utf8 final_test_se.py evaluate --mode reproduced --run-dir outputs/reproduced_se_test
.\.venv\Scripts\python.exe -B -X utf8 final_test_se.py verify --run-dir outputs/reproduced_se_test

.\.venv\Scripts\python.exe -B -X utf8 loss_ablation.py train
.\.venv\Scripts\python.exe -B -X utf8 loss_ablation.py verify
.\.venv\Scripts\python.exe -B -X utf8 final_test_label_smoothing.py evaluate --mode reproduced --run-dir outputs/reproduced_label_smoothing_test
.\.venv\Scripts\python.exe -B -X utf8 final_test_label_smoothing.py verify --run-dir outputs/reproduced_label_smoothing_test
```

先用验证集选权重并独立重载核对。`reproduced` 模式将这次训练选出的轮次和检查点哈希写入 `lock.json`，再与 AFC 第47轮在官方测试集上逐图比较。测试输出写入单独目录，`verify` 读取其中的锁定记录。若训练时指定了本检出目录内的另一个 `--out`，评价时用 `--training-run` 指向它。若数据放在其他目录，各训练、评价和复核命令均可使用 `--data-dir`。

复核原有已锁定的 SE 和标签平滑时，沿用上节原结果目录，历史评价入口的默认模式为 `archived`，仍要求原第48／43轮检查点的固定哈希。两种模式都使用原 AFC 第47轮作为参照。新目录下的 AFC 评价会校验其历史哈希，因此重训 AFC 需使用可复现该锁定权重的环境。

同机从空的扩展结果目录各重训60轮后，SE-AFC 复现第48轮和86.58%的测试准确率，标签平滑复现第43轮和87.17%的测试准确率。两组的5,000张验证预测及10,000张测试预测均通过独立重载复核，测试结果与各自的历史记录逐图一致。

[SE 重训测试](evidence/reproduced_se_test/comparison_analysis.json) · [SE 重训复核](evidence/reproduced_se_test/verification.json) · [标签平滑重训测试](evidence/reproduced_label_smoothing_test/comparison.json) · [标签平滑重训复核](evidence/reproduced_label_smoothing_test/verification.json)

#### 重训验证与历史预测逐图核对

- [SE-AFC 第48轮重训后的验证集重载复核](evidence/reproduced_se_test/validation_verification.json)
- [标签平滑第43轮重训后的验证集重载复核](evidence/reproduced_label_smoothing_test/validation_verification.json)
- [新旧预测数组的 SHA-256 和逐图差异计数](evidence/prediction_array_comparison.json)

逐图核对先确认样本索引和真实标签逐项相同，再比较类别预测与十类概率。两组各5,000张验证图像和各10,000张测试图像的差异行数均为0，概率最大绝对差均为0。校验值使用小端、C顺序的数组原始字节计算，文件中另记数组形状和数据类型。
