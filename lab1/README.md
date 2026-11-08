# 新闻文本分类实验

本项目使用 TF-IDF 特征比较多项式朴素贝叶斯与 Softmax 逻辑回归，程序入口为 main.py。

## 代码结构

| 文件 | 用途 |
|---|---|
| main.py | 数据划分、TF-IDF、参数搜索、模型选择、训练与预测 |
| logistic_regression.py | 交叉熵梯度和小批量 Adam |
| naive_bayes_training.py | 增量统计、交叉熵评价、贝叶斯学习曲线 |
| experiment_utils.py | 分类指标、分类报告及混淆矩阵 |
| test_logistic_regression.py | 数值梯度、学习和复现性检查 |
| test_naive_bayes_training.py | 增量拟合与全量拟合的等价性检查 |
| requirements.txt | 运行依赖 |
| train_data.csv | 全部带标签数据，运行时划分训练集和验证集 |
| test_data_unlabeled.csv | 无标签测试数据 |

## 运行与复现
使用 Python 3.10 或更新版本。仓库不包含课程数据，请将课程提供的 train_data.csv 和 test_data_unlabeled.csv 放到 main.py 所在目录。在该目录打开终端，安装依赖并运行：

```powershell
python -m pip install -r requirements.txt
python main.py
```

可单独运行测试，检查梯度计算、固定种子复现和贝叶斯增量拟合：

```powershell
python -m unittest discover -p "test_*.py"
```

数据通过相对路径读取，运行时须以 main.py 所在目录为工作目录。程序自动创建 results/，保存预测、图表、训练记录和模型。最终预测文件为 results/predictions.csv，无表头、无索引，共 2457 行，顺序与测试数据一致。

以下命令显式指定默认实验参数：

```powershell
python main.py --alpha-values 0.01 0.1 1 10 --c-values 0.1 1 10 100 --epochs 40 --batch-size 256 --learning-rate 0.01
```
使用 --output-dir 可另存实验。run_config.json 是运行记录，不是配置输入。

## 固定实验设置
7368 条标签数据按 80%/20% 分层划分为 5894 条训练和 1474 条验证，随机种子 42。词表及 IDF 仅在训练部分拟合，最多 5000 特征，默认小写转换、L2 归一化。测试集 2457 条仅作转换与预测。

朴素贝叶斯比较 alpha=0.01、0.1、1、10，以 partial_fit 分批累加一次数据；最终结果与一次性 fit 等价（浮点精度误差除外）。alpha 是加性平滑系数，不是学习率。

逻辑回归比较 C=0.1、1、10、100，用小批量 Adam，默认 40 轮、批次 256、学习率 0.01。目标为平均交叉熵加 ||W||²/(2*C*n)，不惩罚偏置；C 越小正则化越强。每组训练满 40 轮并保存验证指标最佳轮次权重。

模型内按验证 Macro-F1、准确率、较低验证交叉熵排序；跨模型只比较前两项，完全同分选择贝叶斯。锁定参数后，在全部标签数据重新拟合 TF-IDF 并训练最终模型。

## 结果文件与三种曲线
以下文件将在运行后生成于 results/ 下：

- validation_results.csv：8 组配置验证指标；贝叶斯无 C、epoch，留空。
- loss_curves.png：逻辑回归每轮结束后，全量训练/验证集平均交叉熵；0 轮为初始化。
- batch_loss_curves.png：逻辑回归当前批次更新前的损失。40 轮共 960 次更新，末批 6 条，局部波动较大。没有平滑。
- naive_bayes_learning_curve.png：累计统计样本数与固定验证集交叉熵。不是 epoch 收敛曲线。TF-IDF 在完整训练部分拟合后固定，横轴仅指进入贝叶斯统计的样本数。
- logistic_C*_history.csv、logistic_C*_batch_history.csv：逐轮/逐批记录；含正则项目标另列保存。
- nb_alpha*_learning_history.csv：累计样本数、固定训练集/验证集交叉熵；早期完整训练集含尚未加入统计的样本，仅作评价。
- *_classification_report.csv、*_confusion_matrix.csv/png：各类别评价。
- *_validation_model.joblib、*_final_model.joblib：向量化器与对应模型。
- predictions_nb.csv、predictions_logistic.csv、predictions.csv：两种预测及获胜模型预测。
- split_indices.csv、run_config.json：划分与参数环境记录。

默认配置的参考结果：最佳朴素贝叶斯 alpha=0.1，验证准确率 89.15%、Macro-F1 89.22%；最佳逻辑回归 C=10、第 36 轮，准确率 91.32%、Macro-F1 91.35%。实际运行结果见 results/validation_results.csv 和 results/run_config.json。测试集没有公开标签，不能据此报告测试准确率。
