# my_experiment.py

# 1. 从我们提供的帮助脚本中导入加载函数
import pandas as pd

def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled

# 2. 调用函数来获取数据
#    这个函数会自动读取 .csv 文件并返回你需要的所有内容
X_train, y_train, X_test_unlabeled = load_data()

# 3. (验证步骤) 检查一下数据是否加载成功
print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 20)

# 打印第一个训练样本和它的标签，感受一下数据
print("第一个训练样本内容:")
print(X_train[0])
print(f"\n第一个训练样本的标签: {y_train[0]}")
print("-" * 20)

# 打印第一个需要你预测的测试样本
print("第一个无标签测试样本内容:")
print(X_test_unlabeled[0])
print("\n" + "="*50)

# --- 在这里开始你的实验！ ---
# 现在，你可以使用 X_train, y_train, 和 X_test_unlabeled 这三个变量
# 来进行后续的特征提取、模型训练和预测了。

# 比较多项式朴素贝叶斯和 Softmax 逻辑回归。
import argparse
import copy
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from naive_bayes_training import cross_entropy, train_naive_bayes, plot_naive_bayes_losses

from experiment_utils import BASE_DIR, evaluate, plt, save_evaluation
from logistic_regression import LinearTextClassifier


def train_logistic_regression(C, X_train, y_train, X_val, y_val, args):
    # 每组参数从零初始化，使用相同随机种子。
    model = LinearTextClassifier('logistic', C, args.learning_rate, args.batch_size, args.seed)
    model.initialize(X_train.shape[1], y_train)
    history, best_model, best_key, best_epoch = [], None, None, 0
    batch_history = []
    for epoch in range(args.epochs + 1):
        if epoch:  # 第 0 轮是未训练的初始状态。
            batches = model.train_epoch(X_train, y_train, record_batches=True)
            batch_history.extend(dict(epoch=epoch, **row) for row in batches)
        # 逐轮评价完整训练集和验证集，验证数据不参与更新。
        train_loss, objective = model.losses(X_train, y_train)
        val_loss, _ = model.losses(X_val, y_val)
        metrics = evaluate(model, X_val, y_val)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            train_objective=objective, **metrics))
        # 按 Macro-F1、准确率、较低损失择优，复制权重以免被后续更新覆盖。
        key = (metrics['macro_f1'], metrics['accuracy'], -val_loss)
        if epoch and (best_key is None or key > best_key):
            best_key, best_epoch, best_model = key, epoch, copy.deepcopy(model)
    return best_model, best_epoch, pd.DataFrame(history), pd.DataFrame(batch_history)


def plot_losses(best, output_dir):
    # 逻辑回归逐轮损失，虚线标记选定轮次。
    fig, ax = plt.subplots(figsize=(6, 3.6))
    kind = 'logistic'
    entry = best[kind]
    history = entry['history']
    # 每轮一个点，直接连接原始记录。
    ax.plot(history.epoch, history.train_loss, label='Training',
            linestyle='-', marker='o', markersize=3, linewidth=1.2)
    ax.plot(history.epoch, history.val_loss, label='Validation',
            linestyle='-', marker='s', markersize=3, linewidth=1.2)
    ax.axvline(entry['epoch'], color='gray', linestyle='--', label='Selected epoch')
    ax.set(xlabel='Epoch', ylabel='Mean data loss (no regularization)',
           title=f"{kind}: C={entry['C']:g}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'loss_curves.png', dpi=200)
    plt.close(fig)


def plot_batch_losses(best, output_dir):
    # 每个点是更新前当前批次的交叉熵，不含正则项。
    fig, ax = plt.subplots(figsize=(6, 3.6))
    kind = 'logistic'
    entry = best[kind]
    history = entry['batch_history']
    ax.plot(history.step, history.batch_loss, linewidth=0.7,
            marker='.', markersize=2, label='Raw mini-batch loss')
    ax.set(xlabel='Update step (loss measured before update)',
           ylabel='Mean mini-batch data loss', title=f"{kind}: C={entry['C']:g}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'batch_loss_curves.png', dpi=200)
    plt.close(fig)


def parse_args():
    # 默认参数可由命令行覆盖；轮数和学习率仅用于逻辑回归。
    parser = argparse.ArgumentParser(description='TF-IDF + 朴素贝叶斯 / Softmax 逻辑回归实验')
    parser.add_argument('--output-dir', type=Path, default=BASE_DIR / 'results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-size', type=float, default=0.2)
    parser.add_argument('--max-features', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--c-values', nargs='+', type=float, default=[0.1, 1, 10, 100])
    parser.add_argument('--alpha-values', nargs='+', type=float, default=[0.01, 0.1, 1, 10])
    args = parser.parse_args()
    if not 0 < args.validation_size < 1:
        parser.error('validation-size 必须在 0 和 1 之间')
    if min(args.epochs, args.batch_size, args.max_features) < 1:
        parser.error('epochs、batch-size、max-features 必须为正整数')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('learning-rate 必须为有限正数')
    if any(not np.isfinite(c) or c <= 0 for c in args.c_values):
        parser.error('C 必须为有限正数')
    if any(not np.isfinite(a) or a <= 0 for a in args.alpha_values):
        parser.error('alpha 必须为有限正数')
    if args.output_dir.resolve() == Path.cwd().resolve():
        parser.error('输出目录必须与数据目录不同，以保护示例 predictions.csv')
    return args


def main(X_train, y_train, X_test_unlabeled):
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 分层划分训练集和验证集，保留全部标签数据供最终重训。
    texts, labels, test_texts = X_train, y_train, X_test_unlabeled
    train_ix, val_ix = train_test_split(np.arange(len(y_train)),
        test_size=args.validation_size, stratify=y_train, random_state=args.seed)
    pd.DataFrame({'row_index': np.concatenate([train_ix, val_ix]),
        'split': ['train'] * len(train_ix) + ['validation'] * len(val_ix)}).to_csv(
            args.output_dir / 'split_indices.csv', index=False)
    print(f'Training={len(train_ix)}, validation={len(val_ix)}, test={len(test_texts)}', flush=True)

    # 1. 创建 TF-IDF 向量化器
    #    词表和 IDF 仅在训练集上拟合。
    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_train_tfidf = vectorizer.fit_transform([texts[i] for i in train_ix])
    X_val_tfidf = vectorizer.transform([texts[i] for i in val_ix])
    y_train, y_val = labels[train_ix], labels[val_ix]

    # 2. 训练模型
    #    用同一验证集比较贝叶斯的 alpha 和逻辑回归的 C。
    results, best = [], {}
    for alpha in args.alpha_values:
        start = time.perf_counter()
        # 分批统计一遍训练数据，并记录每批更新后的损失。
        model, nb_history = train_naive_bayes(
            alpha, X_train_tfidf, y_train, X_val_tfidf, y_val,
            batch_size=args.batch_size, seed=args.seed)
        nb_history.to_csv(args.output_dir / f'nb_alpha{alpha:g}_learning_history.csv', index=False)
        metrics = evaluate(model, X_val_tfidf, y_val)
        # 全部数据统计完毕后的验证交叉熵。
        val_loss = cross_entropy(model, X_val_tfidf, y_val)
        elapsed = time.perf_counter() - start
        results.append(dict(model='nb', alpha=alpha, C=None, selected_epoch=None,
                            **metrics, val_loss=val_loss, seconds=elapsed))
        key = (metrics['macro_f1'], metrics['accuracy'], -val_loss)
        if 'nb' not in best or key > best['nb']['key']:
            best['nb'] = dict(model=model, alpha=alpha, epoch=None, key=key,
                              learning_history=nb_history)
        print(f"nb alpha={alpha:g}: accuracy={metrics['accuracy']:.4f}, "
              f"macro-F1={metrics['macro_f1']:.4f} ({elapsed:.1f}s)", flush=True)
    save_evaluation(best['nb']['model'], X_val_tfidf, y_val, args.output_dir, 'nb')
    joblib.dump({'vectorizer': vectorizer, 'model': best['nb']['model']},
                args.output_dir / 'nb_validation_model.joblib')

    kind = 'logistic'
    for C in args.c_values:
        start = time.perf_counter()
        model, epoch, history, batch_history = train_logistic_regression(
            C, X_train_tfidf, y_train, X_val_tfidf, y_val, args)
        elapsed = time.perf_counter() - start
        history.to_csv(args.output_dir / f'{kind}_C{C:g}_history.csv', index=False)
        batch_history.to_csv(args.output_dir / f'{kind}_C{C:g}_batch_history.csv', index=False)
        metrics = evaluate(model, X_val_tfidf, y_val)
        val_loss = model.losses(X_val_tfidf, y_val)[0]
        results.append(dict(model=kind, C=C, selected_epoch=epoch, **metrics,
                            val_loss=val_loss, seconds=elapsed))
        key = (metrics['macro_f1'], metrics['accuracy'], -val_loss)
        if kind not in best or key > best[kind]['key']:
            best[kind] = dict(model=model, C=C, epoch=epoch, history=history,
                              batch_history=batch_history, key=key)
        print(f"{kind} C={C:g} epoch={epoch}: accuracy={metrics['accuracy']:.4f}, "
              f"macro-F1={metrics['macro_f1']:.4f} ({elapsed:.1f}s)", flush=True)
    # 保存逻辑回归最佳配置的评价结果和模型。
    chosen = best[kind]
    save_evaluation(chosen['model'], X_val_tfidf, y_val, args.output_dir, kind)
    joblib.dump({'vectorizer': vectorizer, 'model': chosen['model']},
                args.output_dir / f'{kind}_validation_model.joblib')
    # 汇总所有配置的指标，绘制两种模型各自最佳配置的曲线。
    pd.DataFrame(results).to_csv(args.output_dir / 'validation_results.csv', index=False)
    plot_losses(best, args.output_dir)
    plot_batch_losses(best, args.output_dir)
    plot_naive_bayes_losses(best['nb']['learning_history'], best['nb']['alpha'], args.output_dir)

    # 3. 对测试集进行预测
    #    固定选出的参数和轮次，在全部标签数据上重训。
    final_vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_all = final_vectorizer.fit_transform(texts)
    X_test_tfidf = final_vectorizer.transform(test_texts)
    # 按 Macro-F1、准确率选模型；同分时选择贝叶斯。
    winner = max(best, key=lambda kind: best[kind]['key'][:2])
    for kind, chosen in best.items():
        if kind == 'nb':
            final = MultinomialNB(alpha=chosen['alpha']).fit(X_all, labels)
        else:
            final = LinearTextClassifier(kind, chosen['C'], args.learning_rate, args.batch_size, args.seed)
            final.initialize(X_all.shape[1], labels)
            for _ in range(chosen['epoch']):
                final.train_epoch(X_all, labels)
        predictions = final.predict(X_test_tfidf)
        if len(predictions) != len(test_texts) or not np.isin(predictions, np.arange(10)).all():
            raise ValueError('预测数量或标签不符合提交要求')
        # 4. 保存预测结果
        #    按测试集原顺序输出，无表头和索引；获胜模型另存为 predictions.csv。
        pd.DataFrame(predictions).to_csv(args.output_dir / f'predictions_{kind}.csv', index=False, header=False)
        joblib.dump({'vectorizer': final_vectorizer, 'model': final},
                    args.output_dir / f'{kind}_final_model.joblib')
        if kind == winner:
            pd.DataFrame(predictions).to_csv(args.output_dir / 'predictions.csv', index=False, header=False)
    # 保存参数、环境版本和模型选择结果。
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
        n_train=len(train_ix), n_validation=len(val_ix), n_test=len(test_texts), selected_model=winner,
        selection_rule='within model: macro_f1, accuracy, negative val loss; across models: macro_f1, accuracy; tie: nb',
        best={k: ({'alpha': v['alpha'], 'epoch': None} if k == 'nb'
                  else {'C': v['C'], 'epoch': v['epoch']}) for k, v in best.items()},
        implementation='MultinomialNB and custom softmax regression with minibatch Adam')
    (args.output_dir / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Selected: {winner}; outputs: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main(X_train, y_train, X_test_unlabeled)
