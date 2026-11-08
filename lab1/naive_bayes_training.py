"""朴素贝叶斯增量拟合与学习曲线。"""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


def cross_entropy(model, X, y):
    log_probs = model.predict_log_proba(X)
    indices = np.searchsorted(model.classes_, y)
    return float(-log_probs[np.arange(len(y)), indices].mean())


def train_naive_bayes(alpha, X_train, y_train, X_val, y_val, batch_size=256, seed=42):
    """每条样本统计一次，最终与全量 fit 等价（浮点误差除外）。

    每批更新后评价完整训练集和固定验证集，不提前停止。
    早期训练损失包含尚未加入统计的样本；alpha 按完整拟合结果选择。
    """
    if batch_size < 1:
        raise ValueError('batch_size 必须为正整数')
    model = MultinomialNB(alpha=alpha)
    classes = np.unique(y_train)
    order = np.random.default_rng(seed).permutation(len(y_train))
    records = []
    for step, start in enumerate(range(0, len(y_train), batch_size), start=1):
        indices = order[start:start + batch_size]
        model.partial_fit(X_train[indices], y_train[indices], classes=classes)
        records.append(dict(step=step, samples_seen=start + len(indices),
                            batch_size=len(indices),
                            train_loss=cross_entropy(model, X_train, y_train),
                            val_loss=cross_entropy(model, X_val, y_val)))
    return model, pd.DataFrame(records)


def plot_naive_bayes_losses(history, alpha, output_dir):
    """绘制验证交叉熵随累计统计样本数的变化。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(history.samples_seen, history.val_loss, '-o', markersize=3,
            linewidth=1.2, label='Fixed validation set')
    ax.set(xlabel='Cumulative training samples',
           ylabel='Validation cross-entropy',
           title=f'Naive Bayes learning curve: alpha={alpha:g}')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'naive_bayes_learning_curve.png', dpi=200)
    plt.close(fig)
