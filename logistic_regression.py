"""Softmax 多分类逻辑回归，使用小批量 Adam。

目标 = 平均数据损失 + ||W||²/(2*C*n_train)，偏置不正则化。
数据损失为多分类交叉熵。
"""
import numpy as np
from scipy.special import logsumexp, softmax


class LinearTextClassifier:
    def __init__(self, kind, C=1.0, learning_rate=0.01, batch_size=256, seed=42):
        if kind != "logistic" or C <= 0:
            raise ValueError("kind 必须为 logistic，C 必须为正数")
        self.kind, self.C = kind, C
        self.learning_rate, self.batch_size, self.seed = learning_rate, batch_size, seed

    def initialize(self, n_features, y):
        self.classes_ = np.unique(y)
        self.weights_ = np.zeros((n_features, len(self.classes_)), dtype=float)
        self.bias_ = np.zeros(len(self.classes_))
        self.reg_ = 1.0 / (self.C * len(y))
        self.rng_ = np.random.default_rng(self.seed)
        self.mw_, self.vw_ = np.zeros_like(self.weights_), np.zeros_like(self.weights_)
        self.mb_, self.vb_ = np.zeros_like(self.bias_), np.zeros_like(self.bias_)
        self.step_ = 0
        return self

    def decision_function(self, X):
        return np.asarray(X @ self.weights_) + self.bias_

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    def _data_loss_delta(self, X, y):
        indices = np.searchsorted(self.classes_, y)
        scores = self.decision_function(X)
        rows = np.arange(len(y))
        loss = np.mean(logsumexp(scores, axis=1) - scores[rows, indices])
        delta = softmax(scores, axis=1)
        delta[rows, indices] -= 1
        return float(loss), delta / len(y)

    def losses(self, X, y):
        loss, _ = self._data_loss_delta(X, y)
        return loss, float(loss + 0.5 * self.reg_ * np.sum(self.weights_ ** 2))

    def loss_gradient(self, X, y):
        loss, delta = self._data_loss_delta(X, y)
        objective = loss + 0.5 * self.reg_ * np.sum(self.weights_ ** 2)
        return objective, np.asarray(X.T @ delta) + self.reg_ * self.weights_, delta.sum(axis=0)

    def train_epoch(self, X, y, record_batches=False):
        batch_history = []
        order = self.rng_.permutation(len(y))
        for start in range(0, len(y), self.batch_size):
            ix = order[start:start + self.batch_size]
            objective, gw, gb = self.loss_gradient(X[ix], y[ix])
            if record_batches:
                # 更新前记录当前批次的交叉熵和含正则项的目标值。
                penalty = 0.5 * self.reg_ * np.sum(self.weights_ ** 2)
                batch_history.append(dict(step=self.step_ + 1, batch_size=len(ix),
                                          batch_loss=float(objective - penalty),
                                          batch_objective=float(objective)))
            self.step_ += 1
            for param, grad, m, v in [(self.weights_, gw, self.mw_, self.vw_),
                                       (self.bias_, gb, self.mb_, self.vb_)]:
                m *= 0.9
                m += 0.1 * grad
                v *= 0.999
                v += 0.001 * grad ** 2
                param -= self.learning_rate * (m / (1 - 0.9 ** self.step_)) / (
                    np.sqrt(v / (1 - 0.999 ** self.step_)) + 1e-8)
        if not np.isfinite(self.weights_).all():
            raise FloatingPointError("训练出现非有限值，请检查数据或降低学习率")
        return batch_history
