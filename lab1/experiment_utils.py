"""分类指标、分类报告和混淆矩阵。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

BASE_DIR = Path(__file__).resolve().parent


def evaluate(model, X, y):
    prediction = model.predict(X)
    return {"accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0))}


def save_evaluation(model, X, y, output_dir, name):
    prediction = model.predict(X)
    report = classification_report(y, prediction, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / f"{name}_classification_report.csv")
    matrix = confusion_matrix(y, prediction, labels=np.arange(10))
    pd.DataFrame(matrix, index=np.arange(10), columns=np.arange(10)).to_csv(
        output_dir / f"{name}_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(10), yticks=np.arange(10), xlabel="Predicted class",
           ylabel="True class", title=f"{name}: validation confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=180)
    plt.close(fig)
