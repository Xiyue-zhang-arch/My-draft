"""保存必要指标和图表；混淆矩阵按真实类别逐行归一化。"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from cwru_dataset import LABELS


def scores(labels, probabilities):
    predicted = probabilities.argmax(1)
    return dict(
        acc=float(accuracy_score(labels, predicted)),
        f1_macro=float(
            f1_score(
                labels, predicted, average="macro", labels=range(4), zero_division=0
            )
        ),
    )


def confusion_plot(labels, probabilities, destination):
    # 噪声实例分别判别后累加计数，避免把概率平均变成额外的集成模型。
    repeats = probabilities if probabilities.ndim == 3 else probabilities[None]
    counts = sum(
        confusion_matrix(labels, p.argmax(1), labels=range(4)) for p in repeats
    )
    fractions = counts / counts.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{fractions[i,j]:.2f}\n({counts[i,j]})",
                ha="center",
                va="center",
                color="white" if fractions[i, j] > 0.55 else "black",
            )
    ax.set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=LABELS,
        yticklabels=[f"{name} (n={sum(labels == i)})" for i, name in enumerate(LABELS)],
        xlabel="Predicted class",
        ylabel="True class",
        title=f"Window-level confusion\n{len(repeats)} observation(s) per window",
    )
    fig.colorbar(im, ax=ax, label="Row proportion")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return counts


def learning_curve(history, destination):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(history.updates, history.train_loss)
    valid = history.dropna(subset=["validation_nll"])
    axes[1].plot(valid.updates, valid.validation_nll, marker=".")
    axes[0].set(xlabel="Updates", ylabel="Training loss")
    axes[1].set(xlabel="Updates", ylabel="Validation NLL")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def summarize(metrics, windows, destination):
    """先对同一模型的噪声重复取均值，再对模型/子集组合汇总。"""
    keys = ["protocol", "method", "ratio", "subset", "seed", "condition"]
    per_model = metrics.groupby(keys, as_index=False)[["acc", "f1_macro"]].mean()
    summary = per_model.groupby(keys[:3] + ["condition"])[["acc", "f1_macro"]].agg(
        ["mean", "std", "count"]
    )
    summary.to_csv(destination / "summary.csv")
    for protocol in metrics.protocol.unique():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        data = per_model[
            (per_model.protocol == protocol) & (per_model.condition == "clean")
        ]
        for ax, metric in zip(axes, ["acc", "f1_macro"]):
            for method, group in data.groupby("method"):
                averages = group.groupby("ratio")[metric].mean()
                ax.plot(
                    averages.index * 100,
                    averages.values * 100,
                    marker="o",
                    label=method,
                )
            ax.set(
                xlabel="Training windows (%)", ylabel=f"{metric} (%)", title=protocol
            )
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(destination / f"{protocol}_accuracy.png", dpi=180)
        plt.close(fig)
    noisy = per_model[per_model.condition.str.startswith("white_")].copy()
    if not noisy.empty:
        noisy["snr"] = noisy.condition.str.removeprefix("white_").astype(int)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ["acc", "f1_macro"]):
            for (method, ratio), group in noisy.groupby(["method", "ratio"]):
                averages = group.groupby("snr")[metric].mean()
                ax.plot(
                    averages.index,
                    averages.values * 100,
                    marker=".",
                    label=f"{method} {ratio:.0%}",
                )
            ax.set(xlabel="SNR (dB)", ylabel=f"{metric} (%)")
        axes[1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        fig.savefig(destination / "noise_robustness.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    support = windows.groupby(["role", "label"]).size().unstack(fill_value=0)
    lines = [
        "# 本次复现实验",
        "",
        "ACC与F1按窗口计算，F1采用宏平均。",
        "",
        "划分支持数（列为Normal、IR、OR、B）",
        "",
        support.to_csv(),
        "噪声重复先在同一模型内取均值，不作为新增独立样本。跨种子/子集的标准差是描述性统计，不是置信区间。",
        "验证集标注预算独立于小样本训练比例。既有测试曝光仍需披露。",
        "本次为精简统一入口的重新实现；在线提取成本纳入训练和推理计时，不声称逐位重现旧训练结果。",
        "详见metrics.csv、summary.csv、complexity.csv，以及各模型目录的训练曲线和混淆矩阵。",
    ]
    (destination / "REPORT.md").write_text("\n\n".join(lines), encoding="utf-8")
