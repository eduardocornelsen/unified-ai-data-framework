# Databricks Model Training & Inferencing helpers
# Import via: %run ./utils/model_helpers
import json

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    confusion_matrix, roc_auc_score,
)

# ── Plot defaults ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
MODEL_COLORS = {
    "LogisticRegression": "#5577AA",
    "RandomForest":       "#44AA77",
    "GBT":                "#EE6644",
    "GBT_tuned":          "#AA44BB",
}


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_binary(predictions: DataFrame, label_col: str = "label") -> dict:
    """Return dict of AUC-ROC, AUC-PR, F1, Precision, Recall, Accuracy."""
    bin_eval = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction"
    )
    mc_eval = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction"
    )
    return {
        "auc_roc":   round(bin_eval.evaluate(predictions, {bin_eval.metricName: "areaUnderROC"}), 4),
        "auc_pr":    round(bin_eval.evaluate(predictions, {bin_eval.metricName: "areaUnderPR"}),  4),
        "f1":        round(mc_eval.evaluate(predictions,  {mc_eval.metricName: "f1"}),            4),
        "precision": round(mc_eval.evaluate(predictions,  {mc_eval.metricName: "weightedPrecision"}), 4),
        "recall":    round(mc_eval.evaluate(predictions,  {mc_eval.metricName: "weightedRecall"}),    4),
        "accuracy":  round(mc_eval.evaluate(predictions,  {mc_eval.metricName: "accuracy"}),          4),
    }


def predictions_to_sklearn(predictions: DataFrame,
                             label_col: str = "label") -> tuple[np.ndarray, np.ndarray]:
    """Extract (y_true, y_prob) as numpy arrays for sklearn plotting utilities."""
    pdf = predictions.select(
        label_col,
        F.udf(lambda v: float(v[1]), "double")(F.col("probability")).alias("prob"),
    ).toPandas()
    return pdf[label_col].values, pdf["prob"].values


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_roc_curves(curves: dict[str, tuple], caption: str = "") -> plt.Figure:
    """
    curves: {model_name: (y_true, y_prob)}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    for name, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})",
                color=MODEL_COLORS.get(name, None), lw=2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Candidates")
    ax.legend(loc="lower right", fontsize=9)
    if caption:
        fig.text(0.5, -0.02, f"Insight: {caption}", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    return fig


def plot_pr_curves(curves: dict[str, tuple], caption: str = "") -> plt.Figure:
    """curves: {model_name: (y_true, y_prob)}"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_prob) in curves.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(rec, prec, label=name, color=MODEL_COLORS.get(name, None), lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Candidates")
    ax.legend(fontsize=9)
    if caption:
        fig.text(0.5, -0.02, f"Insight: {caption}", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str = "", caption: str = "") -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = np.array([[f"{v}\n({p:.1%})" for v, p in zip(row_v, row_p)]
                        for row_v, row_p in zip(cm, cm_pct)])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=ax,
                xticklabels=["No Click", "Click"],
                yticklabels=["No Click", "Click"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix{' — ' + model_name if model_name else ''}")
    if caption:
        fig.text(0.5, -0.03, f"Insight: {caption}", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    return fig


def plot_metric_comparison(metrics_table: pd.DataFrame,
                            metrics: list[str] = None,
                            caption: str = "") -> plt.Figure:
    if metrics is None:
        metrics = ["auc_roc", "auc_pr", "f1"]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    for ax, metric in zip(axes, metrics):
        colors = [MODEL_COLORS.get(m, "steelblue") for m in metrics_table["model"]]
        bars = ax.bar(metrics_table["model"], metrics_table[metric], color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_title(metric.upper().replace("_", "-"))
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(metrics_table["model"], rotation=20, ha="right", fontsize=9)
    fig.suptitle("Model Comparison — Validation Set", fontsize=13)
    if caption:
        fig.text(0.5, -0.02, f"Insight: {caption}", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    return fig


def plot_feature_importance(importances: pd.Series, model_name: str = "",
                             top_n: int = 20, caption: str = "") -> plt.Figure:
    data = importances.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, len(data) * 0.35)))
    ax.barh(data.index[::-1], data.values[::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance{' — ' + model_name if model_name else ''} (Top {top_n})")
    if caption:
        fig.text(0.5, -0.02, f"Insight: {caption}", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    return fig


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray,
                     n_bins: int = 10, model_name: str = "") -> plt.Figure:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_mean_pred = [y_prob[bin_idx == i].mean() if (bin_idx == i).any() else np.nan
                     for i in range(n_bins)]
    bin_actual    = [y_true[bin_idx == i].mean()  if (bin_idx == i).any() else np.nan
                     for i in range(n_bins)]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.scatter(bin_mean_pred, bin_actual, s=60, zorder=3, color="#EE6644")
    ax.plot(bin_mean_pred, bin_actual, color="#EE6644", lw=1.5,
            label=model_name or "Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Actual Click Rate")
    ax.set_title("Calibration Plot")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_drift_report(drift_pdf: pd.DataFrame, alpha: float = 0.05) -> plt.Figure:
    drift_pdf = drift_pdf.sort_values("ks_stat", ascending=False)
    colors = ["#EE6644" if d else "steelblue" for d in drift_pdf["drift_flag"]]
    fig, ax = plt.subplots(figsize=(10, max(4, len(drift_pdf) * 0.4)))
    bars = ax.barh(drift_pdf["feature"][::-1], drift_pdf["ks_stat"][::-1], color=colors[::-1])
    ax.axvline(x=0, color="black", lw=0.5)
    ax.set_xlabel("KS Statistic")
    ax.set_title(f"Feature Drift — KS Test (α={alpha})\n"
                 f"Red = drift detected (p < {alpha})")
    plt.tight_layout()
    return fig


# ── MLflow helpers ────────────────────────────────────────────────────────────
def log_metrics_to_mlflow(metrics: dict, prefix: str = "") -> None:
    import mlflow
    for k, v in metrics.items():
        mlflow.log_metric(f"{prefix}{k}" if prefix else k, v)


def log_figures_to_mlflow(figures: dict) -> None:
    import mlflow
    for name, fig in figures.items():
        if fig is not None:
            mlflow.log_figure(fig, f"{name}.png")
            plt.close(fig)
