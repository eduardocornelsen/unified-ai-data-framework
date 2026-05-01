# Databricks EDA helper functions — import via: %run ./utils/eda_helpers
import subprocess, sys, importlib
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "typing_extensions>=4.6.0", "seaborn", "scipy"],
    stdout=subprocess.DEVNULL,
)
import typing_extensions; importlib.reload(typing_extensions)

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import TimestampType, NumericType, StringType
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np


# ── Plotting defaults ────────────────────────────────────────────────────────
PALETTE = "coolwarm"
FIG_SIZE = (10, 5)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ── Spark helpers ────────────────────────────────────────────────────────────
def null_report(df: DataFrame) -> pd.DataFrame:
    """Return a pandas DataFrame with null count and % per column."""
    total = df.count()
    rows = []
    for c in df.columns:
        n = df.filter(F.col(c).isNull()).count()
        rows.append({"column": c, "null_count": n, "null_pct": round(n / total * 100, 2)})
    return pd.DataFrame(rows).sort_values("null_pct", ascending=False)


def duplicate_report(df: DataFrame) -> dict:
    """Return total rows, distinct rows, and duplicate count."""
    total = df.count()
    distinct = df.distinct().count()
    return {"total": total, "distinct": distinct, "duplicates": total - distinct}


def numeric_cols(df: DataFrame) -> list[str]:
    return [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]


def categorical_cols(df: DataFrame) -> list[str]:
    return [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]


def click_rate_by(df: DataFrame, col: str, target: str = "Clicked on Ad") -> pd.DataFrame:
    """Compute click rate (mean of target) grouped by a single column."""
    return (
        df.groupBy(col)
        .agg(
            F.mean(target).alias("click_rate"),
            F.count("*").alias("count"),
        )
        .orderBy("click_rate", ascending=False)
        .toPandas()
    )


def outlier_bounds(df: DataFrame, col: str) -> dict:
    """Return IQR-based lower/upper outlier bounds for a numeric column."""
    q1, q3 = df.approxQuantile(col, [0.25, 0.75], 0.01)
    iqr = q3 - q1
    return {"lower": q1 - 1.5 * iqr, "upper": q3 + 1.5 * iqr, "q1": q1, "q3": q3, "iqr": iqr}


# ── Plot helpers ─────────────────────────────────────────────────────────────
def plot_click_rate_bar(pdf: pd.DataFrame, col: str, caption: str = "", top_n: int = 20):
    pdf = pdf.head(top_n)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.barh(pdf[col].astype(str), pdf["click_rate"], color=sns.color_palette(PALETTE, len(pdf)))
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_xlabel("Click Rate")
    ax.set_title(f"Click Rate by {col}")
    ax.invert_yaxis()
    if caption:
        fig.text(0.5, -0.02, f"Insight: {caption}", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    return fig


def plot_distribution(pdf: pd.DataFrame, col: str, target: str = "Clicked on Ad", caption: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Histogram
    for val, label in [(0, "No Click"), (1, "Click")]:
        axes[0].hist(pdf[pdf[target] == val][col].dropna(), bins=30, alpha=0.6, label=label)
    axes[0].set_title(f"{col} — Distribution by Target")
    axes[0].set_xlabel(col)
    axes[0].legend()
    # Boxplot
    sns.boxplot(data=pdf, x=target, y=col, ax=axes[1], palette=["#5577AA", "#EE6644"])
    axes[1].set_xticklabels(["No Click", "Click"])
    axes[1].set_title(f"{col} — Boxplot by Target")
    if caption:
        fig.text(0.5, -0.02, f"Insight: {caption}", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    return fig


def plot_heatmap(pdf: pd.DataFrame, row: str, col: str, val: str, title: str = "", caption: str = ""):
    pivot = pdf.pivot(index=row, columns=col, values=val)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=PALETTE, ax=ax, linewidths=0.3)
    ax.set_title(title or f"{val} by {row} × {col}")
    if caption:
        fig.text(0.5, -0.02, f"Insight: {caption}", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    return fig


def plot_null_bar(null_pdf: pd.DataFrame, caption: str = ""):
    non_zero = null_pdf[null_pdf["null_pct"] > 0]
    if non_zero.empty:
        print("No missing values found in any column.")
        return None
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.barh(non_zero["column"], non_zero["null_pct"], color="#EE6644")
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Values by Column")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    if caption:
        fig.text(0.5, -0.02, f"Insight: {caption}", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    return fig


def log_figures_to_mlflow(figures: dict):
    """Log a dict of {name: matplotlib_figure} to the active MLflow run."""
    import mlflow
    for name, fig in figures.items():
        if fig is not None:
            mlflow.log_figure(fig, f"{name}.png")
            plt.close(fig)
