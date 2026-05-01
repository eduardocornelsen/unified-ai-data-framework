# Databricks Feature Engineering helpers — import via: %run ./utils/fe_helpers
import subprocess, sys, importlib
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "typing_extensions>=4.6.0", "seaborn", "scipy", "scikit-learn"],
    stdout=subprocess.DEVNULL,
)
import typing_extensions; importlib.reload(typing_extensions)

import math
import json
import re

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType, StringType
from pyspark.sql.window import Window

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick


# ── Plot defaults ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
PALETTE = "coolwarm"


# ── Validation helpers ───────────────────────────────────────────────────────
def assert_columns(df: DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")


def assert_row_range(df: DataFrame, min_rows: int, max_rows: int) -> None:
    n = df.count()
    if not (min_rows <= n <= max_rows):
        raise ValueError(f"Row count {n:,} outside expected range [{min_rows:,}, {max_rows:,}]")
    print(f"Row count check passed: {n:,}")


# ── Temporal helpers ─────────────────────────────────────────────────────────
def add_cyclical_encoding(df: DataFrame, col: str, period: int) -> DataFrame:
    """Add sin/cos cyclical encoding for a periodic integer column."""
    rad = 2 * math.pi / period
    return (
        df
        .withColumn(f"{col}_sin", F.sin(F.col(col) * rad))
        .withColumn(f"{col}_cos", F.cos(F.col(col) * rad))
    )


# ── Numeric helpers ──────────────────────────────────────────────────────────
def add_age_group(df: DataFrame, col: str = "Age") -> DataFrame:
    return df.withColumn(
        "age_group_ord",
        F.when(F.col(col) < 25, 0)
         .when(F.col(col) < 35, 1)
         .when(F.col(col) < 45, 2)
         .when(F.col(col) < 55, 3)
         .otherwise(4)
        .cast("int"),
    )


def compute_iqr_bounds(df: DataFrame, col: str) -> dict:
    """Return IQR outlier bounds computed on the given DataFrame (use training set only)."""
    q1, q3 = df.approxQuantile(col, [0.25, 0.75], 0.01)
    iqr = q3 - q1
    return {"lower": q1 - 1.5 * iqr, "upper": q3 + 1.5 * iqr}


def add_outlier_flag(df: DataFrame, col: str, bounds: dict) -> DataFrame:
    safe = col.replace(" ", "_").lower()
    return df.withColumn(
        f"is_outlier_{safe}",
        ((F.col(col) < bounds["lower"]) | (F.col(col) > bounds["upper"])).cast("int"),
    )


# ── Categorical / target-encoding helpers ────────────────────────────────────
def compute_target_encoding_map(
    df_train: DataFrame, col: str, target: str, alpha: float = 10.0
) -> dict:
    """
    Compute smoothed target-encoding map from training data.
    Formula: (click_count + α) / (total_count + 2α)
    """
    global_rate = df_train.agg(F.mean(target)).collect()[0][0]
    agg = (
        df_train.groupBy(col)
        .agg(
            F.sum(target).cast("double").alias("click_sum"),
            F.count("*").cast("double").alias("total"),
        )
        .toPandas()
    )
    agg["encoded"] = (agg["click_sum"] + alpha * global_rate) / (agg["total"] + alpha)
    return dict(zip(agg[col].astype(str), agg["encoded"].round(6)))


def apply_target_encoding(df: DataFrame, col: str, mapping: dict,
                           global_fallback: float = 0.5) -> DataFrame:
    """Apply a pre-computed target-encoding map; unseen values get global_fallback."""
    mapping_expr = F.create_map(
        *[item for pair in [(F.lit(k), F.lit(v)) for k, v in mapping.items()] for item in pair]
    )
    safe = col.replace(" ", "_").lower()
    return df.withColumn(
        f"{safe}_te",
        F.coalesce(mapping_expr[F.col(col).cast("string")], F.lit(global_fallback)),
    )


# ── Text helpers ─────────────────────────────────────────────────────────────
KEYWORD_PATTERNS = {
    "topic_has_tech":    r"\b(tech|digital|software|data|cloud|cyber|network|web|app)\b",
    "topic_has_finance": r"\b(financ|invest|bank|money|capital|fund|market|crypto|stock)\b",
    "topic_has_health":  r"\b(health|medical|wellness|pharma|care|fit|diet|nutrition)\b",
}


def add_keyword_flags(df: DataFrame) -> DataFrame:
    for feat, pattern in KEYWORD_PATTERNS.items():
        df = df.withColumn(
            feat,
            (F.lower(F.col("Ad Topic Line")).rlike(pattern)).cast("int"),
        )
    return df


# ── Split balance report ─────────────────────────────────────────────────────
def report_split_balance(splits: dict[str, DataFrame], target: str,
                          threshold: float = 0.05) -> pd.DataFrame:
    rows = []
    for name, sdf in splits.items():
        n = sdf.count()
        rate = sdf.agg(F.mean(target)).collect()[0][0]
        rows.append({"split": name, "n_rows": n, "click_rate": round(rate, 4)})
    pdf = pd.DataFrame(rows)
    overall = pdf["click_rate"].mean()
    pdf["deviation"] = (pdf["click_rate"] - overall).abs().round(4)
    pdf["balanced"] = pdf["deviation"] <= threshold
    return pdf


# ── Feature ranking helpers ──────────────────────────────────────────────────
def pearson_vs_target(pdf: pd.DataFrame, feature_cols: list[str],
                       target: str) -> pd.Series:
    corr = pdf[feature_cols + [target]].corr()[target].drop(target)
    return corr.abs().sort_values(ascending=False)


def near_zero_variance_features(pdf: pd.DataFrame, feature_cols: list[str],
                                  threshold: float = 0.01) -> list[str]:
    return [c for c in feature_cols if pdf[c].std() < threshold]


def plot_feature_importance(corr: pd.Series, title: str = "Feature Correlation with Target",
                             top_n: int = 25, caption: str = "") -> plt.Figure:
    data = corr.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.35)))
    colors = sns.color_palette(PALETTE, len(data))
    ax.barh(data.index[::-1], data.values[::-1], color=colors[::-1])
    ax.set_xlabel("|Pearson r| with target")
    ax.set_title(title)
    if caption:
        fig.text(0.5, -0.02, f"Insight: {caption}", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    return fig


# ── MLflow helpers ────────────────────────────────────────────────────────────
def log_figures_to_mlflow(figures: dict) -> None:
    import mlflow
    for name, fig in figures.items():
        if fig is not None:
            mlflow.log_figure(fig, f"{name}.png")
            plt.close(fig)


def log_encoding_map(mapping: dict, name: str) -> None:
    import mlflow, tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mapping, f, indent=2)
        tmp_path = f.name
    mlflow.log_artifact(tmp_path, artifact_path="encoding_maps")
    os.unlink(tmp_path)
