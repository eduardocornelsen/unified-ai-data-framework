# Inferencing Playbook

## Role

You are a senior Data Scientist operationalizing the champion model. Your input is raw data in the same schema as the original training dataset. Your output is a fully executed notebook (`inferencing.ipynb`) that loads the registered production model, applies the full feature engineering pipeline, scores a batch of records, writes predictions, performs data drift monitoring, and includes a rollback runbook.

---

## Environment

**Compute:** Databricks Serverless or any Python environment with PySpark / pandas
**Storage:** Delta Lake — read new data, write predictions
**Model source:** MLflow Model Registry — champion model at Production stage
**Artifact source:** `fe_output/artifacts/` — fitted pipelines and encoding maps
**Output:** `inference_output/predictions` (Delta or Parquet)

---

## Inputs

| Input | Source | Description |
|---|---|---|
| New data | Data platform / files | Raw data in the same schema as training data |
| Champion model | MLflow Model Registry | Registered model at Production stage |
| FE artifacts | `fe_output/artifacts/` | Fitted scalers, encoders, encoding maps, pipelines |
| Training reference stats | `fe_output/artifacts/train_stats.json` | Baseline distributions for drift monitoring |
| Optimal threshold | MLflow model tags | Cost-optimal prediction threshold from `05_MODEL_TRAINING.md` |

---

## Inference Flow

```
Raw data (CSV / Delta / Parquet)
        |
        v
 [1] Schema validation & null guard
        |
        v
 [2] Apply feature engineering pipeline (exact same order as training)
        |
        v
 [3] Load champion model -> score -> attach probability + prediction
        |
        v
 [4] Post-process & enrich output
        |
        v
 [5] Write predictions to Delta
        |
        v
 [6] Data drift monitoring
        |
        v
 [7] (Optional) Shadow mode comparison
        |
        v
 [8] SHAP explainability snapshot
        |
        v
 [9] MLflow logging
```

---

## Deliverables

1. **`inferencing.ipynb`** — fully executed notebook
2. **`inference_output/predictions`** — scored records with probabilities and predictions
3. **`inference_output/drift_report`** — per-feature drift statistics
4. **`inference_output/shap_snapshot`** — per-row SHAP values for a sample (optional but recommended)

---

## Notebook Structure

### 0. Setup

- `%pip install` in its own cell (shap, scipy)
- Imports: PySpark, MLflow, pandas, scipy (KS test for drift), shap
- Constants: `INFERENCE_INPUT_PATH`, `FE_ARTIFACTS_PATH`, `MODEL_NAME`, `MODEL_STAGE`, `OUTPUT_PATH`, `PREDICTION_THRESHOLD`, `DRIFT_KS_ALPHA`, `SHAP_SAMPLE_SIZE`
- Load optimal threshold from MLflow model tags (fall back to `PREDICTION_THRESHOLD` constant)

### 1. Load & Validate Inference Data

- Load raw data from `INFERENCE_INPUT_PATH` via `spark.read` or `pd.read_csv()`
- Assert the same columns as training (reuse schema validation from `utils/contract_helpers.py` or `utils/fe_helpers.py`)
- Check for unexpected nulls — log a warning (do **NOT** silently drop rows)
- Report row count and a sample
- Cross-reference against the data contract (if available)

### 2. Apply Feature Engineering Pipeline

Apply **every transformation in the exact same order** as `feature_engineering.ipynb`:

1. **Temporal features:** Cast timestamps, extract hour/day_of_week/month, cyclical encoding, drop raw columns
2. **Numeric features:** Interaction terms, log transforms, bins; load `outlier_bounds.json` and apply outlier flags (use saved bounds — do **NOT** recompute)
3. **Categorical encoding:** Load fitted encoders from `fe_output/artifacts/`; apply; drop raw columns
4. **Text features:** Load fitted TF-IDF pipeline; apply keyword flags; drop raw text columns
5. **Final assembly:** Load fitted scaler; assemble feature vector

**Critical:** Every encoder, scaler, and encoding map is loaded from disk — **never refit on inference data**.

For unseen categories: handle gracefully with fallback values (OHE `handleInvalid="keep"`, target encoding smoothed global mean).

### 3. Load & Apply Champion Model

```python
import mlflow

# Load registered model
model = mlflow.spark.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
# Or for sklearn: model = mlflow.sklearn.load_model(...)

# Score
predictions = model.transform(df_features)
```

- Extract prediction probability (for classification):
  ```python
  prob_col = F.udf(lambda v: float(v[1]))(F.col("probability"))
  ```
- Apply cost-optimal threshold:
  ```python
  predicted_label = (prob_col >= PREDICTION_THRESHOLD).cast("int")
  ```
- Attach `inference_timestamp`: `F.current_timestamp()`
- Display: prediction distribution, mean predicted probability, top-probability records

### 4. Post-Process & Enrich Output

- Select output columns:
  - All original raw input columns (for business context and traceability)
  - `prediction_probability`, `predicted_label`, `inference_timestamp`
  - `model_version` tag (MLflow run ID or registry version)
- Compute and display: predicted positive rate, confidence distribution histogram

### 5. Write Predictions

- Write to Delta (or Parquet):
  ```python
  predictions_out.write.format("delta").mode("append").save(f"{OUTPUT_PATH}/predictions")
  ```
- Use `.mode("append")` — predictions accumulate over time
- Print row count written and sample output rows

### 6. Data Drift Monitoring

Lightweight statistical drift check comparing inference data distributions against training reference stats.

```python
def monitor_drift(df_inference, train_stats, numeric_cols, categorical_cols, alpha=0.05):
    """Compare inference distributions against training baseline.

    Args:
        df_inference: inference DataFrame (post-FE)
        train_stats: dict from fe_output/artifacts/train_stats.json
        numeric_cols: list of numeric feature names
        categorical_cols: list of categorical feature names
        alpha: significance level for KS test

    Returns:
        List of drift alerts with KS statistic, p-value, and status
    """
    from scipy import stats
    alerts = []

    for col in numeric_cols:
        inference_vals = df_inference.select(col).toPandas()[col].dropna().values
        baseline_mean = train_stats[col]["mean"]
        baseline_std = train_stats[col]["std"]

        # KS test against training sample (if available) or z-score against stats
        if "sample" in train_stats[col]:
            ks_stat, p_val = stats.ks_2samp(train_stats[col]["sample"], inference_vals)
        else:
            # Approximate: check if mean has shifted significantly
            inf_mean = inference_vals.mean()
            z = abs(inf_mean - baseline_mean) / (baseline_std / len(inference_vals)**0.5) if baseline_std > 0 else 0
            ks_stat, p_val = z, 2 * (1 - stats.norm.cdf(z))

        alerts.append({
            "column": col, "type": "numeric",
            "statistic": round(ks_stat, 4), "p_value": round(p_val, 6),
            "drift_detected": p_val < alpha,
            "baseline_mean": round(baseline_mean, 4),
            "inference_mean": round(inference_vals.mean(), 4),
        })

    for col in categorical_cols:
        if col in train_stats and "value_counts" in train_stats[col]:
            inference_vc = df_inference.groupBy(col).count().toPandas().set_index(col)["count"].to_dict()
            new_categories = set(inference_vc.keys()) - set(train_stats[col]["value_counts"].keys())
            if new_categories:
                alerts.append({
                    "column": col, "type": "categorical",
                    "statistic": len(new_categories), "p_value": None,
                    "drift_detected": True,
                    "detail": f"New categories: {new_categories}",
                })

    return alerts
```

- Produce a drift summary table and bar chart of drift statistics
- Write drift report to `inference_output/drift_report` as Delta or CSV
- **Decision rules for drift alerts:**
  - 0 alerts: all clear
  - 1-3 alerts: investigate but continue scoring
  - 4+ alerts or drift in the target distribution: **pause and investigate** — consider rollback

### 7. Shadow Mode (Optional)

Shadow mode runs a new model candidate alongside the production model without serving its predictions to end users. Use this when:
- Deploying a new model version and wanting to compare before promoting
- Testing a retrained model after drift was detected

```python
def shadow_comparison(df, production_model, shadow_model, label_col=None):
    """Score with both models and compare predictions.

    If labels are available, compare accuracy. Otherwise compare prediction agreement.
    """
    prod_preds = production_model.transform(df).select("prediction").withColumnRenamed("prediction", "prod_pred")
    shadow_preds = shadow_model.transform(df).select("prediction").withColumnRenamed("prediction", "shadow_pred")

    comparison = prod_preds.join(shadow_preds)
    agreement_rate = comparison.filter(F.col("prod_pred") == F.col("shadow_pred")).count() / comparison.count()

    print(f"Production vs Shadow agreement rate: {agreement_rate:.4f}")

    if label_col:
        # Compare accuracy if labels are available
        pass

    return agreement_rate
```

- Log shadow comparison results to MLflow
- If shadow model outperforms production on a holdout: flag for model swap consideration

### 8. SHAP Explainability Snapshot

Per-row feature attributions help debug unexpected predictions and build stakeholder trust.

```python
def compute_shap_snapshot(model, df_sample, feature_names, n_samples=100):
    """Compute SHAP values for a sample of predictions.

    Args:
        model: fitted model (sklearn-compatible or tree-based)
        df_sample: pandas DataFrame with features
        feature_names: list of feature names
        n_samples: number of rows to explain

    Returns:
        shap_values DataFrame, summary plot
    """
    import shap

    sample = df_sample.head(n_samples)

    # Use TreeExplainer for tree models, KernelExplainer as fallback
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(sample, 50))

    shap_values = explainer.shap_values(sample)

    # Summary plot
    fig = shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)

    return shap_values, fig
```

- Compute SHAP values for a sample of predictions (default: 100 rows)
- Save SHAP values to `inference_output/shap_snapshot` as Parquet
- Log the SHAP summary plot to MLflow
- Flag any prediction where a single feature has outsized SHAP contribution (> 50% of total)

### 9. MLflow Logging

Log to a new run under the inferencing experiment:
- **Params:** `model_name`, `model_stage`, `model_version`, `n_input_rows`, `prediction_threshold`, `drift_alpha`
- **Metrics:** `n_predictions`, `predicted_positive_rate`, `n_drift_alerts`, mean and std of prediction probability
- **Artifacts:** drift report CSV, prediction distribution chart, SHAP summary plot

---

## Rollback Runbook

If drift monitoring or shadow mode reveals a problem, follow this runbook:

### When to Rollback

| Signal | Severity | Action |
|---|---|---|
| 4+ numeric features show significant drift | High | Pause scoring; investigate data pipeline |
| Predicted positive rate deviates > 2x from training rate | High | Pause scoring; check for data quality issues |
| Shadow model significantly outperforms production | Medium | Promote shadow model to production |
| SHAP shows single feature dominating all predictions | Medium | Investigate for data leakage or pipeline error |
| Labels arrive and test-set metrics drop > 10% | Critical | Rollback to previous model version |

### How to Rollback

1. **Identify the previous stable model version** in the MLflow Model Registry
2. **Transition the current Production model** to "Archived" stage
3. **Transition the previous version** back to "Production" stage:
   ```python
   from mlflow.tracking import MlflowClient
   client = MlflowClient()
   client.transition_model_version_stage(MODEL_NAME, version=prev_version, stage="Production")
   client.transition_model_version_stage(MODEL_NAME, version=current_version, stage="Archived")
   ```
4. **Re-run inference** with the rolled-back model
5. **Document the incident** — what triggered rollback, root cause, resolution plan

---

## Code Standards

- Every section has a Markdown cell explaining what and why
- Encoders loaded from disk — never refit on inference data
- `display()` for interactive previews; matplotlib for saved charts
- Constants defined at top; no magic numbers
- `%pip install` in its own first cell
- Functions > 15 lines go in `utils/` helpers

---

## Constraints & Guardrails

- **NEVER** refit or retrain any encoder on inference data — load from `fe_output/artifacts/` only
- **NEVER** drop rows silently — flag or log unexpected nulls and unseen categories
- **NEVER** use the default 0.5 threshold without checking the cost-optimal threshold from training
- Write mode for predictions must be **`append`** — do not overwrite historical predictions
- Drift monitoring is **not optional** — it runs every time the inference notebook runs
- SHAP snapshot is recommended for every batch; mandatory for the first inference run

---

## Acceptance Criteria

| Criterion | Definition of Done |
|---|---|
| Pipeline integrity | All FE steps applied in exact training order using saved artifacts |
| Null safety | Null guard runs before any transformation; no silent row drops |
| Threshold applied | Cost-optimal threshold from training used (not default 0.5) |
| Predictions written | Output table with probability + prediction + timestamp + model version |
| Drift report | KS test (or equivalent) run on all numeric features; report written |
| Rollback documented | Rollback runbook present with signals, severity, and procedure |
| SHAP snapshot | SHAP values computed for a sample; summary plot logged |
| No leakage | Zero encoder refitting on inference data |
| MLflow logged | Run with params, metrics, and >= 3 artifacts logged |
| Reproducibility | Notebook runs clean top-to-bottom |

---

## Downstream Handoffs

| Recipient | What they receive | Playbook |
|---|---|---|
| Business stakeholders | Predictions table for decision-making | — |
| ML Engineer | Drift alerts for monitoring and retraining triggers | `08_MONITORING.md` |
| Data Scientist | SHAP values for error analysis and model improvement | `06_MODEL_EVALUATION.md` |
| Data Scientist Reviewer | Drift report, SHAP snapshot for quality gate | Review gate |
