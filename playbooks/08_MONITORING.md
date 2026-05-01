# Monitoring Playbook

## Role

You are a senior Data Scientist and ML Engineer setting up operational monitoring for the deployed champion model. Your input is the production model from `07_INFERENCING.md` and the evaluation baseline from `06_MODEL_EVALUATION.md`. Your output is a monitoring system that detects data drift, prediction drift, and performance decay — with automated alerting, retraining triggers, and a rollback runbook.

This playbook formalizes and expands the drift monitoring introduced in `07_INFERENCING.md` (Section 6) and the rollback procedure in `07_INFERENCING.md` (Section 9).

---

## Environment

**Compute:** Databricks Serverless or any Python environment with scipy / scikit-learn
**Storage:** Delta Lake — read inference predictions, write monitoring signals
**Experiment Tracking:** MLflow — log monitoring metrics, drift reports, and alert history
**Scheduling:** Databricks Jobs / cron / Airflow — automated monitoring runs
**Alerting:** Email, Slack, PagerDuty, or equivalent — threshold-based notifications

---

## Inputs

| Input | Source | Description |
|---|---|---|
| Production predictions | `inference_output/predictions` | Scored records with probabilities and timestamps |
| Training reference stats | `fe_output/artifacts/train_stats.json` | Feature distributions at training time |
| Evaluation baseline | `model_evaluation.ipynb` | Performance metrics, fairness slices, SHAP baseline |
| Ground truth labels | Label pipeline (delayed) | Actual outcomes — may arrive hours, days, or weeks after prediction |
| Model metadata | MLflow Model Registry | Model version, threshold, feature schema |
| Problem statement | `templates/problem_statement.md` | KPI, SLA, acceptable degradation bounds |

---

## Deliverables

1. **`monitoring_setup.ipynb`** — fully executed notebook that configures all monitoring signals
2. **`utils/monitoring_helpers.py`** — drift computation, alerting, and visualization helpers
3. **`monitoring_runbook.md`** — standalone operational runbook for on-call response

---

## Notebook Structure

### 0. Setup

- `%pip install` in its own cell (scipy, scikit-learn, evidently — optional)
- Import monitoring libraries: scipy.stats, sklearn.metrics, numpy, pandas
- Define constants: `MODEL_NAME`, `MODEL_VERSION`, `MONITORING_WINDOW_DAYS`, `DRIFT_THRESHOLDS`, `PERFORMANCE_SLA`, `ALERT_CHANNELS`, `LABEL_ARRIVAL_LAG_DAYS`
- Start MLflow experiment: `mlflow.set_experiment("<project>_monitoring")`
- Load training reference distributions and evaluation baseline metrics

### 1. Monitoring Signal Definition

Define the three layers of monitoring signals:

| Layer | Signal | Detection Speed | Label Required? |
|---|---|---|---|
| **Feature drift** | Input distribution shift | Immediate | No |
| **Prediction drift** | Output distribution shift | Immediate | No |
| **Performance decay** | Metric degradation | Delayed (needs labels) | Yes |

Each layer serves a different purpose:
- **Feature drift** is the early warning — detects upstream data changes before they affect predictions
- **Prediction drift** catches model behavior changes even when individual features look stable
- **Performance decay** is the ground truth — but arrives late because labels take time

### 2. Label Arrival Tracking

Labels may arrive with significant delay. Track and monitor the label pipeline itself:

```python
def track_label_arrival(predictions_df, labels_df, prediction_ts_col, label_ts_col):
    """Monitor label arrival latency and coverage.

    Args:
        predictions_df: DataFrame with prediction timestamps
        labels_df: DataFrame with label timestamps
        prediction_ts_col: timestamp column in predictions
        label_ts_col: timestamp column in labels

    Returns:
        dict with coverage_rate, median_lag_hours, p95_lag_hours, missing_windows
    """
    import pandas as pd
    import numpy as np

    merged = predictions_df.merge(labels_df, how="left", on="record_id")
    has_label = merged[label_ts_col].notna()

    coverage = has_label.mean()
    lags = (merged.loc[has_label, label_ts_col] - merged.loc[has_label, prediction_ts_col])
    lag_hours = lags.dt.total_seconds() / 3600

    return {
        "coverage_rate": coverage,
        "median_lag_hours": np.median(lag_hours),
        "p95_lag_hours": np.percentile(lag_hours, 95),
        "missing_windows": (~has_label).sum(),
    }
```

**Alert:** If label coverage drops below 80% for the monitoring window, alert the data engineering team — performance monitoring becomes unreliable.

### 3. Feature Drift Detection

Monitor each feature for distribution shift against the training reference.

**Drift statistics per feature:**

| Statistic | Use Case | Alert Threshold | Critical Threshold |
|---|---|---|---|
| KS statistic | Continuous features | > 0.10 | > 0.20 |
| PSI (Population Stability Index) | Continuous and categorical | > 0.10 | > 0.25 |
| Jensen-Shannon divergence | Any distribution | > 0.05 | > 0.10 |

```python
def compute_feature_drift(current_values, reference_values, method="all"):
    """Compute drift statistics for a single feature.

    Args:
        current_values: current window feature values (array-like)
        reference_values: training reference values (array-like)
        method: "ks", "psi", "js", or "all"

    Returns:
        dict with drift statistics and alert levels
    """
    from scipy.stats import ks_2samp
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    results = {}

    if method in ("ks", "all"):
        stat, p_value = ks_2samp(reference_values, current_values)
        results["ks_statistic"] = stat
        results["ks_p_value"] = p_value
        results["ks_alert"] = "critical" if stat > 0.20 else "warn" if stat > 0.10 else "ok"

    if method in ("psi", "all"):
        # PSI computation with binning
        n_bins = 10
        breakpoints = np.percentile(reference_values, np.linspace(0, 100, n_bins + 1))
        breakpoints = np.unique(breakpoints)

        ref_counts = np.histogram(reference_values, bins=breakpoints)[0] / len(reference_values)
        cur_counts = np.histogram(current_values, bins=breakpoints)[0] / len(current_values)

        # Avoid division by zero
        ref_counts = np.clip(ref_counts, 1e-6, None)
        cur_counts = np.clip(cur_counts, 1e-6, None)

        psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))
        results["psi"] = psi
        results["psi_alert"] = "critical" if psi > 0.25 else "warn" if psi > 0.10 else "ok"

    if method in ("js", "all"):
        n_bins = 10
        breakpoints = np.percentile(
            np.concatenate([reference_values, current_values]),
            np.linspace(0, 100, n_bins + 1)
        )
        breakpoints = np.unique(breakpoints)

        ref_hist = np.histogram(reference_values, bins=breakpoints, density=True)[0] + 1e-10
        cur_hist = np.histogram(current_values, bins=breakpoints, density=True)[0] + 1e-10

        js_div = jensenshannon(ref_hist, cur_hist)
        results["js_divergence"] = js_div
        results["js_alert"] = "critical" if js_div > 0.10 else "warn" if js_div > 0.05 else "ok"

    return results


def compute_all_feature_drift(current_df, reference_stats, feature_cols):
    """Compute drift for all features and return a summary DataFrame.

    Args:
        current_df: current monitoring window data
        reference_stats: training reference statistics (dict of arrays or stats)
        feature_cols: list of feature column names

    Returns:
        DataFrame with drift statistics per feature, sorted by severity
    """
    import pandas as pd

    results = []
    for col in feature_cols:
        drift = compute_feature_drift(
            current_df[col].dropna().values,
            reference_stats[col],
        )
        drift["feature"] = col
        results.append(drift)

    df = pd.DataFrame(results)
    df["max_alert"] = df.apply(
        lambda row: "critical" if "critical" in row.values else
                    "warn" if "warn" in row.values else "ok",
        axis=1,
    )
    return df.sort_values("psi", ascending=False)
```

### 4. Prediction Drift Detection

Monitor the output distribution — shifts here indicate the model is behaving differently even if individual features look stable.

```python
def compute_prediction_drift(current_probs, reference_probs):
    """Compare current prediction distribution against training reference.

    Args:
        current_probs: predicted probabilities from current window
        reference_probs: predicted probabilities from training/validation

    Returns:
        dict with distribution statistics and drift alerts
    """
    import numpy as np
    from scipy.stats import ks_2samp

    stat, p = ks_2samp(reference_probs, current_probs)

    return {
        "mean_prob_current": np.mean(current_probs),
        "mean_prob_reference": np.mean(reference_probs),
        "mean_shift": np.mean(current_probs) - np.mean(reference_probs),
        "std_current": np.std(current_probs),
        "std_reference": np.std(reference_probs),
        "ks_statistic": stat,
        "ks_p_value": p,
        "positive_rate_current": (current_probs >= 0.5).mean(),
        "alert": "critical" if stat > 0.15 else "warn" if stat > 0.08 else "ok",
    }
```

Monitor the **positive prediction rate** over time — a sudden spike or drop in the fraction of records predicted positive is a strong signal of upstream data change or model degradation.

### 5. Performance Decay Monitoring

Once labels arrive, compute actual model performance and track degradation over time.

```python
def compute_rolling_performance(df, y_true_col, y_prob_col, date_col,
                                  window_days=7, step_days=1):
    """Compute rolling performance metrics over time windows.

    Args:
        df: DataFrame with true labels, predicted probabilities, and dates
        y_true_col: ground truth column
        y_prob_col: predicted probability column
        date_col: date column
        window_days: rolling window size in days
        step_days: step size between windows

    Returns:
        DataFrame with date, auc, f1, precision, recall, brier per window
    """
    from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
    import pandas as pd
    import numpy as np

    df = df.sort_values(date_col)
    dates = pd.date_range(df[date_col].min(), df[date_col].max(), freq=f"{step_days}D")

    results = []
    for start in dates:
        end = start + pd.Timedelta(days=window_days)
        window = df[(df[date_col] >= start) & (df[date_col] < end)]

        if len(window) < 50 or window[y_true_col].nunique() < 2:
            continue

        y_true = window[y_true_col].values
        y_prob = window[y_prob_col].values

        results.append({
            "window_start": start,
            "window_end": end,
            "n": len(window),
            "auc": roc_auc_score(y_true, y_prob),
            "brier": brier_score_loss(y_true, y_prob),
            "positive_rate": y_true.mean(),
        })

    return pd.DataFrame(results)
```

**Time-to-breach analysis:**

Estimate when performance will cross the SLA threshold if the current decay trend continues:

```python
def estimate_time_to_breach(performance_df, metric_col, sla_threshold, date_col="window_start"):
    """Estimate when a metric will cross the SLA threshold based on trend.

    Args:
        performance_df: rolling performance DataFrame
        metric_col: column to track (e.g., "auc")
        sla_threshold: minimum acceptable value
        date_col: date column

    Returns:
        dict with slope, days_to_breach (None if no breach projected), r_squared
    """
    import numpy as np

    df = performance_df.dropna(subset=[metric_col]).copy()
    if len(df) < 3:
        return {"slope": None, "days_to_breach": None, "r_squared": None}

    x = (df[date_col] - df[date_col].min()).dt.days.values.astype(float)
    y = df[metric_col].values

    # Linear regression
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    y_hat = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Project to breach
    if slope < 0 and intercept > sla_threshold:
        days_to_breach = (sla_threshold - intercept) / slope
    else:
        days_to_breach = None

    return {
        "slope": slope,
        "days_to_breach": days_to_breach,
        "r_squared": r_squared,
        "current_value": y[-1],
        "sla_threshold": sla_threshold,
    }
```

### 6. Retraining Trigger Rules

Define automated rules for when retraining should be initiated:

| Trigger | Condition | Action |
|---|---|---|
| **Performance breach** | AUC drops below SLA threshold for 2 consecutive windows | Trigger retraining pipeline |
| **Feature drift (critical)** | ≥ 3 features show PSI > 0.25 in same window | Alert + schedule retraining |
| **Feature drift (sustained)** | Any feature shows PSI > 0.10 for 4 consecutive windows | Alert + schedule retraining |
| **Prediction drift** | KS statistic > 0.15 on prediction distribution | Alert + investigate |
| **Label distribution shift** | Positive rate changes by > 30% vs. training | Alert + investigate upstream |
| **Scheduled** | Calendar-based (e.g., monthly or quarterly) | Scheduled retraining |

```python
def check_retraining_triggers(drift_summary, performance_summary, config):
    """Evaluate all retraining triggers and return action recommendations.

    Args:
        drift_summary: DataFrame from compute_all_feature_drift
        performance_summary: DataFrame from compute_rolling_performance
        config: dict with thresholds (perf_sla, drift_critical_count, etc.)

    Returns:
        list of triggered rules with severity and recommended action
    """
    triggers = []

    # Performance breach
    if len(performance_summary) >= 2:
        recent = performance_summary.tail(2)
        if (recent["auc"] < config["perf_sla"]).all():
            triggers.append({
                "rule": "performance_breach",
                "severity": "critical",
                "action": "retrain",
                "detail": f"AUC below {config['perf_sla']} for 2 consecutive windows",
            })

    # Feature drift critical
    critical_features = drift_summary[drift_summary["max_alert"] == "critical"]
    if len(critical_features) >= config.get("drift_critical_count", 3):
        triggers.append({
            "rule": "feature_drift_critical",
            "severity": "critical",
            "action": "retrain",
            "detail": f"{len(critical_features)} features in critical drift",
        })

    # Prediction drift
    if drift_summary.get("prediction_ks", 0) > 0.15:
        triggers.append({
            "rule": "prediction_drift",
            "severity": "warning",
            "action": "investigate",
            "detail": "Prediction distribution has shifted significantly",
        })

    return triggers
```

### 7. Alerting & Escalation

Define the alerting matrix:

| Severity | Response Time | Channel | Escalation |
|---|---|---|---|
| **Info** | Next business day | Dashboard + email digest | None |
| **Warning** | Within 4 hours | Email + Slack | Data Scientist on-call |
| **Critical** | Within 1 hour | Slack + PagerDuty | DS Lead + ML Engineer |

```python
def send_alert(severity, message, channels, context=None):
    """Send monitoring alert through configured channels.

    Args:
        severity: "info", "warning", or "critical"
        message: alert message
        channels: list of channel types ("email", "slack", "pagerduty")
        context: dict with additional context (metrics, timestamps, etc.)

    Returns:
        dict with send status per channel
    """
    alert_payload = {
        "severity": severity,
        "message": message,
        "timestamp": pd.Timestamp.now().isoformat(),
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "context": context or {},
    }

    # Implementation depends on infrastructure
    # Log to MLflow as a record regardless of channel
    import mlflow
    mlflow.log_dict(alert_payload, f"alerts/{severity}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")

    return alert_payload
```

### 8. Monitoring Job Definition

Define the scheduled monitoring job:

```yaml
# monitoring_job.yaml
name: model_monitoring_daily
schedule:
  cron: "0 6 * * *"  # Daily at 6 AM UTC
tasks:
  - name: feature_drift_check
    notebook: monitoring_setup.ipynb
    parameters:
      mode: drift_only
      window_days: 7
    alerts:
      on_failure: critical
  - name: performance_check
    notebook: monitoring_setup.ipynb
    parameters:
      mode: performance
      window_days: 14
    alerts:
      on_failure: critical
    depends_on: feature_drift_check
```

**Monitoring cadence recommendations:**

| Signal | Cadence | Window Size |
|---|---|---|
| Feature drift | Daily | 7-day rolling |
| Prediction drift | Daily | 7-day rolling |
| Performance decay | Weekly (or as labels arrive) | 14-day rolling |
| Label coverage | Daily | 1-day |
| Full evaluation re-run | Monthly | Full dataset |

### 9. Rollback & Recovery Procedure

When monitoring triggers a critical alert and the model must be rolled back:

**Step 1: Confirm the issue**
- Is the alert a true positive? Check the raw data and metrics manually.
- Is the issue in the model or the data pipeline?

**Step 2: Roll back the model**
```python
def rollback_model(model_name, target_version=None):
    """Roll back to the previous production model version.

    Args:
        model_name: registered model name in MLflow
        target_version: specific version to roll back to (default: previous)

    Returns:
        dict with old_version, new_version, and rollback status
    """
    import mlflow
    client = mlflow.tracking.MlflowClient()

    # Find current production version
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        return {"status": "error", "message": "No production model found"}

    current = prod_versions[0]

    if target_version is None:
        # Find previous version
        all_versions = client.search_model_versions(f"name='{model_name}'")
        previous = sorted(
            [v for v in all_versions if int(v.version) < int(current.version)],
            key=lambda v: int(v.version),
            reverse=True,
        )
        if not previous:
            return {"status": "error", "message": "No previous version to roll back to"}
        target_version = previous[0].version

    # Transition
    client.transition_model_version_stage(model_name, target_version, "Production")
    client.transition_model_version_stage(model_name, current.version, "Archived")

    return {
        "status": "success",
        "old_version": current.version,
        "new_version": target_version,
    }
```

**Step 3: Validate the rollback**
- Run inference with the rolled-back model on the current data
- Verify predictions are reasonable (no null outputs, distribution looks like training)
- Confirm downstream systems are consuming the new predictions

**Step 4: Root cause analysis**
- Was it data drift? → Fix upstream pipeline, retrain on refreshed data
- Was it a model bug? → Return to `05_MODEL_TRAINING.md` with the new failure mode documented
- Was it a label shift? → Update the problem framing (`00_PROBLEM_FRAMING.md`)

**Step 5: Document in the monitoring log**
- Record the incident: timestamp, trigger, root cause, resolution, time-to-resolve

### 10. Monitoring Dashboard Specification

Define the dashboard panels for the monitoring system:

| Panel | Visualization | Refresh |
|---|---|---|
| Model health summary | Traffic light (green/yellow/red) per signal layer | Real-time |
| Feature drift heatmap | Features × time windows, colored by PSI | Daily |
| Prediction distribution | Histogram overlay (current vs. reference) | Daily |
| Performance trend | Line chart with SLA threshold line | Weekly |
| Label coverage | Bar chart showing coverage by day | Daily |
| Alert history | Timeline of alerts with severity | Real-time |
| Retraining trigger status | Table of trigger rules with current state | Daily |

### 11. Final MLflow Logging

- Log monitoring configuration: thresholds, cadence, alert channels
- Log baseline reference distributions as artifacts
- Log the monitoring job definition (YAML)
- Tag the MLflow run with `monitoring_status = active`
- Log the rollback runbook as an artifact

---

## Code Standards

- Every section has a Markdown cell explaining what is being done and why
- Constants at top of notebook: `MODEL_NAME`, `MODEL_VERSION`, `MONITORING_WINDOW_DAYS`, `DRIFT_THRESHOLDS`, `PERFORMANCE_SLA`
- All monitoring plots use `plt.savefig()` and log to MLflow
- Functions > 15 lines go in `utils/monitoring_helpers.py`
- Drift computations must handle missing values gracefully (drop NaN before computing statistics)
- All timestamps in UTC

---

## Constraints & Guardrails

- Do **NOT** set drift thresholds without context — calibrate them against observed natural variation during the evaluation period
- Do **NOT** alert on every drift signal — use the severity matrix to avoid alert fatigue
- Do **NOT** skip label arrival tracking — if labels stop arriving, all performance metrics become stale without warning
- Do **NOT** retrain automatically without human review — triggers should escalate to a data scientist, not directly launch retraining
- Do **NOT** discard monitoring history — keep all drift reports and alerts in MLflow for trend analysis
- Do **NOT** roll back without validating the previous version first — the previous model may have its own issues

---

## Acceptance Criteria

| Criterion | Definition of Done |
|---|---|
| Feature drift | KS, PSI, and JS divergence computed for all features against training reference |
| Prediction drift | Output distribution monitored and compared to reference |
| Performance decay | Rolling performance with time-to-breach estimation configured |
| Label tracking | Label arrival latency and coverage monitored |
| Retraining triggers | Rules table defined with thresholds, conditions, and actions |
| Alerting | Severity matrix defined with channels and response times |
| Monitoring job | Scheduled job definition (YAML/cron) documented |
| Rollback runbook | Step-by-step rollback procedure documented and tested |
| Dashboard spec | All monitoring panels defined with refresh cadence |
| MLflow logging | All monitoring config, baselines, and job definitions logged |

---

## Downstream Handoffs

| Recipient | What they receive | Playbook |
|---|---|---|
| Data Scientist | Drift alerts, performance decay trends, retraining trigger notifications | `05_MODEL_TRAINING.md` (retraining) |
| ML Engineer | Rollback runbook, monitoring job definition, alert configuration | Operations |
| Stakeholder Communicator | Model health status, drift summary, any performance incidents | `10_STAKEHOLDER_COMMUNICATION.md` |
| DS Reviewer | Monitoring setup for quality review | Review gate |
