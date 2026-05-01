# Model Evaluation Playbook

## Role

You are a senior Data Scientist performing a deep evaluation of the champion model before promotion to production. Your input is the registered champion model from `model_training.ipynb` and its test-set predictions. Your output is a fully executed notebook (`model_evaluation.ipynb`) that goes beyond training-time metrics to assess calibration depth, error analysis, fairness, robustness, and SHAP stability — culminating in a formal **Go/No-Go promotion decision**.

This playbook expands on the initial calibration and fairness assessments in `05_MODEL_TRAINING.md` (Sections 8.5–8.6) with deeper analysis that warrants a dedicated evaluation phase.

---

## Environment

**Compute:** Databricks Serverless or any Python environment with scikit-learn / PySpark
**Storage:** Delta Lake — read test predictions, feature-engineered splits, and model artifacts
**ML Framework:** scikit-learn metrics, SHAP, fairlearn (fairness), scipy (bootstrap)
**Experiment Tracking:** MLflow — log evaluation metrics, plots, and promotion decision
**Model Registry:** MLflow Model Registry — update model stage based on decision

---

## Inputs

| Input | Source | Description |
|---|---|---|
| Champion model | MLflow Model Registry | Registered model from `05_MODEL_TRAINING.md` |
| Test predictions | `model_training.ipynb` Section 9 | Predicted probabilities and labels on the held-out test set |
| Test split | `fe_output/test` | Feature-engineered test data with ground truth labels |
| Validation predictions | `model_training.ipynb` | Validation-set predictions for stability comparison |
| Problem statement | `templates/problem_statement.md` | Cost asymmetry, KPI, protected attributes, business constraints |
| Optimal threshold | MLflow model tags | Cost-optimal prediction threshold from training |
| Training reference stats | `fe_output/artifacts/train_stats.json` | Feature distributions for stability checks |

---

## Deliverables

1. **`model_evaluation.ipynb`** — fully executed notebook with all evaluation analyses
2. **`utils/eval_helpers.py`** — evaluation, plotting, and fairness helper functions
3. **Promotion decision** — `PROMOTE` / `CONDITIONAL` / `REJECT` logged to MLflow with rationale

---

## Notebook Structure

### 0. Setup

- `%pip install` in its own cell (shap, fairlearn, scipy, scikit-learn)
- Import evaluation libraries: sklearn.metrics, sklearn.calibration, shap, fairlearn, scipy.stats
- Define constants: `MODEL_NAME`, `EXPERIMENT_NAME`, `THRESHOLD`, `COST_FP`, `COST_FN`, `PROTECTED_ATTRS`, `MIN_AUC`, `MAX_BRIER`, `MAX_ECE`
- Start MLflow experiment: `mlflow.set_experiment("<project>_model_evaluation")`
- Load the champion model from the registry
- Load test predictions and ground truth labels

### 1. Baseline Comparison

Compare the champion model against **at least 4 baselines** to contextualize performance:

| Baseline | Description |
|---|---|
| Naive (majority class) | Predict the most frequent class for every example |
| Random | Predict class proportional to training class distribution |
| Simple heuristic | A single-rule business heuristic (e.g., "flag all users with > 30 days inactive") |
| Interpretable model | Logistic regression or single decision tree from training |

```python
def compare_baselines(y_true, y_pred_champion, baselines, threshold):
    """Compare champion model against multiple baselines.

    Args:
        y_true: ground truth labels
        y_pred_champion: champion model predicted probabilities
        baselines: dict of {name: predicted_probabilities_or_labels}
        threshold: classification threshold

    Returns:
        DataFrame with metrics per model (AUC, F1, precision, recall, cost)
    """
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    import pandas as pd

    results = []
    for name, y_pred in baselines.items():
        y_label = (y_pred >= threshold).astype(int) if y_pred.max() <= 1 else y_pred
        results.append({
            "model": name,
            "auc": roc_auc_score(y_true, y_pred) if y_pred.max() <= 1 else None,
            "f1": f1_score(y_true, y_label),
            "precision": precision_score(y_true, y_label),
            "recall": recall_score(y_true, y_label),
        })

    # Add champion
    y_champ_label = (y_pred_champion >= threshold).astype(int)
    results.append({
        "model": "champion",
        "auc": roc_auc_score(y_true, y_pred_champion),
        "f1": f1_score(y_true, y_champ_label),
        "precision": precision_score(y_true, y_champ_label),
        "recall": recall_score(y_true, y_champ_label),
    })

    return pd.DataFrame(results).sort_values("auc", ascending=False)
```

**Gate:** Champion must beat all baselines on the primary selection metric. If it does not, stop and return to `05_MODEL_TRAINING.md`.

### 2. Calibration Deep-Dive

Calibration tells you whether a predicted probability of 0.3 really means ~30% chance. This goes deeper than the initial reliability diagram in `05_MODEL_TRAINING.md`.

**Metrics:**

| Metric | Formula | Threshold |
|---|---|---|
| Brier score | `mean((p - y)^2)` | Lower is better; compare to naive Brier |
| Expected Calibration Error (ECE) | Weighted average of `|acc(b) - conf(b)|` across bins | < 0.05 preferred |
| Maximum Calibration Error (MCE) | Max bin-level `|acc(b) - conf(b)|` | < 0.15 preferred |

```python
def calibration_analysis(y_true, y_prob, n_bins=10):
    """Compute calibration metrics and produce reliability diagram.

    Args:
        y_true: ground truth binary labels
        y_prob: predicted probabilities
        n_bins: number of bins for reliability diagram

    Returns:
        dict with brier_score, ece, mce, and bin-level details
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    import numpy as np

    brier = brier_score_loss(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # ECE and MCE
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    bin_weights = bin_counts / len(y_prob)
    bin_errors = np.abs(prob_true - prob_pred)
    ece = np.sum(bin_weights[:len(bin_errors)] * bin_errors)
    mce = np.max(bin_errors)

    return {
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "prob_true": prob_true,
        "prob_pred": prob_pred,
    }
```

**Recalibration (if needed):**

If ECE > 0.05 or MCE > 0.15, apply recalibration:
- **Platt scaling** — fit a logistic regression on validation-set probabilities vs. labels
- **Isotonic regression** — non-parametric monotonic fit on validation-set probabilities

Compare pre- and post-recalibration reliability diagrams side by side. If recalibration improves ECE by > 20%, apply it to the champion model and re-register.

### 3. Discrimination Analysis

Evaluate the model's ability to separate positive and negative classes.

**ROC curve with bootstrap confidence intervals:**

```python
def roc_with_ci(y_true, y_prob, n_bootstraps=1000, ci=0.95, seed=42):
    """Compute ROC AUC with bootstrap confidence interval.

    Args:
        y_true: ground truth labels
        y_prob: predicted probabilities
        n_bootstraps: number of bootstrap iterations
        ci: confidence level
        seed: random seed

    Returns:
        dict with auc, ci_lower, ci_upper, and bootstrap distribution
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np

    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    alpha = (1 - ci) / 2
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "ci_lower": np.percentile(aucs, alpha * 100),
        "ci_upper": np.percentile(aucs, (1 - alpha) * 100),
        "bootstrap_aucs": aucs,
    }
```

**Precision-Recall curve** — especially important for imbalanced classes. Report AP (average precision) alongside AUC-ROC.

**Lift and gain charts** — show how much better the model is than random at each decile.

### 4. Error Analysis & Slice Discovery

Identify where the model fails. This is often the most actionable section of the evaluation.

**Step 1: Confusion matrix profiling**

For each cell of the confusion matrix, profile the examples:
- **True Positives** — what features characterize correctly identified positives?
- **False Positives** — what looks like a positive but isn't? Are there systemic patterns?
- **False Negatives** — what did the model miss? Are these near-boundary cases or completely different?
- **True Negatives** — sanity check — are these genuinely negative?

```python
def profile_confusion_cells(df, y_true_col, y_pred_col, feature_cols):
    """Profile feature distributions for each confusion matrix cell.

    Args:
        df: DataFrame with features, true labels, and predicted labels
        y_true_col: column name for ground truth
        y_pred_col: column name for predicted labels
        feature_cols: list of feature column names to profile

    Returns:
        dict of DataFrames, one per confusion cell (TP, FP, FN, TN)
    """
    cells = {
        "TP": df[(df[y_true_col] == 1) & (df[y_pred_col] == 1)],
        "FP": df[(df[y_true_col] == 0) & (df[y_pred_col] == 1)],
        "FN": df[(df[y_true_col] == 1) & (df[y_pred_col] == 0)],
        "TN": df[(df[y_true_col] == 0) & (df[y_pred_col] == 0)],
    }
    profiles = {}
    for cell_name, cell_df in cells.items():
        profiles[cell_name] = cell_df[feature_cols].describe()
    return profiles
```

**Step 2: Slice discovery**

Identify worst-performing subgroups across feature combinations:

```python
def discover_worst_slices(df, y_true_col, y_prob_col, slice_cols, min_size=50, metric="auc"):
    """Find feature slices where model performance degrades most.

    Args:
        df: DataFrame with features, true labels, and predicted probabilities
        y_true_col: ground truth column
        y_prob_col: predicted probability column
        slice_cols: list of categorical feature columns to slice by
        min_size: minimum slice size to consider
        metric: evaluation metric ("auc", "f1", "precision", "recall")

    Returns:
        DataFrame of slices sorted by metric (ascending = worst first)
    """
    from sklearn.metrics import roc_auc_score
    import pandas as pd

    results = []
    for col in slice_cols:
        for val, group in df.groupby(col):
            if len(group) < min_size or group[y_true_col].nunique() < 2:
                continue
            score = roc_auc_score(group[y_true_col], group[y_prob_col])
            results.append({
                "slice": f"{col}={val}",
                "n": len(group),
                "positive_rate": group[y_true_col].mean(),
                metric: score,
            })

    return pd.DataFrame(results).sort_values(metric, ascending=True)
```

**Gate:** If any slice with n > 100 has AUC < 0.60, flag for investigation. Document whether the degradation is acceptable or requires model iteration.

### 5. Fairness Assessment

Evaluate fairness across protected attributes. This goes deeper than the informational slicing in `05_MODEL_TRAINING.md` (Section 8.6) — here it can be a **blocking gate**.

**Metrics per protected group:**

| Metric | Definition | Threshold |
|---|---|---|
| Selection rate | Fraction predicted positive | Four-fifths rule: min/max ratio ≥ 0.80 |
| TPR (equal opportunity) | True positive rate per group | Difference < 0.05 |
| FPR (predictive equality) | False positive rate per group | Difference < 0.05 |
| Equalized odds | Both TPR and FPR parity | Both differences < 0.05 |

```python
def evaluate_fairness(df, y_true_col, y_pred_col, protected_col):
    """Compute fairness metrics across protected attribute groups.

    Args:
        df: DataFrame with true labels, predicted labels, and protected attribute
        y_true_col: ground truth column
        y_pred_col: predicted label column
        protected_col: protected attribute column

    Returns:
        DataFrame with per-group metrics and parity assessments
    """
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    results = []
    for group, gdf in df.groupby(protected_col):
        tn, fp, fn, tp = confusion_matrix(
            gdf[y_true_col], gdf[y_pred_col], labels=[0, 1]
        ).ravel()
        results.append({
            "group": group,
            "n": len(gdf),
            "selection_rate": (tp + fp) / len(gdf),
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        })

    result_df = pd.DataFrame(results)

    # Four-fifths rule
    sr = result_df["selection_rate"]
    result_df.attrs["four_fifths_ratio"] = sr.min() / sr.max() if sr.max() > 0 else 1.0
    result_df.attrs["four_fifths_pass"] = result_df.attrs["four_fifths_ratio"] >= 0.80

    return result_df
```

**Fairness-constrained threshold optimization:**

If the default threshold fails fairness criteria, search for a threshold that satisfies both performance and fairness constraints:

```python
def fairness_constrained_threshold(y_true, y_prob, protected, cost_fp, cost_fn,
                                     min_four_fifths=0.80, thresholds=None):
    """Find threshold that minimizes cost while satisfying fairness constraints.

    Args:
        y_true: ground truth labels
        y_prob: predicted probabilities
        protected: protected attribute values
        cost_fp: cost of a false positive
        cost_fn: cost of a false negative
        min_four_fifths: minimum selection rate ratio
        thresholds: candidate thresholds to search

    Returns:
        dict with optimal threshold, cost, and fairness metrics
    """
    import numpy as np

    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)

    best = {"threshold": 0.5, "cost": float("inf"), "feasible": False}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        cost = fp * cost_fp + fn * cost_fn

        # Check four-fifths rule
        groups = np.unique(protected)
        rates = [y_pred[protected == g].mean() for g in groups]
        ratio = min(rates) / max(rates) if max(rates) > 0 else 1.0

        if ratio >= min_four_fifths and cost < best["cost"]:
            best = {"threshold": t, "cost": cost, "four_fifths_ratio": ratio, "feasible": True}

    return best
```

### 6. Cost-Matrix Threshold Optimization

Revisit the cost-optimal threshold from training with the evaluation lens. This section validates and potentially refines the threshold.

- Recompute the cost curve on the test set using the FP/FN costs from the problem statement
- Plot the cost surface: x = threshold, y = total cost, with lines for FP cost, FN cost, and total
- Mark the training-selected threshold and the test-optimal threshold
- If they differ by > 0.05, flag for review — the model may be overfitting the validation set's threshold

Log the final threshold decision and rationale to MLflow.

### 7. SHAP Stability Analysis

Verify that SHAP explanations are stable across data subsets — unstable explanations undermine trust.

```python
def shap_stability_check(model, X, n_subsets=5, subset_frac=0.6, seed=42):
    """Check SHAP value stability across random data subsets.

    Args:
        model: trained model
        X: feature matrix
        n_subsets: number of random subsets to evaluate
        subset_frac: fraction of data per subset
        seed: random seed

    Returns:
        dict with mean_rank_correlation (Spearman) and feature_rank_variance
    """
    import numpy as np
    import shap
    from scipy.stats import spearmanr

    rng = np.random.RandomState(seed)
    explainer = shap.Explainer(model, X)

    rankings = []
    for _ in range(n_subsets):
        idx = rng.choice(len(X), int(len(X) * subset_frac), replace=False)
        sv = explainer(X[idx])
        mean_abs = np.abs(sv.values).mean(axis=0)
        rankings.append(np.argsort(-mean_abs))

    # Pairwise rank correlation
    corrs = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            corrs.append(spearmanr(rankings[i], rankings[j]).correlation)

    return {
        "mean_rank_correlation": np.mean(corrs),
        "min_rank_correlation": np.min(corrs),
        "feature_rank_variance": np.var(rankings, axis=0).tolist(),
    }
```

**Gate:** Mean Spearman rank correlation across subsets should be > 0.85. If lower, feature importances are unstable — investigate whether the model is relying on noisy features.

### 8. Robustness Checks

Test whether model performance is stable under perturbation.

**Bootstrap confidence intervals:**

```python
def bootstrap_metric_ci(y_true, y_prob, metric_fn, n_bootstraps=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for any metric.

    Args:
        y_true: ground truth labels
        y_prob: predicted probabilities or labels
        metric_fn: callable(y_true, y_pred) -> float
        n_bootstraps: iterations
        ci: confidence level
        seed: random seed

    Returns:
        dict with point_estimate, ci_lower, ci_upper
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    estimates = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        estimates.append(metric_fn(y_true[idx], y_prob[idx]))

    alpha = (1 - ci) / 2
    return {
        "point_estimate": metric_fn(y_true, y_prob),
        "ci_lower": np.percentile(estimates, alpha * 100),
        "ci_upper": np.percentile(estimates, (1 - alpha) * 100),
    }
```

**Sensitivity analysis:**
- Perturb each feature by ±1 standard deviation and measure impact on predictions
- Features where small perturbations cause large prediction shifts should be flagged
- This surfaces fragile dependencies that could break under distribution shift

**Validation-to-test stability:**
- Compare key metrics (AUC, F1, Brier) between validation set and test set
- If the gap exceeds 0.03 (AUC) or 0.05 (F1), investigate overfitting to the validation set

### 9. Evaluation Summary & Comparison Table

Consolidate all findings into a single comparison table:

| Metric Category | Metric | Value | Threshold | Status |
|---|---|---|---|---|
| Discrimination | AUC-ROC (95% CI) | — | > MIN_AUC | — |
| Discrimination | Average Precision | — | > naive AP | — |
| Calibration | Brier Score | — | < MAX_BRIER | — |
| Calibration | ECE | — | < MAX_ECE | — |
| Fairness | Four-fifths ratio | — | ≥ 0.80 | — |
| Fairness | Max TPR gap | — | < 0.05 | — |
| Robustness | AUC CI width | — | < 0.05 | — |
| SHAP Stability | Mean rank correlation | — | > 0.85 | — |
| Baselines | Champion vs. best baseline lift | — | > 0% | — |

Log the full table to MLflow as an artifact.

### 10. Go/No-Go Promotion Decision

Based on all evaluation evidence, make a formal promotion decision:

| Decision | Criteria |
|---|---|
| **PROMOTE** | All gates passed. Model is ready for production deployment via `07_INFERENCING.md`. |
| **CONDITIONAL** | Minor issues found (e.g., fairness gap close to threshold, one unstable slice). Document conditions and monitoring requirements. Proceed with enhanced monitoring via `08_MONITORING.md`. |
| **REJECT** | Blocking issues found (e.g., fails fairness gate, AUC below minimum, unstable SHAP). Return to `05_MODEL_TRAINING.md` for iteration. |

```python
def promotion_decision(eval_summary):
    """Make Go/No-Go promotion decision based on evaluation summary.

    Args:
        eval_summary: dict with all evaluation metrics and gate statuses

    Returns:
        dict with decision (PROMOTE/CONDITIONAL/REJECT), rationale, and conditions
    """
    blockers = [k for k, v in eval_summary.items() if v.get("status") == "FAIL"]
    warnings = [k for k, v in eval_summary.items() if v.get("status") == "WARN"]

    if blockers:
        decision = "REJECT"
        rationale = f"Blocking failures: {', '.join(blockers)}"
        action = "Return to 05_MODEL_TRAINING.md for model iteration"
    elif warnings:
        decision = "CONDITIONAL"
        rationale = f"Warnings requiring monitoring: {', '.join(warnings)}"
        action = "Proceed with enhanced monitoring in 08_MONITORING.md"
    else:
        decision = "PROMOTE"
        rationale = "All evaluation gates passed"
        action = "Deploy via 07_INFERENCING.md"

    return {"decision": decision, "rationale": rationale, "action": action}
```

**Log the decision to MLflow:**
- Tag the model run with `promotion_decision = PROMOTE|CONDITIONAL|REJECT`
- Log the rationale and any conditions as artifacts
- If PROMOTE: transition the model to "Staging" in the registry (production transition happens during inferencing)
- If REJECT: log the specific failures and recommended next steps

### 11. Final MLflow Logging

- Log all evaluation metrics: AUC (with CI), Brier, ECE, MCE, fairness metrics, SHAP stability
- Log all charts: reliability diagram, ROC with CI, PR curve, confusion matrix, cost curve, fairness comparison, SHAP stability
- Log the evaluation summary table as a CSV artifact
- Log the promotion decision and rationale
- Tag the MLflow run with `evaluation_status = {PROMOTE|CONDITIONAL|REJECT}`

---

## Code Standards

- Every section has a Markdown cell explaining what is being done and why
- Constants at top of notebook: `MODEL_NAME`, `THRESHOLD`, `COST_FP`, `COST_FN`, `PROTECTED_ATTRS`, `MIN_AUC`, `MAX_BRIER`, `MAX_ECE`
- All evaluation plots use `plt.savefig()` and log to MLflow via `mlflow.log_artifact()`
- Functions > 15 lines go in `utils/eval_helpers.py`
- Cache DataFrames reused > twice; unpersist after use
- Every metric includes a confidence interval where applicable

---

## Constraints & Guardrails

- Do **NOT** skip the baseline comparison — a model that barely beats a majority-class classifier should not be promoted
- Do **NOT** use a single metric to make the promotion decision — the Go/No-Go gate considers calibration, fairness, stability, and robustness together
- Do **NOT** ignore fairness failures — if the four-fifths rule fails and no fairness-constrained threshold is feasible, the decision is REJECT
- Do **NOT** report metrics without confidence intervals — point estimates without uncertainty are misleading
- Do **NOT** cherry-pick slices — report worst-performing slices, not just overall performance
- Do **NOT** skip SHAP stability — unstable explanations undermine stakeholder trust and indicate fragile models
- Do **NOT** modify the model in this notebook — evaluation is observation only. If changes are needed, return to `05_MODEL_TRAINING.md`

---

## Acceptance Criteria

| Criterion | Definition of Done |
|---|---|
| Baseline comparison | Champion compared against 4+ baselines; beats all on primary metric |
| Calibration deep-dive | Brier, ECE, MCE computed; recalibration applied if thresholds exceeded |
| Discrimination analysis | ROC and PR curves with bootstrap CIs; lift/gain charts produced |
| Error analysis | Confusion cells profiled; worst-performing slices identified |
| Fairness assessment | Four-fifths rule, TPR gap, and FPR gap evaluated; fairness-constrained threshold explored if needed |
| Cost-matrix validation | Test-set cost curve computed; threshold validated against training selection |
| SHAP stability | Mean rank correlation > 0.85 across subsets |
| Robustness | Bootstrap CIs computed; validation-to-test stability checked |
| Promotion decision | Formal PROMOTE/CONDITIONAL/REJECT decision logged to MLflow with rationale |
| MLflow artifacts | All metrics, charts, summary table, and decision logged |

---

## Downstream Handoffs

| Recipient | What they receive | Playbook |
|---|---|---|
| ML Engineer | Promoted model (if PROMOTE/CONDITIONAL), threshold, monitoring requirements | `07_INFERENCING.md` |
| Data Scientist | Evaluation summary, conditions (if CONDITIONAL), rejection rationale (if REJECT) | `05_MODEL_TRAINING.md` (iteration) |
| Monitoring Engineer | Fairness slices, sensitivity analysis, SHAP baseline for drift comparison | `08_MONITORING.md` |
| DS Reviewer | Full evaluation notebook, promotion decision, summary table | Review gate |
| Stakeholder Communicator | Key metrics, fairness assessment, model strengths/limitations | `10_STAKEHOLDER_COMMUNICATION.md` |
