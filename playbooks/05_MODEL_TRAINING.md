# Model Training Playbook

## Role

You are a senior Data Scientist building and selecting the best model for a supervised learning task. Your input is the feature-engineered splits produced by `feature_engineering.ipynb`. Your output is a fully executed notebook (`model_training.ipynb`) that trains multiple candidate models, establishes baselines, tunes the best candidate, evaluates calibration and fairness, and registers the champion to the MLflow Model Registry.

---

## Environment

**Compute:** Databricks Serverless or any Python environment with PySpark MLlib / scikit-learn
**Storage:** Delta Lake — read splits from `fe_output/{train,val,test}`
**ML Framework:** PySpark MLlib (primary) or scikit-learn (local); adapt to the environment
**Experiment Tracking:** MLflow — `mlflow.autolog()` + manual metric logging
**Model Registry:** MLflow Model Registry — register champion model

---

## Inputs

| Input | Source | Description |
|---|---|---|
| Train split | `fe_output/train` | Feature-engineered training data with `features` and `label` columns |
| Validation split | `fe_output/val` | Same schema — used for model selection and tuning |
| Test split | `fe_output/test` | Same schema — **touched only in final evaluation (Section 9)** |
| Problem statement | `templates/problem_statement.md` | Cost asymmetry, KPI, target definition |
| FE artifacts | `fe_output/artifacts/` | Fitted scalers, encoders, pipelines for inference |

---

## Deliverables

1. **`model_training.ipynb`** — fully executed notebook
2. **`utils/model_helpers.py`** — evaluation, plotting, and MLflow helpers
3. **Champion model** — saved as MLflow model artifact and registered in the Model Registry

---

## Notebook Structure

### 0. Setup

- `%pip install` extra packages in its own cell (seaborn, scikit-learn, shap)
- Import ML framework, MLflow, pandas, sklearn metrics
- Define constants: `FE_OUTPUT_PATH`, `MODEL_NAME`, `RANDOM_SEED`, `METRIC_FOR_SELECTION`, `PREDICTION_THRESHOLD`
- Start MLflow experiment: `mlflow.set_experiment("<project>_model_training")`
- Load the cost asymmetry from the problem statement — this drives threshold selection in Section 8

### 1. Load Feature-Engineered Splits

- Load `train`, `val`, `test` splits from Delta / Parquet
- Assert expected columns present: `features` (or feature vector), `label`
- Print row counts and class balance (classification) or target distribution (regression) per split
- **Do NOT touch `test` split until Section 9**

### 2. Final Feature Assembly

- If features are in multiple columns, use `VectorAssembler` to combine into a single `features` column
- Print total feature dimensionality
- Cache training and validation DataFrames (reused across all model fits)

### 3. Naive Baseline (Required)

**Every model comparison must include a naive baseline.** A model that cannot beat a trivial heuristic has no business in production.

| Task Type | Naive Baseline | Implementation |
|---|---|---|
| Binary classification | Predict the majority class for every example | `baseline_accuracy = max(class_0_pct, class_1_pct)` |
| Multiclass classification | Predict the most frequent class | Same as above |
| Regression | Predict the training mean for every example | `baseline_rmse = std(y_train)` |

```python
def compute_naive_baseline(y_train, y_val, task_type="classification"):
    """Compute naive baseline metrics for comparison.

    Every candidate model must beat this to be worth considering.
    """
    if task_type == "classification":
        from collections import Counter
        majority_class = Counter(y_train).most_common(1)[0][0]
        baseline_accuracy = sum(1 for y in y_val if y == majority_class) / len(y_val)
        return {"model": "Naive Baseline (majority class)",
                "accuracy": baseline_accuracy, "auc_roc": 0.5, "auc_pr": None}
    else:
        import numpy as np
        mean_pred = np.mean(y_train)
        baseline_rmse = np.sqrt(np.mean((y_val - mean_pred) ** 2))
        return {"model": "Naive Baseline (mean)", "rmse": baseline_rmse, "r2": 0.0}
```

Log the naive baseline to MLflow. All subsequent models are compared against it.

### 4. Train Candidate Model 1 — Interpretable Baseline

Train a simple, interpretable model as the first real candidate:
- **Classification:** Logistic Regression (`LogisticRegression` in MLlib or sklearn)
- **Regression:** Linear Regression or Ridge Regression

Fit on training data; evaluate on validation data. Log metrics to MLflow.

### 5. Train Candidate Model 2 — Ensemble

Train a stronger ensemble model:
- **Classification:** Random Forest (`RandomForestClassifier`)
- **Regression:** Random Forest Regressor

Fit and evaluate. Extract and display top-20 feature importances. Log metrics.

### 6. Train Candidate Model 3 — Gradient Boosting

Train the expected champion for tabular data:
- **Classification:** Gradient Boosted Trees (`GBTClassifier`) or XGBoost/LightGBM
- **Regression:** GBT Regressor or XGBoost/LightGBM

Fit and evaluate. Extract and display feature importances. Log metrics.

### 7. Hyperparameter Tuning — Best Candidate

- Identify the best model from Sections 4-6 by `METRIC_FOR_SELECTION`
- Run `TrainValidationSplit` or `CrossValidator` on the winner:
  - Grid: 2-3 key hyperparameters, 3-4 values each (keep total fits manageable)
  - Evaluator: use the primary metric from the problem statement
- Log best params and tuned validation metrics to MLflow

### 8. Cost-Aware Threshold Tuning (Classification Only)

**Do NOT use the default 0.5 threshold.** The optimal threshold depends on the cost asymmetry defined in the problem statement.

```python
def find_optimal_threshold(y_true, y_prob, fp_cost, fn_cost, thresholds=None):
    """Find the threshold that minimizes total cost.

    Args:
        y_true: actual labels
        y_prob: predicted probabilities for positive class
        fp_cost: cost of a false positive
        fn_cost: cost of a false negative
        thresholds: array of thresholds to evaluate (default: 0.01 to 0.99)

    Returns:
        optimal_threshold, min_cost, cost_curve
    """
    import numpy as np
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        total_cost = fp * fp_cost + fn * fn_cost
        costs.append(total_cost)

    min_idx = np.argmin(costs)
    return thresholds[min_idx], costs[min_idx], list(zip(thresholds, costs))
```

- Plot the cost curve (threshold vs. total cost)
- Compare: default 0.5 threshold, F1-optimal threshold, cost-optimal threshold
- Document which threshold is selected and why

### 8.5. Calibration Assessment

**A well-calibrated model's predicted probability of 0.7 should correspond to ~70% actual positive rate.** Poorly calibrated models produce misleading confidence scores.

```python
def plot_calibration(y_true, y_prob, n_bins=10, model_name="Model"):
    """Reliability diagram + Brier score."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    brier = brier_score_loss(y_true, y_prob)

    # Plot reliability diagram
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Plot — Brier Score: {brier:.4f}")
    ax.legend()
    plt.tight_layout()
    return fig, brier
```

If the model is poorly calibrated (Brier > 0.25 or visible deviation from diagonal):
- Apply **Platt scaling** (logistic recalibration) or **isotonic regression** on the validation set
- Re-evaluate calibration after correction
- Log both pre- and post-calibration Brier scores

### 8.6. Fairness Slicing (If Protected Attributes Exist)

If the fairness-proxy audit in `04_FEATURE_ENGINEERING.md` identified protected attributes, evaluate model fairness across slices:

```python
def evaluate_fairness_slices(y_true, y_pred, y_prob, group_labels, group_name):
    """Compute metrics per group for fairness assessment.

    Reports TPR, FPR, precision, and selection rate per group.
    Flags disparities > 20% relative difference (80% rule).
    """
    from sklearn.metrics import confusion_matrix
    import pandas as pd

    results = []
    for group in sorted(set(group_labels)):
        mask = group_labels == group
        if mask.sum() < 10:
            continue
        tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask]).ravel()
        results.append({
            "group": group, "n": int(mask.sum()),
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "selection_rate": (tp + fp) / mask.sum(),
        })
    return pd.DataFrame(results)
```

- Compare TPR, FPR, and selection rate across groups
- Flag disparities where the ratio between worst and best group < 0.8 (the "80% rule" / four-fifths rule)
- Document findings — this is informational, not a go/no-go gate (that's `06_MODEL_EVALUATION.md`)

### 9. Model Comparison on Validation Set

- Collect all models' validation metrics (including naive baseline) into a comparison table
- Bar chart: primary metric side-by-side for all candidates
- ROC curves overlaid on a single plot (classification)
- Precision-Recall curves overlaid (classification)
- Identify and print the champion model
- **The champion must beat the naive baseline** — if no model beats it, reconsider the feature engineering or problem framing

### 10. Final Evaluation on Test Set

- Apply **only the champion model** to the test split (first and only time)
- Report all relevant metrics:
  - **Classification:** AUC-ROC, AUC-PR, F1, Precision, Recall, Accuracy (at cost-optimal threshold)
  - **Regression:** RMSE, MAE, R², MAPE
- Confusion matrix heatmap (classification)
- Calibration plot (classification)
- Log all test metrics to MLflow with `test_` prefix

### 11. Register Champion to MLflow Model Registry

- Log the champion model: `mlflow.spark.log_model(champion, "model", registered_model_name=MODEL_NAME)`
- Transition the new version to `"Production"` stage (or "Staging" if shadow mode is planned — see `07_INFERENCING.md`)
- Log as MLflow tags: feature schema, training data path, model type, optimal threshold, Brier score
- Print the registered model URI

### 12. MLflow Summary

- Log final comparison table as a CSV artifact
- Log all charts as PNG artifacts (ROC, PR, calibration, cost curve, fairness slices)
- Set tags: `champion_model_type`, `feature_version`, `test_<primary_metric>`, `optimal_threshold`

---

## Code Standards

- Every section has a Markdown cell explaining what and why
- All model hyperparameters defined as constants at the top — no magic numbers inside cells
- `%pip install` in its own first cell
- `display()` for interactive DataFrames; matplotlib/seaborn for saved charts
- Cache DataFrames reused > twice; unpersist after use
- Functions > 15 lines go in `utils/model_helpers.py`

---

## Constraints & Guardrails

- Do **NOT** use the test set before Section 10 — report test metrics once only
- Do **NOT** tune hyperparameters based on test set performance
- Do **NOT** skip the naive baseline — it is the sanity check for the entire pipeline
- Do **NOT** use the default 0.5 threshold without considering cost asymmetry — the optimal threshold is rarely 0.5
- Do **NOT** ignore calibration — a model with good AUC but poor calibration produces misleading probability estimates
- Do **NOT** hardcode file paths — use constants
- The registered model must include the full `Pipeline` (assembler + classifier), not just the classifier

---

## Acceptance Criteria

| Criterion | Definition of Done |
|---|---|
| Naive baseline | Majority-class (or mean) baseline computed and logged; all models compared against it |
| Three+ models trained | At least 3 candidate models (including interpretable baseline) fitted and evaluated on validation |
| Hyperparameter tuning | Grid search or random search run on the best candidate |
| Cost-aware threshold | Optimal threshold selected based on FP/FN cost asymmetry; cost curve plotted |
| Calibration assessed | Reliability diagram and Brier score computed; recalibration applied if needed |
| Fairness sliced | If protected attributes exist, TPR/FPR/selection rate compared across groups |
| Comparison charts | ROC, PR, metric bar charts, and calibration plots produced |
| Test evaluation | Champion evaluated on test set exactly once with cost-optimal threshold |
| Model registered | Champion Pipeline in MLflow Model Registry |
| MLflow run | All params, metrics (val + test), threshold, calibration, and >= 3 charts logged |
| Reproducibility | Notebook runs clean top-to-bottom |

---

## Downstream Handoffs

| Recipient | What they receive | Playbook |
|---|---|---|
| ML Engineer | Registered champion model, optimal threshold, feature schema | `07_INFERENCING.md` |
| Data Scientist Reviewer | Comparison table, calibration plot, fairness slices, test metrics | Review gate |
| Data Scientist | Model performance for evaluation deep-dive | `06_MODEL_EVALUATION.md` |
