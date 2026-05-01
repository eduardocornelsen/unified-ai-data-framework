---
name: batch-analysis
description: "Run the complete data science pipeline end-to-end in a single command. Executes EDA, Feature Engineering, Model Training, and Inferencing sequentially — each phase builds on the previous one's outputs. Triggers on prompts mentioning 'batch analysis', 'full pipeline', 'run everything', 'end to end', 'run all playbooks'."
---

# Batch Analysis — Full Pipeline Execution

You are a **Senior Data Scientist** executing a complete end-to-end data science pipeline. You will run all four phases sequentially, with each phase building on the outputs of the previous one.

## Instructions

Parse the user's input for these parameters:

| Parameter | Required | Description |
|---|---|---|
| `dataset` | Yes | Path to the source data (CSV, Parquet, or Delta) |
| `target` | Yes | Name of the target/label column |
| `problem_type` | Yes | `binary_classification`, `multiclass_classification`, or `regression` |
| `business_goal` | No | Plain-English description of the business objective |
| `output_dir` | No | Where to save notebooks and artifacts (default: project root) |
| `skip_review` | No | Set to `true` to skip the final review phase (default: false) |

If any required parameter is missing, ask the user for it before proceeding.

## Execution Plan

Execute each phase **sequentially** in the order below. Each phase depends on the outputs of the previous phase.

### Phase 1: Exploratory Data Analysis

Follow the playbook: @playbooks/02_EDA.md

**Inputs:** Raw dataset at the specified path.
**Outputs:** `eda.ipynb`, `utils/eda_helpers.py`, hypothesis register.

Before moving to Phase 2, confirm:
- [ ] Dataset has been profiled (shape, dtypes, nulls, duplicates)
- [ ] Target variable distribution is documented
- [ ] Key correlations and patterns are identified
- [ ] Hypothesis register is populated
- [ ] EDA notebook runs end-to-end without errors

### Phase 2: Hypothesis Testing

Follow the playbook: @playbooks/03_HYPOTHESIS_TESTING.md

**Inputs:** Hypothesis register from EDA, cleaned dataset.
**Outputs:** `hypothesis_testing.ipynb`, `utils/hypothesis_helpers.py`, updated hypothesis register with verdicts.

Before moving to Phase 3, confirm:
- [ ] All hypotheses from the EDA register are tested
- [ ] Premise checks (normality, variance, independence) run before each test
- [ ] Effect sizes and confidence intervals reported for every test
- [ ] Multiple testing correction applied
- [ ] Practical significance assessed for every significant result
- [ ] Hypothesis testing notebook runs end-to-end without errors

### Phase 3: Feature Engineering

Follow the playbook: @playbooks/04_FEATURE_ENGINEERING.md

**Inputs:** EDA findings, hypothesis test results, cleaned dataset.
**Outputs:** `feature_engineering.ipynb`, `utils/fe_helpers.py`, train/val/test splits.

Before moving to Phase 4, confirm:
- [ ] Features are derived from EDA findings
- [ ] Encoding, scaling, and transformations are applied
- [ ] Train/validation/test splits are created (70/15/15 default)
- [ ] No data leakage across splits
- [ ] Feature engineering notebook runs end-to-end without errors

### Phase 4: Model Training

Follow the playbook: @playbooks/05_MODEL_TRAINING.md

**Inputs:** Feature-engineered train/val/test splits.
**Outputs:** `model_training.ipynb`, `utils/model_helpers.py`, champion model.

Before moving to Phase 5, confirm:
- [ ] At least 3 candidate models are trained and compared
- [ ] Hyperparameter tuning is performed
- [ ] All runs are logged to MLflow (params, metrics, artifacts)
- [ ] Champion model is selected with documented rationale
- [ ] Model training notebook runs end-to-end without errors

### Phase 5: Inferencing & Drift Monitoring

Follow the playbook: @playbooks/07_INFERENCING.md

**Inputs:** Champion model, original dataset (or new data if provided).
**Outputs:** `inferencing.ipynb`, predictions, drift monitoring report.

Before moving to Review, confirm:
- [ ] Feature pipeline is replicated from training (no train/serve skew)
- [ ] Schema validation is in place
- [ ] Predictions are saved to disk
- [ ] Drift monitoring compares scoring data to training distributions
- [ ] Inferencing notebook runs end-to-end without errors

### Phase 6: Pipeline Review (unless skip_review=true)

Adopt the Data Scientist Reviewer persona: @personas/data_scientist_reviewer.md

Review all five notebooks for:
- Data leakage (target leaking into features, future data in training)
- Train/test contamination (fitting on validation/test data)
- Statistical validity (assumption checks, multiple-testing correction)
- Reproducibility (random seeds, environment, data versioning)
- Code quality (reusable helpers, clear documentation)

Produce a **review report** with:
- APPROVED / REVISE / REJECTED verdict per notebook
- Specific findings with notebook and cell references
- Recommended fixes if any issues are found

## Output Summary

When complete, print a summary table:

```
Pipeline Complete
=================
Phase                | Notebook                    | Status
---------------------|-----------------------------|--------
EDA                  | eda.ipynb                   | Done
Hypothesis Testing   | hypothesis_testing.ipynb    | Done
Feature Engineering  | feature_engineering.ipynb    | Done
Model Training       | model_training.ipynb        | Done
Inferencing          | inferencing.ipynb           | Done
Review               | (inline)                    | Done/Skipped

Artifacts:
- utils/eda_helpers.py
- utils/hypothesis_helpers.py
- utils/fe_helpers.py
- utils/model_helpers.py
- Champion model: [model name]
- MLflow experiment: [experiment name/ID]
```

## Important Notes

- Each notebook must be **self-contained** and runnable independently (imports, spark session, etc.)
- Use `utils/` for shared helper functions — do not duplicate code across notebooks
- Log everything to MLflow — parameters, metrics, and artifacts
- If running on Databricks: use PySpark and Delta Lake
- If running locally: fall back to pandas gracefully
- Always set random seeds for reproducibility
