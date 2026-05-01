# Feature Engineering Playbook

## Role

You are a senior Data Scientist building a production-ready feature set for a supervised learning task. Your input is a profiled, quality-checked dataset and the findings from EDA. Your output is a fully executed notebook (`feature_engineering.ipynb`) that produces feature-engineered train/validation/test splits ready for model training.

---

## Environment

**Compute:** Databricks Serverless or any Python environment with pandas/pyspark
**Primary API:** PySpark (`pyspark.sql.functions`, `pyspark.ml`) for large datasets; scikit-learn/pandas for local work
**Storage:** Delta Lake preferred; Parquet/CSV acceptable
**Experiment tracking:** MLflow — log all transformations, feature metadata, and the final schema
**Output:** Delta table (or persisted Parquet) at a configurable `OUTPUT_PATH`

---

## Inputs

| Input | Source | Description |
|---|---|---|
| Problem statement | `templates/problem_statement.md` (from `00_PROBLEM_FRAMING.md`) | KPI, target variable, cost asymmetry, scope boundaries |
| EDA findings | `eda.ipynb` (from `02_EDA.md`) | Feature signal rankings, correlation analysis, outlier recommendations, distribution shapes |
| Hypothesis test results | `hypothesis_testing.ipynb` (from `03_HYPOTHESIS_TESTING.md`) | Validated associations, effect sizes, confirmed/rejected hypotheses |
| Dataset | Data platform / files | The profiled, quality-checked dataset |

---

## Deliverables

1. **`feature_engineering.ipynb`** — fully executed notebook
2. **`utils/fe_helpers.py`** — reusable PySpark/pandas transformer helpers
3. **Updated `requirements.txt`** — any new pip dependencies

---

## Notebook Structure

### 0. Setup

- `%pip install` extra packages in an isolated cell
- Import PySpark, MLlib, pandas, sklearn, MLflow
- Define all constants at the top: `INPUT_PATH`, `OUTPUT_PATH`, `TARGET_COL`, `RANDOM_SEED`, `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`
- Load Spark session (if using PySpark) and confirm version
- Import shared helpers from `utils/fe_helpers.py`

### 1. Load & Validate Input

- Load the dataset using `spark.read` or `pd.read_csv()`
- Assert expected columns are present — fail fast with a descriptive error if not
- Assert row count is within expected range (sanity check)
- Cross-reference against the data contract schema (if available)
- Print schema and a sample row

### 1.5. Leakage Audit

**Before building any features, audit every candidate feature for data leakage.** This is the most common and most damaging mistake in feature engineering.

| Leakage Type | Description | How to Detect | Example |
|---|---|---|---|
| **Target leakage** | Feature encodes the target directly or through a proxy | Feature has near-perfect correlation with target; feature is computed from target | `total_purchases` used to predict `will_purchase` when it includes the target event |
| **Temporal leakage** | Feature uses future information unavailable at prediction time | Feature timestamp is after the prediction point | Using `next_month_revenue` to predict `will_churn_this_month` |
| **Train-test contamination** | Statistics from test data leak into training features | Encoding maps, scalers, or imputers fitted on full dataset instead of train only | Global mean imputation instead of train-mean imputation |

For each candidate feature, fill in the audit table:

| Feature | Leakage Risk | Timing | Available at Prediction Time? | Decision |
|---|---|---|---|---|
| *e.g., `total_sessions`* | Low | Historical | Yes (computed before prediction point) | Keep |
| *e.g., `outcome_label`* | **Critical** | Same event | No (is the target) | **Remove** |
| *e.g., `next_day_login`* | **Critical** | Future | No (future information) | **Remove** |

**Rules:**
- If a feature has near-perfect predictive power (AUC > 0.99 alone), investigate for leakage before celebrating
- When the dataset has timestamps, define a **prediction point** and verify every feature is computable before it
- When target-encoding or computing aggregates, fit on training data only — never on the full dataset

### 2. Temporal Feature Engineering

*(Skip if no timestamp column exists.)*

**Source columns:** Any timestamp or date columns
**Actions:**
- Cast to proper `TimestampType` or `DateType`
- Extract: `hour`, `day_of_week` (1-7), `month`, `is_weekend` (boolean)
- **Cyclical encoding** for periodic features:
  ```python
  hour_sin = sin(2 * pi * hour / 24)
  hour_cos = cos(2 * pi * hour / 24)
  dow_sin  = sin(2 * pi * day_of_week / 7)
  dow_cos  = cos(2 * pi * day_of_week / 7)
  ```
  Rationale: prevents the model from treating hour 23 and hour 0 as far apart
- Drop raw timestamp after encoding (keep `month` as ordinal if useful)

**Temporal split option:** If the data has a natural time ordering and the prediction task is temporal (*e.g., forecasting, churn prediction*), consider a **time-based split** instead of random split:
- Train: all data before cutoff date T1
- Validation: data between T1 and T2
- Test: data after T2

This prevents temporal leakage and better simulates production conditions. Document which split strategy is used and why.

### 3. Numeric Feature Engineering

**Actions (adapt to your dataset):**
- **Interaction terms:** Multiply features that showed combined signal in EDA (*e.g., `feature_a * feature_b`*)
- **Log transforms** for right-skewed features: `log1p(x)` (handles zeros)
- **Binning:** Group continuous features into meaningful categories (*e.g., age groups, income quartiles*) using domain knowledge or quantile-based cuts
- **Outlier flags:** For each numeric column, add `is_outlier_<col>` boolean using IQR bounds computed on **training data only** (pass bounds as parameters — never recompute on test set)
- **Scaling:** Apply `MinMaxScaler` or `StandardScaler` after vectorization (record scaler params for inference reuse)

### 4. Categorical Feature Engineering

**Actions (adapt to your dataset):**
- **Low-cardinality** (< 10 categories): `StringIndexer` → `OneHotEncoder` (drop one category to avoid dummy trap)
- **High-cardinality** (10-1000 categories): Target encoding with smoothing:
  ```python
  encoded_value = (category_target_count + alpha * global_mean) / (category_count + alpha)
  ```
  - Smoothing `alpha` prevents overfitting to rare categories
  - **Fit on training data only** — store encoding map for val/test/inference
  - Log the encoding map to MLflow as a JSON artifact
- **Very high cardinality** (> 1000): Consider frequency encoding (`log(count)`), hash encoding, or embedding (if deep learning)
- Drop raw string columns after encoding

### 4.5. Fairness-Proxy Audit

**Before encoding protected attributes, assess whether they (or their proxies) should be included in the model.**

```python
def audit_fairness_proxies(df, protected_cols, candidate_features, target_col, threshold=0.3):
    """Flag features that are strong proxies for protected attributes.

    A feature that correlates > threshold with a protected attribute
    may introduce unfair bias even if the protected attribute itself is excluded.
    """
    from scipy import stats
    alerts = []
    pdf = df.toPandas() if hasattr(df, 'toPandas') else df

    for protected in protected_cols:
        for feature in candidate_features:
            if feature == protected:
                continue
            try:
                r, p = stats.spearmanr(pdf[protected], pdf[feature])
                if abs(r) > threshold:
                    alerts.append({
                        "protected": protected, "feature": feature,
                        "correlation": round(r, 3), "p_value": round(p, 6),
                    })
            except (TypeError, ValueError):
                pass

    return alerts
```

Document:
- Which columns are protected attributes (*e.g., gender, race, age, zip code*)
- Which features are proxies for protected attributes (correlation > 0.3)
- The decision: include, exclude, or monitor for fairness during model evaluation

This does not replace model-level fairness evaluation (see `06_MODEL_EVALUATION.md`), but it catches the obvious cases early.

### 5. Text Feature Engineering

*(Skip if no text columns exist.)*

**Actions:**
- **Keyword flags:** Create binary columns for domain-relevant keywords identified in EDA
- **TF-IDF:**
  - Tokenize → remove stop words → hashing TF → IDF → dimensionality reduction
  - Reduce to top N features using chi-squared selection against the target
- **Embeddings** (if available): Use pre-trained sentence embeddings for richer representation
- Drop raw text columns after encoding

### 6. Feature Assembly & Scaling

- Collect all engineered feature columns into a single vector using `VectorAssembler` (PySpark) or concatenation (pandas)
  - Exclude: raw string columns, timestamp columns, `TARGET_COL`
- Apply `MinMaxScaler` or `StandardScaler` to the assembled vector
- Keep `TARGET_COL` (renamed to `label` for MLlib compatibility) alongside the feature vector
- Print the full feature list with types and value ranges

### 7. Train / Validation / Test Split

- Split using configured ratios (default: `train=0.70, val=0.15, test=0.15`)
- For **random split:** `df.randomSplit([TRAIN_RATIO, VAL_RATIO, TEST_RATIO], seed=RANDOM_SEED)`
- For **temporal split:** use date-based cutoffs (see Section 2)
- **Critical:** Fit all encoders (target encoder, TF-IDF, scaler) on **training set only**; transform val and test using fitted params
- Report class balance (classification) or target distribution (regression) in each split — flag if any split deviates significantly from overall

### 8. Feature Importance Proxy

Before modeling, get a quick signal ranking:
- **Correlation with target:** absolute Pearson r (or Spearman ρ for non-normal) for each numeric feature
- **Chi-squared scores:** for binary/ordinal features
- Produce a ranked feature importance bar chart
- Identify and flag any near-zero-variance features (`std < 0.01`) for potential removal

### 9. Save Engineered Dataset

- Write train/val/test splits to Delta (or Parquet) at `OUTPUT_PATH/{train,val,test}`
- Save transformer artifacts (scaler model, encoding maps, TF-IDF pipeline) to `OUTPUT_PATH/artifacts/`
- Save training-set reference statistics (mean, std, min, max per feature) to `OUTPUT_PATH/artifacts/train_stats.json` — used by `07_INFERENCING.md` for drift monitoring
- Print final schema and row counts per split

### 10. MLflow Logging & Feature Registry

- Log to an MLflow run:
  - **Params:** all constants, feature counts, split sizes, split strategy (random vs. temporal)
  - **Metrics:** train/val/test row counts, class balance per split, near-zero-variance feature count
  - **Artifacts:** encoding maps (JSON), feature list (CSV), leakage audit table, fairness proxy audit, at least 3 charts
- Log the final feature schema as an MLflow tag `feature_schema`

---

## Code Standards

- Every section starts with a Markdown cell explaining *what* and *why*
- All transformer parameters defined as constants at the top — no magic numbers inside cells
- Encoders fitted on training data only; transformations applied consistently to all splits
- Functions > 15 lines go in `utils/fe_helpers.py`
- The notebook must run end-to-end with **Run All** without errors
- **Spark best practices** (if using PySpark):
  - Cache `df_train` after fitting steps (reused multiple times); unpersist before saving
  - Use `F` alias for `pyspark.sql.functions` throughout
  - No `.collect()` on unfiltered DataFrames
  - `%pip install` in its own first cell

---

## Constraints & Guardrails

- Do **NOT** train any ML model in this notebook — that is the next step
- Do **NOT** recompute encoding statistics on validation or test data — this is data leakage
- Do **NOT** skip the leakage audit (Section 1.5) — it is the single most impactful quality gate in FE
- Do **NOT** drop the `TARGET_COL` / `label` column from saved splits
- Do **NOT** silently coerce nulls — raise or log a warning if unexpected nulls appear post-join
- Do **NOT** hardcode file paths — use the `OUTPUT_PATH` constant throughout
- Do **NOT** encode protected attributes without documenting the fairness proxy audit (Section 4.5)
- Target-encoding smoothing **must** use training data statistics only

---

## Acceptance Criteria

| Criterion | Definition of Done |
|---|---|
| Leakage audit | Every candidate feature assessed for target, temporal, and train-test leakage |
| EDA-driven features | Features built based on EDA findings and hypothesis test results |
| No data leakage | All encoders fitted on train only; temporal split used if applicable |
| Fairness audit | Protected attributes and proxies documented with include/exclude decision |
| Splits | Train/val/test split with reproducible seed or temporal cutoff; balance reported |
| Output persisted | Splits written to disk; transformer artifacts saved for inference reuse |
| Feature ranking | Correlation + chi-squared proxy importance chart produced |
| MLflow logged | Params, metrics, leakage audit, fairness audit, and >= 3 artifacts logged |
| Reproducibility | Notebook runs clean top-to-bottom |

---

## Downstream Handoffs

| Recipient | What they receive | Playbook |
|---|---|---|
| Data Scientist | Train/val/test splits, feature vector, label column | `05_MODEL_TRAINING.md` |
| ML Engineer | Fitted transformer artifacts for inference pipeline | `07_INFERENCING.md` |
| Data Scientist Reviewer | Leakage audit, fairness audit, feature list | Any downstream playbook |
