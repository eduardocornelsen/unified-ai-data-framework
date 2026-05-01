# Exploratory Data Analysis Playbook

## Role

You are a senior Data Scientist responsible for delivering a rigorous EDA on a tabular dataset. Your output is a well-structured notebook (`eda.ipynb`) that a business stakeholder can read top-to-bottom and an ML engineer can use as the foundation for feature engineering.

---

## Environment

**Compute:** Databricks Serverless or any Python environment with pandas/pyspark
**Primary API:** PySpark for large datasets; pandas for small datasets or post-aggregation analysis
**Visualization:** `display()` for Databricks native rendering; matplotlib/seaborn for custom charts; plotly for interactive exploration
**MLflow:** Log key statistics, summary tables, and plots via `mlflow.log_metric` / `mlflow.log_figure` for traceability

---

## Inputs

| Input | Source | Description |
|---|---|---|
| Problem statement | `templates/problem_statement.md` (from `00_PROBLEM_FRAMING.md`) | Business question, KPI, target variable, cost asymmetry, scope |
| Data contract | `data_contract.ipynb` (from `01_DATA_CONTRACT.md`) | Validated schema, quality rules, PII handling decisions |
| Dataset | Data platform / files | The contracted, validated dataset to explore |

If no problem statement or data contract exists yet, proceed with EDA but note the assumptions you are making.

---

## Deliverables

1. **`eda.ipynb`** — fully executed notebook with narrative, code, and visualizations
2. **`utils/eda_helpers.py`** — reusable helper functions extracted to `.py` scripts
3. **`requirements.txt`** — additional pip dependencies (beyond what the environment pre-installs)

---

## Notebook Structure

### 0. Setup

- Install any extra packages with `%pip install` (run in its own cell so the kernel restarts cleanly)
- Import libraries (pyspark.sql.functions, pandas, numpy, matplotlib, seaborn, plotly, scipy, etc.)
- Confirm Spark session is active (if using PySpark): `spark.version`
- Set global plot style and random seed
- Define constants: `DATA_PATH`, `TARGET_COL`, `RANDOM_SEED`
- Define reusable helper functions (or `%run ./utils/eda_helpers` to import from the utils script)

### 0.5. Problem Framing Carry-Over

If a problem statement exists, load it and extract the key context that shapes EDA:

- **Business question:** What specific question are we answering?
- **Target variable:** What are we predicting or measuring?
- **KPI and baseline:** What metric matters and where does it stand today?
- **Cost asymmetry:** Is a false positive or false negative more costly? (This shapes what segments to investigate.)
- **Scope boundaries:** Geographic, temporal, or segment constraints that limit the data.

Print these as a summary table at the top of the notebook. Every subsequent section should be read through this lens.

If no problem statement exists, define the target variable and business context from the user's description.

### 1. Data Loading & Initial Inspection

- Load the dataset using `spark.read` (Delta, Parquet, or CSV) or `pd.read_csv()` for local work
  - Prefer Delta format: `spark.read.format("delta").load("<path>")` or `spark.table("<catalog>.<schema>.<table>")`
  - If loading CSV: use `inferSchema=True` and then validate inferred types
- Display an interactive preview and print schema
- Report: row count, column count, sample of distinct values per column
- Check for duplicate rows
- Cross-reference against the data contract schema (if available) — flag any mismatches

### 2. Data Quality Assessment

- **Missing values:** count and percentage per column; visualize with a bar chart
- **Duplicates:** identify and document handling strategy
- **Type validation:** confirm numeric vs. categorical assignments are correct; cast if needed
- **Outlier scan:** compute per-column mean/std; flag columns with values beyond 3 sigma or IQR fences
- **Date/time parsing:** cast timestamp strings and extract `hour`, `day_of_week`, `month`
- **Delta table metadata** (if source is Delta): run `DESCRIBE HISTORY` to confirm data freshness

### 3. Univariate Analysis

For every feature, report:
- **Numeric:** mean, median, std, min, max, skewness, kurtosis — plot histograms and boxplots
- **Categorical:** value counts and top-N bar chart; flag high-cardinality columns (> 50 unique values)
- Highlight any suspicious distributions (near-zero variance, extreme skew, unexpected modes)

### 4. Target Variable Analysis

- Class distribution (for classification) or distribution shape (for regression): absolute counts, percentages, histogram
- Class imbalance ratio — flag if > 3:1
- Target rate by each categorical feature (bar charts sorted by rate)
- Target distribution bucketed by key numeric features (quartile bins)

### 5. Bivariate & Multivariate Analysis

- **Numeric vs. target:** box plots and violin plots for each numeric feature split by target
- **Categorical vs. target:** grouped bar charts, heatmaps (*e.g., category A x category B vs. target rate*)
- **Numeric vs. numeric:** scatter matrix / pairplot colored by target (subsample if large)

#### 5.1 Correlation Matrix (Assumption-Aware)

**Do NOT blindly compute Pearson correlation on all numeric features.** Pearson assumes approximately normal distributions and linear relationships.

```python
def assumption_aware_correlation(df, numeric_cols, alpha=0.05):
    """Compute Pearson or Spearman based on normality of each pair.

    Uses Shapiro-Wilk (n < 5000) or D'Agostino-Pearson (n >= 5000) to test normality.
    Falls back to Spearman if either variable in a pair is non-normal.
    """
    from scipy import stats
    results = {}
    pdf = df.select(numeric_cols).toPandas() if hasattr(df, 'toPandas') else df[numeric_cols]

    # Test normality for each column
    normality = {}
    for col in numeric_cols:
        data = pdf[col].dropna()
        if len(data) < 5000:
            _, p = stats.shapiro(data)
        else:
            _, p = stats.normaltest(data)
        normality[col] = p > alpha

    # Compute correlations
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i+1:]:
            if normality[col_a] and normality[col_b]:
                r, p = stats.pearsonr(pdf[col_a].dropna(), pdf[col_b].dropna())
                method = "Pearson"
            else:
                r, p = stats.spearmanr(pdf[col_a].dropna(), pdf[col_b].dropna())
                method = "Spearman"
            results[(col_a, col_b)] = {"r": r, "p": p, "method": method}

    return results
```

**Rules:**
- Annotate cells with |r| > 0.5
- Label each cell with "P" (Pearson) or "S" (Spearman) so the reader knows which method was used
- Flag strongly correlated pairs (|r| > 0.8) as potential multicollinearity candidates for FE

#### 5.2 Cramer's V Matrix (Categorical Associations)

For categorical feature pairs, compute Cramer's V. **Small-cell warning:** if any expected cell count < 5 in the chi-squared contingency table, the V statistic is unreliable — flag it.

```python
def cramers_v_with_warning(contingency_table):
    """Compute Cramer's V with small-cell warning."""
    from scipy.stats import chi2_contingency
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    small_cells = (expected < 5).sum()
    warning = f"WARNING: {small_cells} cells with expected count < 5" if small_cells > 0 else None

    return {"v": v, "p": p, "warning": warning}
```

### 6. Temporal Analysis

*(Skip if no timestamp or date column exists.)*

- Hourly / daily / weekly target rate line charts
- Day-of-week target rate bar chart
- Heatmap: hour x day_of_week vs. target rate
- Identify peak and off-peak patterns
- Check for **seasonality** — plot target rate by month/quarter if data spans > 3 months

### 7. Segment & Cohort Analysis

Group the data by key dimensions from the problem statement and analyze target rate:
- Target rate by each meaningful segment (*e.g., age groups, income quartiles, geographic regions*)
- Cross-segment analysis (*e.g., segment A x segment B target rates*)
- Top and bottom performers by target rate (horizontal bar charts)
- Flag segments with very small sample sizes (n < 30) — statistical claims on small segments are unreliable

**Sample-size adequacy:** For each segment, report the sample size and flag segments where n < 30 or where the segment represents < 1% of the total dataset. Claims about these segments should carry explicit caveats.

### 8. Domain-Specific Analysis

*(Adapt this section to the domain of the dataset.)*

If the dataset has domain-specific structure (*e.g., funnel stages, product categories, user journeys, financial transactions*), analyze it here:
- Domain-specific metrics and KPIs
- Funnel or flow analysis (if applicable)
- Domain-specific segmentation

### 9. Outlier & Anomaly Deep-Dive

- Investigate flagged outliers from Section 2 in context of the target
- Determine whether outliers are data errors or genuine extreme behavior
- Document the recommended treatment (cap, remove, keep, or flag as a feature)

### 10. Key Findings & Hypotheses

Bullet-point summary of the top 5-10 findings:
- For each finding: **observation** → **business interpretation** → **suggested follow-up**
- List features ranked by apparent predictive signal for the target
- Identify features that are likely redundant or low-value for modeling
- **Causal vs. correlational disclaimer:** explicitly state that EDA findings show associations, not causal relationships. If causal claims are needed, they require hypothesis testing (`03_HYPOTHESIS_TESTING.md`) or experimentation (`09_EXPERIMENTATION.md`).

### 11. Hypothesis Register (Handoff to Hypothesis Testing)

Build a structured hypothesis register from the findings in Section 10. This is the formal handoff to `03_HYPOTHESIS_TESTING.md`.

| # | Claim | H0 | H1 | Direction | Groups / Variables | Suggested Test | Priority |
|---|---|---|---|---|---|---|---|
| 1 | *e.g., "Users over 35 have a higher target rate"* | No difference by age group | Rate differs by age group | Two-tailed | age <= 35 vs. age > 35 | Two-proportion z-test | High |
| 2 | *e.g., "Income is negatively correlated with target"* | No correlation | Negative correlation | One-tailed (less) | income vs. target | Spearman rank | Medium |

**Rules:**
- Every strong finding from Section 10 should have at least one testable hypothesis
- H0 always states "no effect / no difference"
- Default to two-tailed unless there is strong prior justification for direction
- Mark priority: High (directly related to KPI), Medium (supporting signal), Low (exploratory)
- Minimum 5 hypotheses for a thorough EDA

---

## Code Standards

- Every section starts with a Markdown cell explaining *what* is being done and *why*
- No magic numbers — define constants (data path, target column name) at the top of the notebook
- Each visualization has: title, labeled axes, and a one-sentence insight caption below it
- Functions longer than ~15 lines go in `utils/eda_helpers.py`
- The notebook must run end-to-end with **Run All** without errors
- **Spark best practices** (if using PySpark):
  - Avoid `.collect()` on large DataFrames — aggregate first, then collect small results
  - Cache (`df.cache()`) only DataFrames reused more than twice; unpersist when done
  - Prefer Spark SQL aggregations over converting to pandas early
  - Use `F` alias for `pyspark.sql.functions` consistently throughout
- **MLflow logging:** at the end of the notebook, log key summary stats and all figures to an MLflow run for reproducibility

---

## Constraints & Guardrails

- Do **NOT** perform model training or feature engineering in this notebook — that is a separate step
- Do **NOT** drop columns or rows silently — document every removal decision
- Flag but do **NOT** automatically fix data quality issues; document the recommended fix
- Do **NOT** compute Pearson correlation without checking normality — use Spearman as default for non-normal data
- Do **NOT** compute chi-squared statistics when expected cell counts are < 5 — use Fisher's exact test instead
- Do **NOT** make causal claims from correlational findings — use "associated with", not "caused by"
- Do **NOT** draw conclusions from segments with n < 30 without explicit sample-size caveats
- Keep individual cells focused — one analysis per cell
- Total notebook length: aim for **30-60 cells**

---

## Acceptance Criteria

| Criterion | Definition of Done |
|---|---|
| Problem framing | Section 0.5 references the problem statement (or documents assumptions if none exists) |
| Coverage | All features analyzed in at least one section |
| Target analysis | Class imbalance / distribution documented; target rate computed per segment |
| Visualizations | Minimum 15 distinct charts, each with caption |
| Quality report | Missing values, duplicates, and outliers explicitly reported |
| Correlation method | Pearson/Spearman chosen per pair based on normality; labeled in matrix |
| Hypothesis register | At least 5 testable hypotheses with H0, H1, direction, and priority |
| Causal disclaimer | Section 10 explicitly states findings are correlational, not causal |
| Reproducibility | Notebook runs clean from top to bottom |
| MLflow | Key metrics and at least 3 figures logged to an MLflow run |

---

## Downstream Handoffs

| Recipient | What they receive | Playbook |
|---|---|---|
| Data Scientist | Hypothesis register for formal testing | `03_HYPOTHESIS_TESTING.md` |
| Data Scientist | Feature signal rankings, outlier treatment recommendations | `04_FEATURE_ENGINEERING.md` |
| Data Scientist Reviewer | EDA notebook for methodology gate review | Any downstream playbook |
