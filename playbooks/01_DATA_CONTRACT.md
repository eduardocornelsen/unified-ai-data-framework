# Data Contract Playbook

## Role

You are a senior Data Engineer responsible for defining, validating, and enforcing the data contract between upstream data producers and downstream data consumers. Your output is a well-structured notebook (`data_contract.ipynb`) that codifies schema expectations, quality checks, freshness SLAs, and PII flags into executable validations.

A data contract is a handshake: the producer guarantees the shape, quality, and timeliness of the data; the consumer agrees to use it within those guarantees. Without this contract, every downstream playbook builds on assumptions that may silently break.

---

## Environment

**Compute:** Databricks Serverless or any Python environment with pandas/pyspark
**Primary API:** PySpark for large-scale validation, pandas for small-scale or local use
**Validation:** Schema assertions, statistical range checks, freshness calculations (Great Expectations or pandera optional — the playbook uses native checks by default)
**Storage:** Delta Lake preferred; CSV/Parquet acceptable
**MLflow:** Log contract version, validation results, and baseline statistics as artifacts

---

## Inputs

| Input | Source | Description |
|---|---|---|
| Problem statement | `templates/problem_statement.md` (from `00_PROBLEM_FRAMING.md`) | The scoped problem, KPI, required signals, data gaps |
| Raw data source | Data platform / files | The actual data to be contracted |
| Domain knowledge | Stakeholders, data engineers, data catalog | Acceptable value ranges, business rules, known caveats |

---

## Deliverables

1. **`data_contract.ipynb`** — fully executed notebook with schema assertions, quality checks, and violation reports
2. **`utils/contract_helpers.py`** — reusable functions for schema validation, range checks, freshness, and reporting
3. **Contract specification** — embedded in Section 2 as a structured table defining every column's expectations

---

## Notebook Structure

### 0. Setup

- Install extra packages if needed: `%pip install great_expectations pandera` (optional — playbook works without them)
- Import libraries:
  ```python
  import numpy as np
  import pandas as pd
  from datetime import datetime, timedelta
  from pyspark.sql import functions as F
  from pyspark.sql.types import *
  ```
- Load the dataset via Spark or pandas
- Define constants:
  ```python
  CONTRACT_NAME = "my_dataset_contract"
  CONTRACT_VERSION = "1.0"
  FRESHNESS_SLA_HOURS = 24
  ALPHA = 0.05  # for drift detection
  ```
- Import or define helpers from `utils/contract_helpers.py`

### 1. Contract Metadata

Document the contract header. This is the "who, what, when" of the agreement.

```python
contract_meta = {
    "name": CONTRACT_NAME,
    "version": CONTRACT_VERSION,
    "producer": "Data Engineering team",
    "consumer": ["Data Science", "Analytics"],
    "source_system": "production_db.public.orders",
    "refresh_cadence": "Daily at 03:00 UTC",
    "last_validated": datetime.now().isoformat(),
    "review_cadence": "Quarterly or on schema change",
}
```

Display as a formatted Markdown table in the notebook.

### 2. Schema Definition

The core of the contract. For every column, define expectations:

| Column | Type | Nullable | Unique | PII | Description | Allowed Values / Range |
|---|---|---|---|---|---|---|
| `user_id` | `LongType` | No | Yes | No | Unique user identifier | > 0 |
| `email` | `StringType` | No | No | **Yes** | User email address | Valid email regex |
| `age` | `IntegerType` | Yes | No | No | User age in years | 13–120 |
| `signup_date` | `DateType` | No | No | No | Account creation date | 2015-01-01 to today |
| `category` | `StringType` | No | No | No | Product category | ["electronics", "clothing", "food", ...] |

Implement as executable assertions:

```python
def assert_schema(df, expected_schema):
    """Validate DataFrame schema against expected column names and types.

    Args:
        df: Spark DataFrame or pandas DataFrame
        expected_schema: dict of {column_name: expected_type_string}

    Returns:
        List of violations (empty if all pass)
    """
    violations = []
    actual_cols = {f.name: str(f.dataType) for f in df.schema.fields}

    for col_name, expected_type in expected_schema.items():
        if col_name not in actual_cols:
            violations.append(f"MISSING COLUMN: {col_name}")
        elif expected_type not in actual_cols[col_name]:
            violations.append(
                f"TYPE MISMATCH: {col_name} — expected {expected_type}, got {actual_cols[col_name]}"
            )

    unexpected = set(actual_cols.keys()) - set(expected_schema.keys())
    for col in unexpected:
        violations.append(f"UNEXPECTED COLUMN: {col}")

    return violations
```

### 3. Value Range Validation

For each column, validate that values fall within contracted ranges.

```python
def validate_ranges(df, range_spec):
    """Check that column values fall within expected ranges.

    Args:
        df: Spark DataFrame
        range_spec: dict of {column: {"min": v, "max": v}} for numeric,
                    or {column: {"allowed": [list]}} for categorical

    Returns:
        List of {column, check, violation_count, sample_violations}
    """
    results = []
    for col, spec in range_spec.items():
        if "min" in spec and "max" in spec:
            violations = df.filter(
                (F.col(col) < spec["min"]) | (F.col(col) > spec["max"])
            ).count()
            if violations > 0:
                samples = (
                    df.filter((F.col(col) < spec["min"]) | (F.col(col) > spec["max"]))
                    .select(col)
                    .limit(5)
                    .toPandas()[col]
                    .tolist()
                )
                results.append({
                    "column": col, "check": "range",
                    "violation_count": violations, "samples": samples,
                })
        elif "allowed" in spec:
            violations = df.filter(~F.col(col).isin(spec["allowed"])).count()
            if violations > 0:
                samples = (
                    df.filter(~F.col(col).isin(spec["allowed"]))
                    .select(col)
                    .distinct()
                    .limit(5)
                    .toPandas()[col]
                    .tolist()
                )
                results.append({
                    "column": col, "check": "allowed_values",
                    "violation_count": violations, "samples": samples,
                })
    return results
```

### 4. Freshness SLA

How recent must the data be? Stale data produces stale models.

```python
def check_freshness(df, timestamp_col, sla_hours):
    """Verify data freshness against SLA.

    Args:
        df: Spark DataFrame
        timestamp_col: name of the timestamp column
        sla_hours: maximum acceptable lag in hours

    Returns:
        dict with max_timestamp, lag_hours, sla_met (bool)
    """
    max_ts = df.agg(F.max(F.col(timestamp_col))).collect()[0][0]
    lag = (datetime.now() - max_ts).total_seconds() / 3600

    return {
        "max_timestamp": str(max_ts),
        "lag_hours": round(lag, 2),
        "sla_hours": sla_hours,
        "sla_met": lag <= sla_hours,
    }
```

If the data is stale beyond the SLA, flag it as a **critical violation**. Downstream playbooks should not proceed on stale data without explicit acknowledgment.

### 5. Completeness & Null Policy

Define a null policy per column: `"required"` (must not be null), `"nullable_with_default"` (nullable, has a documented default), `"nullable"` (null is acceptable).

```python
def check_nulls(df, null_policy):
    """Validate null rates against the contracted null policy.

    Args:
        df: Spark DataFrame
        null_policy: dict of {column: "required" | "nullable_with_default" | "nullable"}

    Returns:
        List of violations for required columns that have nulls
    """
    total = df.count()
    results = []
    for col, policy in null_policy.items():
        null_count = df.filter(F.col(col).isNull()).count()
        null_pct = (null_count / total * 100) if total > 0 else 0

        if policy == "required" and null_count > 0:
            results.append({
                "column": col, "policy": policy,
                "null_count": null_count, "null_pct": round(null_pct, 2),
                "status": "FAIL",
            })
        else:
            results.append({
                "column": col, "policy": policy,
                "null_count": null_count, "null_pct": round(null_pct, 2),
                "status": "PASS",
            })
    return results
```

### 6. Uniqueness & Key Integrity

Validate primary key uniqueness and, if applicable, foreign key referential integrity.

```python
def check_uniqueness(df, key_cols):
    """Verify that the specified columns form a unique key.

    Args:
        df: Spark DataFrame
        key_cols: list of column names that should be unique together

    Returns:
        dict with total_rows, distinct_keys, duplicates, is_unique (bool)
    """
    total = df.count()
    distinct = df.select(key_cols).distinct().count()
    duplicates = total - distinct

    return {
        "key_columns": key_cols,
        "total_rows": total,
        "distinct_keys": distinct,
        "duplicates": duplicates,
        "is_unique": duplicates == 0,
    }


def check_referential_integrity(child_df, parent_df, child_fk, parent_pk):
    """Verify foreign key references exist in the parent table.

    Args:
        child_df: DataFrame containing the foreign key
        parent_df: DataFrame containing the primary key
        child_fk: column name in child
        parent_pk: column name in parent

    Returns:
        dict with orphan_count, sample_orphans, integrity_met (bool)
    """
    orphans = child_df.join(
        parent_df.select(F.col(parent_pk).alias(child_fk)),
        on=child_fk, how="left_anti"
    )
    orphan_count = orphans.count()

    return {
        "child_fk": child_fk,
        "parent_pk": parent_pk,
        "orphan_count": orphan_count,
        "sample_orphans": orphans.select(child_fk).limit(5).toPandas()[child_fk].tolist() if orphan_count > 0 else [],
        "integrity_met": orphan_count == 0,
    }
```

### 7. PII Inventory & Handling

For every column flagged as PII in Section 2, document the handling decision:

| Column | PII Type | Handling | Justification |
|---|---|---|---|
| `email` | Direct identifier | Hash with SHA-256 + salt | Not needed for modeling; used only for join keys |
| `full_name` | Direct identifier | Drop before modeling | No analytical value |
| `ip_address` | Quasi-identifier | Mask to /24 subnet | Geographic signal needed, full IP unnecessary |
| `age` | Quasi-identifier | Retain | Required for segmentation; not identifying alone |

**Regulatory checklist:**
- [ ] GDPR applicability assessed (EU data subjects?)
- [ ] CCPA applicability assessed (California residents?)
- [ ] HIPAA applicability assessed (health data?)
- [ ] Data retention policy documented
- [ ] Right-to-deletion mechanism identified (if applicable)

Do **not** proceed to EDA with unaddressed PII. Every PII column must have an explicit handling decision.

### 8. Drift Tolerance

Establish baseline statistics for each column. These become the reference for drift monitoring in `07_INFERENCING.md`.

```python
def compute_baseline_stats(df, numeric_cols, categorical_cols):
    """Compute baseline statistics for drift monitoring.

    Args:
        df: Spark DataFrame (training or contract-time snapshot)
        numeric_cols: list of numeric column names
        categorical_cols: list of categorical column names

    Returns:
        dict of {column: {mean, std, min, max, quantiles}} for numeric,
              {column: {value_counts, cardinality}} for categorical
    """
    baseline = {}

    for col in numeric_cols:
        stats = df.select(
            F.mean(col).alias("mean"),
            F.stddev(col).alias("std"),
            F.min(col).alias("min"),
            F.max(col).alias("max"),
            F.percentile_approx(col, [0.25, 0.5, 0.75]).alias("quantiles"),
        ).collect()[0]
        baseline[col] = stats.asDict()

    for col in categorical_cols:
        vc = df.groupBy(col).count().toPandas().set_index(col)["count"].to_dict()
        baseline[col] = {"value_counts": vc, "cardinality": len(vc)}

    return baseline


def check_drift(df, baseline, numeric_cols, threshold_std=2.0):
    """Compare current data distribution against baseline.

    Args:
        df: Spark DataFrame (current batch)
        baseline: dict from compute_baseline_stats
        numeric_cols: columns to check
        threshold_std: number of std deviations to flag as drift

    Returns:
        List of drift alerts
    """
    alerts = []
    for col in numeric_cols:
        current_mean = df.select(F.mean(col)).collect()[0][0]
        base_mean = baseline[col]["mean"]
        base_std = baseline[col]["std"]

        if base_std and base_std > 0:
            z_score = abs(current_mean - base_mean) / base_std
            if z_score > threshold_std:
                alerts.append({
                    "column": col,
                    "baseline_mean": round(base_mean, 4),
                    "current_mean": round(current_mean, 4),
                    "z_score": round(z_score, 2),
                    "status": "DRIFT DETECTED",
                })
    return alerts
```

Define drift tolerance thresholds per column. Not all drift matters — a seasonal shift in `month` is expected; a shift in `user_age` distribution may signal a data pipeline issue.

### 9. Contract Violations Report

Aggregate all checks from Sections 2–8 into a single summary.

```python
def generate_contract_report(schema_results, range_results, freshness_result,
                              null_results, uniqueness_result, drift_alerts):
    """Aggregate all contract checks into a summary report.

    Returns:
        DataFrame with columns: Check, Scope, Status, Detail
    """
    rows = []

    # Schema
    if schema_results:
        for v in schema_results:
            rows.append({"Check": "Schema", "Scope": "All columns", "Status": "FAIL", "Detail": v})
    else:
        rows.append({"Check": "Schema", "Scope": "All columns", "Status": "PASS", "Detail": "All columns match"})

    # Ranges
    for r in range_results:
        rows.append({
            "Check": "Value Range", "Scope": r["column"],
            "Status": "FAIL", "Detail": f"{r['violation_count']} violations, samples: {r['samples']}",
        })

    # Freshness
    rows.append({
        "Check": "Freshness SLA", "Scope": "Dataset",
        "Status": "PASS" if freshness_result["sla_met"] else "FAIL",
        "Detail": f"Lag: {freshness_result['lag_hours']}h (SLA: {freshness_result['sla_hours']}h)",
    })

    # Nulls
    for n in null_results:
        if n["status"] == "FAIL":
            rows.append({
                "Check": "Null Policy", "Scope": n["column"],
                "Status": "FAIL", "Detail": f"{n['null_count']} nulls ({n['null_pct']}%) in required column",
            })

    # Uniqueness
    if not uniqueness_result["is_unique"]:
        rows.append({
            "Check": "Key Uniqueness", "Scope": str(uniqueness_result["key_columns"]),
            "Status": "FAIL", "Detail": f"{uniqueness_result['duplicates']} duplicate keys",
        })
    else:
        rows.append({
            "Check": "Key Uniqueness", "Scope": str(uniqueness_result["key_columns"]),
            "Status": "PASS", "Detail": "All keys unique",
        })

    # Drift
    for d in drift_alerts:
        rows.append({
            "Check": "Drift", "Scope": d["column"],
            "Status": "WARNING", "Detail": f"z={d['z_score']} (baseline={d['baseline_mean']}, current={d['current_mean']})",
        })

    return pd.DataFrame(rows)
```

**Decision rule:** If any check has Status = `FAIL`, the contract is violated. Downstream playbooks should not proceed until violations are resolved or explicitly accepted with documented justification.

### 10. MLflow Logging & Versioning

At the end of the notebook:

```python
import mlflow

with mlflow.start_run(run_name=f"contract_{CONTRACT_NAME}_v{CONTRACT_VERSION}"):
    # Log contract metadata
    mlflow.log_params(contract_meta)

    # Log validation report as artifact
    report_df.to_csv("/tmp/contract_report.csv", index=False)
    mlflow.log_artifact("/tmp/contract_report.csv")

    # Log baseline statistics as artifact (JSON)
    import json
    with open("/tmp/baseline_stats.json", "w") as f:
        json.dump(baseline_stats, f, default=str)
    mlflow.log_artifact("/tmp/baseline_stats.json")

    # Log summary metrics
    mlflow.log_metric("total_checks", len(report_df))
    mlflow.log_metric("checks_passed", len(report_df[report_df["Status"] == "PASS"]))
    mlflow.log_metric("checks_failed", len(report_df[report_df["Status"] == "FAIL"]))
    mlflow.log_metric("checks_warning", len(report_df[report_df["Status"] == "WARNING"]))
```

Tag the run with the contract version so future runs can compare against the same baseline.

---

## Code Standards

- Every section starts with a Markdown cell explaining *what* is being checked and *why*
- No magic numbers — define `FRESHNESS_SLA_HOURS`, `DRIFT_THRESHOLD_STD`, allowed value ranges in the contract spec table at the top
- Functions longer than ~15 lines go in `utils/contract_helpers.py`
- The notebook must run end-to-end with **Run All** without errors
- **Spark best practices:** avoid `.collect()` on large DataFrames — aggregate first, then collect small results
- Contract violations are **reported, not auto-fixed** — the notebook flags issues; humans decide how to resolve them

---

## Constraints & Guardrails

- Do **NOT** proceed to EDA if the data contract has critical violations — resolve or explicitly accept with documented justification
- Do **NOT** hardcode allowed value ranges in code — define them in the contract spec table (Section 2) and reference programmatically
- Do **NOT** treat PII columns casually — every PII column must have an explicit handling decision before data leaves this notebook
- Do **NOT** assume schema stability — the contract must be re-validated on every data refresh, not just the first load
- Do **NOT** silently coerce data types — a type mismatch is a contract violation, not an auto-fix opportunity
- Do **NOT** skip the freshness check — stale data produces stale models; flag it even if the pipeline "works"
- Do **NOT** define the contract in isolation — the data producer (Data Engineer) and data consumer (Data Analyst / ML Engineer) must both be represented

---

## Acceptance Criteria

| Criterion | Definition of Done |
|---|---|
| Schema validated | Every column checked for type, nullability, and uniqueness against the contract spec |
| Value ranges | Numeric ranges and categorical allowed values validated; violations counted and sampled |
| Freshness SLA | SLA check implemented with threshold; pass/fail status reported |
| Null policy | Per-column null policy defined and enforced; required columns with nulls flagged |
| PII flagged | All PII columns identified with handling decision documented; regulatory checklist addressed |
| Drift baseline | Baseline statistics computed and stored for future drift checks in `07_INFERENCING.md` |
| Contract report | Summary table with PASS/FAIL/WARNING per check; overall contract status stated |
| MLflow logged | Contract version, validation report, and baseline statistics logged as MLflow artifacts |

---

## Downstream Handoffs

| Recipient | What they receive | Playbook |
|---|---|---|
| Analytics Engineer | Validated schema, grain hints, PII handling decisions | `01b_DATA_MODELING.md` |
| Data Analyst | Clean, contracted data ready for exploration | `02_EDA.md` |
| ML Engineer | Baseline statistics for drift monitoring | `07_INFERENCING.md` |
| Data Scientist Reviewer | Contract report for methodology gate review | Any downstream playbook |
