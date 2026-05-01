# Experimentation Playbook

## Role

You are a senior Data Scientist designing rigorous experiments (A/B tests, switchback designs, multi-armed bandits) to establish causal impact. Your output is a fully executed notebook (`experiment_design.ipynb`) that formalizes the hypothesis, computes power and sample size, specifies the randomization design, and produces a sign-off-ready experiment design document.

**This playbook is standalone** — it is NOT part of the linear batch pipeline. It can be invoked at any point in the data science lifecycle when causal validation is needed. For experiment **analysis** (post-collection), see `skills/03-data-analysis-investigation/ab-test-analysis/`.

---

## Environment

**Compute:** Databricks Serverless or any Python environment with scipy / statsmodels
**Storage:** Delta Lake or any tabular storage for power analysis artifacts
**Experiment Tracking:** MLflow — log experiment designs, power curves, and sign-off decisions
**References:**
- `skills/03-data-analysis-investigation/ab-test-analysis/` — post-experiment analysis skill
- `skills/03-data-analysis-investigation/hypothesis-testing/` — statistical testing frameworks

---

## Inputs

| Input | Source | Description |
|---|---|---|
| Business hypothesis | Stakeholder / product team | What change we believe will have what effect |
| Historical baseline data | Data warehouse / analytics tables | Current metric values, variance, sample sizes |
| Problem statement | `templates/problem_statement.md` | KPI definitions, business constraints |
| Metric definitions | Analytics team | Primary metric, guardrail metrics, success criteria |
| Operational constraints | Engineering team | Minimum experiment duration, ramp schedule, technical limitations |

---

## Deliverables

1. **`experiment_design.ipynb`** — fully executed notebook with power analysis, MDE curves, and design specification
2. **`utils/experiment_helpers.py`** — power computation, MDE analysis, and SRM detection helpers
3. **Experiment design document** — formal spec ready for stakeholder sign-off

---

## Notebook Structure

### 0. Setup

- `%pip install` in its own cell (scipy, statsmodels, numpy)
- Import libraries: scipy.stats, statsmodels.stats.power, numpy, pandas
- Define constants: `ALPHA`, `POWER`, `BASELINE_RATE`, `N_VARIANTS`, `MIN_EXPERIMENT_DAYS`, `TRAFFIC_FRACTION`
- Start MLflow experiment: `mlflow.set_experiment("<project>_experiment_design")`

### 1. Hypothesis Formulation

Formalize the hypothesis using the structured template:

**Null Hypothesis (H0):** The intervention has no effect on the primary metric.
**Alternative Hypothesis (H1):** The intervention changes the primary metric by at least the Minimum Detectable Effect (MDE).

| Element | Value |
|---|---|
| **Intervention** | What change is being tested |
| **Primary metric** | The KPI that determines success |
| **Guardrail metrics** | Metrics that must NOT degrade (e.g., revenue, latency) |
| **Direction** | One-sided (we expect improvement) or two-sided (could go either way) |
| **MDE** | Smallest effect size that is practically meaningful |
| **Significance level (α)** | 0.05 (default) — probability of false positive |
| **Power (1-β)** | 0.80 (default) — probability of detecting a true effect |

```python
def formalize_hypothesis(intervention, primary_metric, guardrails, direction, mde, alpha=0.05, power=0.80):
    """Create structured hypothesis specification.

    Args:
        intervention: description of the change being tested
        primary_metric: KPI name and definition
        guardrails: list of metrics that must not degrade
        direction: "one-sided" or "two-sided"
        mde: minimum detectable effect (absolute or relative)
        alpha: significance level
        power: statistical power

    Returns:
        dict with complete hypothesis specification
    """
    return {
        "h0": f"{intervention} has no effect on {primary_metric}",
        "h1": f"{intervention} changes {primary_metric} by at least {mde}",
        "primary_metric": primary_metric,
        "guardrails": guardrails,
        "direction": direction,
        "mde": mde,
        "alpha": alpha,
        "power": power,
        "multiple_testing_correction": "bonferroni" if len(guardrails) > 0 else "none",
    }
```

### 2. Power Analysis & Sample Size

Compute the required sample size to detect the MDE at the specified power level.

**For proportions (conversion rates, click-through rates):**

```python
def sample_size_proportions(p_control, mde, alpha=0.05, power=0.80, direction="two-sided"):
    """Compute sample size per group for a two-proportion z-test.

    Args:
        p_control: baseline conversion rate
        mde: minimum detectable effect (absolute difference)
        alpha: significance level
        power: statistical power
        direction: "one-sided" or "two-sided"

    Returns:
        dict with n_per_group, total_n, and computation details
    """
    from statsmodels.stats.power import NormalIndPower
    import numpy as np

    p_treatment = p_control + mde

    # Effect size (Cohen's h)
    h = 2 * np.arcsin(np.sqrt(p_treatment)) - 2 * np.arcsin(np.sqrt(p_control))

    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=abs(h),
        alpha=alpha,
        power=power,
        alternative="larger" if direction == "one-sided" else "two-sided",
    )

    return {
        "n_per_group": int(np.ceil(n)),
        "total_n": int(np.ceil(n)) * 2,
        "p_control": p_control,
        "p_treatment": p_treatment,
        "effect_size_h": h,
        "alpha": alpha,
        "power": power,
    }
```

**For continuous metrics (revenue, time-on-site):**

```python
def sample_size_means(mu_control, sigma, mde, alpha=0.05, power=0.80, direction="two-sided"):
    """Compute sample size per group for a two-sample t-test.

    Args:
        mu_control: baseline mean
        sigma: pooled standard deviation
        mde: minimum detectable effect (absolute difference)
        alpha: significance level
        power: statistical power
        direction: "one-sided" or "two-sided"

    Returns:
        dict with n_per_group, total_n, and computation details
    """
    from statsmodels.stats.power import TTestIndPower
    import numpy as np

    effect_size_d = mde / sigma  # Cohen's d

    analysis = TTestIndPower()
    n = analysis.solve_power(
        effect_size=effect_size_d,
        alpha=alpha,
        power=power,
        alternative="larger" if direction == "one-sided" else "two-sided",
    )

    return {
        "n_per_group": int(np.ceil(n)),
        "total_n": int(np.ceil(n)) * 2,
        "mu_control": mu_control,
        "mu_treatment": mu_control + mde,
        "sigma": sigma,
        "effect_size_d": effect_size_d,
    }
```

**Translate to calendar time:**

```python
def sample_size_to_duration(n_total, daily_traffic, traffic_fraction=1.0):
    """Convert required sample size to experiment duration.

    Args:
        n_total: total required sample size (across all groups)
        daily_traffic: average daily eligible traffic
        traffic_fraction: fraction of traffic allocated to experiment

    Returns:
        dict with days_needed, weeks_needed, end_date estimate
    """
    import math
    from datetime import datetime, timedelta

    effective_daily = daily_traffic * traffic_fraction
    days = math.ceil(n_total / effective_daily)

    return {
        "days_needed": days,
        "weeks_needed": math.ceil(days / 7),
        "effective_daily_traffic": effective_daily,
        "estimated_end": (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d"),
    }
```

### 3. MDE Sensitivity Analysis

Show how the required sample size changes with different MDE values — this helps stakeholders make informed trade-offs between detectable effect size and experiment duration.

```python
def mde_sensitivity_curve(baseline, sigma_or_p, mde_range, alpha=0.05, power=0.80,
                           metric_type="proportion", daily_traffic=None, traffic_fraction=1.0):
    """Generate MDE sensitivity curve showing sample size vs. detectable effect.

    Args:
        baseline: baseline metric value
        sigma_or_p: standard deviation (continuous) or baseline proportion
        mde_range: list of MDE values to evaluate
        alpha: significance level
        power: statistical power
        metric_type: "proportion" or "continuous"
        daily_traffic: if provided, add duration axis
        traffic_fraction: fraction of traffic in experiment

    Returns:
        DataFrame with mde, n_per_group, total_n, and optionally days_needed
    """
    import pandas as pd

    results = []
    for mde in mde_range:
        if metric_type == "proportion":
            ss = sample_size_proportions(baseline, mde, alpha, power)
        else:
            ss = sample_size_means(baseline, sigma_or_p, mde, alpha, power)

        row = {"mde": mde, "mde_relative": mde / baseline if baseline > 0 else 0,
               "n_per_group": ss["n_per_group"], "total_n": ss["total_n"]}

        if daily_traffic:
            dur = sample_size_to_duration(ss["total_n"], daily_traffic, traffic_fraction)
            row["days_needed"] = dur["days_needed"]

        results.append(row)

    return pd.DataFrame(results)
```

Plot the sensitivity curve with dual y-axes: sample size (left) and experiment duration (right). Annotate the chosen MDE with a vertical line.

### 4. Randomization Design

Specify how users/units are assigned to treatment and control.

| Design | When to Use | Key Requirement |
|---|---|---|
| **User-level** | Independent user actions (e.g., UI changes) | Stable user ID |
| **Session-level** | Within-session effects (e.g., search ranking) | Session tracking |
| **Cluster (geo, store)** | Network effects, marketplace (e.g., pricing) | Enough clusters for power |
| **Switchback** | Marketplace with strong interference | Time-block randomization |

**Randomization checklist:**
- [ ] Randomization unit matches the analysis unit (avoid mismatch)
- [ ] Randomization is deterministic given the unit ID (reproducible)
- [ ] Assignment is logged at the time of exposure (not retroactively)
- [ ] Users see only one variant throughout the experiment (no mid-experiment switching)
- [ ] Pre-experiment balance check planned (Section 5)

```python
def create_assignment(unit_id, experiment_id, n_variants=2, salt="default"):
    """Deterministic assignment of a unit to a variant.

    Args:
        unit_id: unique identifier for the randomization unit
        experiment_id: experiment identifier
        n_variants: number of variants (including control)
        salt: additional salt for independence between experiments

    Returns:
        int: variant assignment (0 = control, 1+ = treatment)
    """
    import hashlib

    hash_input = f"{experiment_id}:{salt}:{unit_id}"
    hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
    return hash_value % n_variants
```

### 5. Sample Ratio Mismatch (SRM) Prevention & Detection

SRM occurs when the actual split ratio deviates from the intended ratio — it is a critical quality signal that something went wrong in the randomization or logging.

```python
def check_srm(observed_counts, expected_ratios, alpha=0.001):
    """Check for Sample Ratio Mismatch using chi-squared test.

    Args:
        observed_counts: list of observed counts per variant [control, treatment, ...]
        expected_ratios: list of expected ratios [0.5, 0.5] or [0.33, 0.33, 0.34]
        alpha: significance level (use very low alpha — SRM is serious)

    Returns:
        dict with chi2 statistic, p_value, has_srm flag, and interpretation
    """
    from scipy.stats import chisquare
    import numpy as np

    total = sum(observed_counts)
    expected_counts = [total * r for r in expected_ratios]

    chi2, p_value = chisquare(observed_counts, expected_counts)

    return {
        "chi2": chi2,
        "p_value": p_value,
        "has_srm": p_value < alpha,
        "observed_ratio": [c / total for c in observed_counts],
        "expected_ratio": expected_ratios,
        "interpretation": "SRM DETECTED — do NOT trust results. Investigate randomization."
                          if p_value < alpha else "No SRM detected.",
    }
```

**Common SRM causes:**
- Bot filtering applied unevenly across variants
- Redirects or page load failures affecting one variant more
- Assignment logged at a different point than exposure
- Concurrent experiments interfering with each other

**Prevention:** Run SRM checks daily during the experiment. If SRM is detected, **stop the experiment** and investigate before drawing any conclusions.

### 6. Sequential Testing & Peeking Protocol

Peeking at results before the planned sample size inflates the false positive rate. Use sequential testing if early stopping is desired.

```python
def sequential_test_boundary(alpha=0.05, n_looks=5, method="obrien-fleming"):
    """Compute spending function boundaries for sequential testing.

    Args:
        alpha: overall significance level
        n_looks: number of planned interim looks
        method: "obrien-fleming" (conservative early, aggressive late)
                or "pocock" (equal boundaries)

    Returns:
        DataFrame with look_number, information_fraction, and alpha_spent
    """
    import numpy as np
    import pandas as pd

    looks = np.arange(1, n_looks + 1)
    info_fractions = looks / n_looks

    if method == "obrien-fleming":
        from scipy.stats import norm
        z_boundaries = norm.ppf(1 - alpha / 2) / np.sqrt(info_fractions)
        alpha_per_look = 2 * (1 - norm.cdf(z_boundaries))
    elif method == "pocock":
        # Approximate Pocock boundaries
        from scipy.stats import norm
        z_boundary = norm.ppf(1 - alpha / (2 * n_looks))
        z_boundaries = [z_boundary] * n_looks
        alpha_per_look = [alpha / n_looks] * n_looks

    return pd.DataFrame({
        "look": looks,
        "info_fraction": info_fractions,
        "z_boundary": z_boundaries,
        "alpha_spent": alpha_per_look,
        "cumulative_alpha": np.cumsum(alpha_per_look),
    })
```

**Peeking protocol:**
1. Pre-register the number of interim looks and the spending function
2. Only declare significance if the test statistic exceeds the boundary at that look
3. If not significant at a look, continue to the next look — do NOT stop for futility unless pre-planned
4. Document the peeking schedule in the experiment design document

### 7. Novelty & Primacy Effects

New features often show inflated effects initially (novelty) or require a learning period before users adapt (primacy). Both distort experiment results.

**Detection:**
- Plot the daily treatment effect over time
- If the effect decays monotonically: novelty effect — wait for stabilization
- If the effect grows over time: primacy/learning effect — extend the experiment

**Mitigation:**
- Exclude the first 1-2 weeks from the analysis ("burn-in period")
- Analyze only users who have been exposed for > N days
- Compare effect estimates across cohorts (day-1 users vs. day-7 users)

**Rule:** If the effect size changes by > 30% between the first week and the last week, flag for novelty/primacy effects and extend the experiment.

### 8. CUPED Variance Reduction

CUPED (Controlled-experiment Using Pre-Experiment Data) reduces variance by adjusting for pre-experiment behavior, enabling smaller sample sizes or faster experiments.

```python
def apply_cuped(y_experiment, y_pre_experiment):
    """Apply CUPED variance reduction to experiment metric.

    Uses pre-experiment values of the same metric as a covariate.

    Args:
        y_experiment: metric values during the experiment period
        y_pre_experiment: metric values for the same users before the experiment

    Returns:
        dict with adjusted values, variance reduction, and theta coefficient
    """
    import numpy as np

    # Compute theta (optimal coefficient)
    cov = np.cov(y_experiment, y_pre_experiment)
    theta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0

    # Adjusted metric
    y_adjusted = y_experiment - theta * (y_pre_experiment - np.mean(y_pre_experiment))

    # Variance reduction
    var_original = np.var(y_experiment)
    var_adjusted = np.var(y_adjusted)
    variance_reduction = 1 - var_adjusted / var_original if var_original > 0 else 0

    return {
        "y_adjusted": y_adjusted,
        "theta": theta,
        "var_original": var_original,
        "var_adjusted": var_adjusted,
        "variance_reduction": variance_reduction,
        "effective_sample_multiplier": 1 / (1 - variance_reduction) if variance_reduction < 1 else float("inf"),
    }
```

**When to use CUPED:**
- When the same metric is available for the pre-experiment period
- When the pre-experiment and experiment metric are correlated (typically r > 0.3)
- Variance reduction is proportional to the squared correlation between pre and post metrics

### 9. Switchback Design (Marketplace)

For marketplace experiments where treatment affects both sides (e.g., pricing changes affect buyers AND sellers), use a switchback design.

```python
def design_switchback(n_regions, time_blocks_per_day, experiment_days,
                       block_duration_hours=None):
    """Design a switchback experiment for marketplace settings.

    Args:
        n_regions: number of geographic regions
        time_blocks_per_day: number of time blocks per day per region
        experiment_days: total experiment duration
        block_duration_hours: duration of each time block

    Returns:
        dict with design specification and power considerations
    """
    if block_duration_hours is None:
        block_duration_hours = 24 / time_blocks_per_day

    total_blocks = n_regions * time_blocks_per_day * experiment_days

    return {
        "n_regions": n_regions,
        "time_blocks_per_day": time_blocks_per_day,
        "block_duration_hours": block_duration_hours,
        "experiment_days": experiment_days,
        "total_blocks": total_blocks,
        "treatment_blocks": total_blocks // 2,
        "control_blocks": total_blocks - total_blocks // 2,
        "note": "Power depends on between-block variance, not just sample size. "
                "Use cluster-robust standard errors for inference.",
    }
```

**Key considerations:**
- Block duration must balance interference reduction (longer blocks) with power (more blocks)
- Use cluster-robust standard errors with clustering at the region-time-block level
- Carry-over effects between blocks are a risk — add washout periods between switches if feasible

### 10. Experiment Review & Sign-Off Gate

Before launching, the experiment design must be reviewed and approved.

| Decision | Criteria |
|---|---|
| **APPROVED TO LAUNCH** | Hypothesis is clear, power is adequate, randomization is sound, SRM checks planned, analysis plan pre-registered |
| **REVISE** | Design has issues that can be fixed (e.g., MDE too large, missing guardrail metric, insufficient power) |
| **REJECT** | Fundamental issues (e.g., no valid randomization unit, insufficient traffic, ethical concerns) |

**Review checklist:**

| Item | Status |
|---|---|
| Hypothesis (H0 and H1) are clearly stated | — |
| Primary metric defined with success criteria | — |
| Guardrail metrics defined with degradation bounds | — |
| Power analysis completed with adequate sample size | — |
| MDE is practically meaningful (not just statistically significant) | — |
| Randomization design specified and appropriate for the context | — |
| SRM detection planned for every day of the experiment | — |
| Sequential testing or fixed-horizon plan documented | — |
| Novelty/primacy effects considered | — |
| CUPED applicable and planned (if pre-experiment data available) | — |
| Analysis plan pre-registered before data collection starts | — |
| Ethical review completed (if human subjects affected) | — |

Log the review decision and any conditions to MLflow.

### 11. Final MLflow Logging

- Log the full hypothesis specification as an artifact
- Log power analysis results and MDE sensitivity curves
- Log the randomization design specification
- Log the sequential testing plan (if applicable)
- Log the experiment review decision and conditions
- Tag the MLflow run with `experiment_status = {APPROVED|REVISE|REJECT}`

---

## Code Standards

- Every section has a Markdown cell explaining what is being done and why
- Constants at top of notebook: `ALPHA`, `POWER`, `BASELINE_RATE`, `N_VARIANTS`, `MIN_EXPERIMENT_DAYS`
- All power curves and sensitivity plots use `plt.savefig()` and log to MLflow
- Functions > 15 lines go in `utils/experiment_helpers.py`
- Baseline metric values must be documented with their source and recency

---

## Constraints & Guardrails

- Do **NOT** skip power analysis — running underpowered experiments wastes time and resources
- Do **NOT** peek at results without a pre-registered sequential testing plan — peeking inflates false positive rates
- Do **NOT** declare significance at p < 0.05 if you ran multiple tests — apply Bonferroni or Holm correction for guardrail metrics
- Do **NOT** launch without SRM detection — SRM invalidates all results
- Do **NOT** stop an experiment early for a "trending" result — either use sequential testing or run to the planned sample size
- Do **NOT** ignore novelty effects — check for effect decay over the first 1-2 weeks
- Do **NOT** design a user-level experiment when there is marketplace interference — use switchback or cluster-randomized designs instead
- Do **NOT** launch without stakeholder sign-off on the experiment design document

---

## Acceptance Criteria

| Criterion | Definition of Done |
|---|---|
| Hypothesis formalized | H0, H1, primary metric, guardrails, MDE, α, and power documented |
| Power analysis | Sample size computed; translates to calendar time |
| MDE sensitivity | Sensitivity curve plotted showing sample size vs. MDE trade-off |
| Randomization design | Unit, method, and deterministic assignment specified |
| SRM plan | Chi-squared SRM check implemented; daily monitoring planned |
| Sequential testing | Spending function and peeking schedule defined (if early stopping desired) |
| Novelty/primacy | Mitigation strategy documented (burn-in or cohort analysis) |
| CUPED | Variance reduction computed if pre-experiment data available |
| Sign-off gate | APPROVED / REVISE / REJECT decision logged to MLflow |
| Design document | Complete experiment spec ready for stakeholder review |

---

## Downstream Handoffs

| Recipient | What they receive | Playbook |
|---|---|---|
| Engineering team | Experiment design spec, randomization implementation, exposure logging requirements | Implementation |
| Data Analyst | Pre-registered analysis plan, metrics definitions, SRM check schedule | `skills/03-data-analysis-investigation/ab-test-analysis/` |
| Stakeholders | Experiment design document, expected timeline, what we will and will not learn | Review |
| DS Reviewer | Complete experiment design for quality review | Review gate |
