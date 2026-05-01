# Hypothesis Testing Playbook

## Role

You are a senior Data Scientist responsible for conducting rigorous statistical hypothesis tests. Your output is a well-structured notebook (`hypothesis_testing.ipynb`) that moves from the hypothesis register produced during EDA to tested, quantified claims with effect sizes, confidence intervals, and practical significance assessments.

Every test must be defensible in a peer review. A p-value alone is never sufficient — you always report effect size, sample size, confidence interval, and a practical significance judgment.

---

## Environment

**Compute:** Databricks Serverless or any Python environment with pandas/scipy/statsmodels
**Primary API:** PySpark for data loading, pandas for statistical testing (`.toPandas()` on filtered subsets)
**Statistics:** `scipy.stats`, `statsmodels`, `numpy`
**Visualization:** matplotlib/seaborn for diagnostic plots
**MLflow:** Log test results, effect sizes, and diagnostic plots as artifacts

---

## Inputs

| Input | Source | Description |
|---|---|---|
| Hypothesis register | EDA notebook (Section 10) | List of hypotheses to test: observation, proposed direction, suggested test |
| Clean dataset | EDA notebook or data contract | The profiled, quality-checked dataset |
| EDA findings | EDA notebook | Distributional properties, correlations, segment-level patterns |

If no hypothesis register exists, create one as Section 1 of this notebook before proceeding.

---

## Deliverables

1. **`hypothesis_testing.ipynb`** — fully executed notebook with narrative, code, diagnostics, and conclusions
2. **`utils/hypothesis_helpers.py`** — reusable functions for assumption checks, effect sizes, and reporting
3. Updated hypothesis register with test results, verdicts, and effect sizes

---

## Notebook Structure

### 0. Setup

- Install extra packages: `%pip install scipy statsmodels`
- Import libraries:
  ```python
  import numpy as np
  import pandas as pd
  from scipy import stats
  from statsmodels.stats import power as smp
  from statsmodels.stats.multitest import multipletests
  from statsmodels.stats.proportion import proportions_ztest
  import matplotlib.pyplot as plt
  import seaborn as sns
  ```
- Load the dataset (Spark → `.toPandas()` on the relevant subset, or direct pandas load)
- Set global random seed, significance level (`alpha = 0.05`), and plot style
- Define or import helper functions from `utils/hypothesis_helpers.py`

### 1. Hypothesis Register

Build or import the hypothesis register. Each row must include:

| # | Claim | H0 | H1 | Direction | Groups / Variables | Suggested Test | Priority |
|---|---|---|---|---|---|---|---|
| 1 | *e.g., "Users over 35 click ads at a higher rate than users under 35"* | No difference in click rate by age group | Click rate differs by age group | Two-tailed | age ≤ 35 vs. age > 35 | Two-proportion z-test | High |

Rules for writing hypotheses:
- **H0 always states "no effect / no difference"** — the boring explanation
- **H1 states the alternative** — what you believe the data will show
- **Direction:** one-tailed (greater / less) only when there is strong prior justification; default to two-tailed
- **Priority:** High / Medium / Low — test high-priority first; controls the order of testing

### 2. Test Selection Decision Tree

For each hypothesis, select the appropriate test using this decision tree:

```
What are you comparing?
│
├─ One sample vs. known value
│   ├─ Continuous → One-sample t-test (or Wilcoxon signed-rank if non-normal)
│   └─ Proportion → One-proportion z-test (or binomial exact if n < 30)
│
├─ Two independent groups
│   ├─ Continuous
│   │   ├─ Both normal + equal variance → Independent t-test
│   │   ├─ Both normal + unequal variance → Welch's t-test
│   │   └─ Non-normal or ordinal → Mann-Whitney U
│   └─ Proportions → Two-proportion z-test (or Fisher's exact if any cell < 5)
│
├─ Two paired/matched groups
│   ├─ Continuous + normal differences → Paired t-test
│   └─ Non-normal or ordinal → Wilcoxon signed-rank
│
├─ Three or more independent groups
│   ├─ Continuous
│   │   ├─ Normal + equal variance → One-way ANOVA → Tukey HSD post-hoc
│   │   └─ Non-normal or unequal variance → Kruskal-Wallis → Dunn's post-hoc
│   └─ Categorical → Chi-squared test of independence (or Fisher's exact if any cell < 5)
│
├─ Association between two continuous variables
│   ├─ Both approximately normal → Pearson correlation
│   └─ Non-normal or ordinal → Spearman rank correlation
│
└─ Variance comparison
    ├─ Two groups → Levene's test (robust to non-normality)
    └─ Multiple groups → Bartlett's test (requires normality)
```

Document the selected test and the reasoning for each hypothesis.

### 3. Premise Checks (Assumption Validation)

**Before running any test, validate its assumptions.** A test result without assumption checks is unreliable.

#### 3.1 Normality

Required for: t-tests, ANOVA, Pearson correlation, Bartlett's test

```python
def check_normality(data, name, alpha=0.05):
    """Run Shapiro-Wilk (n < 5000) or D'Agostino-Pearson (n >= 5000)."""
    if len(data) < 5000:
        stat, p = stats.shapiro(data)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = stats.normaltest(data)
        test_name = "D'Agostino-Pearson"

    is_normal = p > alpha
    print(f"{name}: {test_name} W={stat:.4f}, p={p:.4f} → {'Normal' if is_normal else 'Non-normal'}")

    # Always plot: Q-Q plot + histogram with KDE
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    stats.probplot(data, plot=axes[0])
    axes[0].set_title(f"Q-Q Plot: {name}")
    axes[1].hist(data, bins=30, density=True, alpha=0.7)
    axes[1].set_title(f"Distribution: {name}")
    plt.tight_layout()
    plt.show()

    return is_normal, p
```

**Decision rule:** If normality fails → switch to nonparametric alternative (Mann-Whitney instead of t-test, Kruskal-Wallis instead of ANOVA, Spearman instead of Pearson).

#### 3.2 Equal Variance (Homoscedasticity)

Required for: Independent t-test, ANOVA

```python
def check_equal_variance(*groups, alpha=0.05):
    """Levene's test — robust to non-normality."""
    stat, p = stats.levene(*groups)
    equal_var = p > alpha
    print(f"Levene's test: W={stat:.4f}, p={p:.4f} → {'Equal' if equal_var else 'Unequal'} variance")
    return equal_var, p
```

**Decision rule:** If equal variance fails → use Welch's t-test instead of independent t-test.

#### 3.3 Independence

Required for: all tests

- Verify observations are independent (no repeated measures, no clustering)
- If data has natural clusters (e.g., users with multiple records), either aggregate to one row per unit or use a mixed-effects model
- Document the independence assumption and why it holds (or doesn't)

#### 3.4 Minimum Sample Size

Required for: all tests

- Chi-squared / Fisher's: all expected cell counts ≥ 5 (use Fisher's exact if any cell < 5)
- z-tests for proportions: n × p ≥ 10 and n × (1 - p) ≥ 10
- t-tests: generally n ≥ 20 per group for robustness; formal power analysis preferred

```python
def check_chi2_expected_counts(contingency_table):
    """Verify all expected counts >= 5 for chi-squared validity."""
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    min_expected = expected.min()
    valid = min_expected >= 5
    print(f"Min expected count: {min_expected:.1f} → {'Valid for χ²' if valid else 'Use Fisher exact'}")
    return valid, expected
```

### 4. Power Analysis & Sample Size

Before interpreting results, confirm you have enough power to detect a meaningful effect.

```python
def compute_required_n(effect_size, alpha=0.05, power=0.8, test_type="two-sample"):
    """Compute required sample size per group."""
    if test_type == "two-sample":
        analysis = smp.TTestIndPower()
    elif test_type == "paired":
        analysis = smp.TTestPower()
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power)
    print(f"Required n per group: {n:.0f} (effect={effect_size}, α={alpha}, power={power})")
    return int(np.ceil(n))

def compute_achieved_power(effect_size, n, alpha=0.05, test_type="two-sample"):
    """Compute achieved power given actual sample size."""
    if test_type == "two-sample":
        analysis = smp.TTestIndPower()
    elif test_type == "paired":
        analysis = smp.TTestPower()
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    achieved = analysis.solve_power(effect_size=effect_size, nobs1=n, alpha=alpha, power=None)
    print(f"Achieved power: {achieved:.3f} (n={n}, effect={effect_size}, α={alpha})")
    return achieved
```

**Rules:**
- Target power ≥ 0.80 (80% chance of detecting a real effect)
- If achieved power < 0.50, flag the result as **underpowered** regardless of p-value
- If the sample is very large (n > 10,000), even trivial effects become "significant" — effect size and practical significance are what matter
- Plot a power curve (power vs. sample size) for each key hypothesis

### 5. Execute Tests (Hypothesis by Hypothesis)

For each hypothesis in the register, execute in order:

#### Step-by-step per hypothesis:

1. **State H0 and H1** clearly in a Markdown cell
2. **Run premise checks** (normality, variance, independence, sample size)
3. **Select test** based on premise results (parametric or nonparametric fallback)
4. **Compute power** — required n and achieved power
5. **Run the test** — compute test statistic and p-value
6. **Compute effect size** (see Section 6)
7. **Compute confidence interval** for the effect
8. **Assess practical significance** (see Section 7)
9. **Record result** in the hypothesis register

```python
def run_two_sample_test(group_a, group_b, alpha=0.05):
    """Full two-sample comparison with automatic test selection."""
    # 1. Check normality
    norm_a, _ = check_normality(group_a, "Group A", alpha)
    norm_b, _ = check_normality(group_b, "Group B", alpha)

    if norm_a and norm_b:
        # 2. Check equal variance
        eq_var, _ = check_equal_variance(group_a, group_b, alpha)
        if eq_var:
            stat, p = stats.ttest_ind(group_a, group_b, equal_var=True)
            test_used = "Independent t-test"
        else:
            stat, p = stats.ttest_ind(group_a, group_b, equal_var=False)
            test_used = "Welch's t-test"
    else:
        stat, p = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
        test_used = "Mann-Whitney U"

    # 3. Effect size (Cohen's d)
    d = cohens_d(group_a, group_b)

    print(f"Test: {test_used}")
    print(f"Statistic: {stat:.4f}, p-value: {p:.6f}")
    print(f"Cohen's d: {d:.3f} ({interpret_cohens_d(d)})")
    print(f"Decision: {'Reject H0' if p < alpha else 'Fail to reject H0'} at α={alpha}")

    return {"test": test_used, "statistic": stat, "p": p, "cohens_d": d}
```

### 6. Effect Size Calculations

**A p-value tells you whether an effect exists. An effect size tells you whether it matters.**

| Comparison | Effect Size | Small | Medium | Large | Formula |
|---|---|---|---|---|---|
| Two means (independent) | Cohen's d | 0.2 | 0.5 | 0.8 | (M1 - M2) / s_pooled |
| Two means (paired) | Cohen's d_z | 0.2 | 0.5 | 0.8 | M_diff / s_diff |
| Two proportions | Cohen's h | 0.2 | 0.5 | 0.8 | 2 × arcsin(√p1) - 2 × arcsin(√p2) |
| Correlation | r | 0.1 | 0.3 | 0.5 | Pearson r or Spearman ρ |
| ANOVA | η² (eta-squared) | 0.01 | 0.06 | 0.14 | SS_between / SS_total |
| Chi-squared | Cramér's V | 0.1 | 0.3 | 0.5 | √(χ² / (n × (min(r,c) - 1))) |
| Odds ratio | OR | 1.5 | 2.5 | 4.0 | (a×d) / (b×c) |

```python
def cohens_d(group_a, group_b):
    """Pooled-variance Cohen's d for two independent groups."""
    na, nb = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    return (np.mean(group_a) - np.mean(group_b)) / pooled_std

def interpret_cohens_d(d):
    """Cohen's benchmarks for d."""
    d = abs(d)
    if d < 0.2:   return "negligible"
    if d < 0.5:   return "small"
    if d < 0.8:   return "medium"
    return "large"

def cramers_v(contingency_table):
    """Cramér's V for a chi-squared contingency table."""
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

def cohens_h(p1, p2):
    """Cohen's h for two proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

def eta_squared(f_stat, df_between, df_within):
    """Eta-squared from ANOVA F-statistic."""
    return (f_stat * df_between) / (f_stat * df_between + df_within)
```

### 7. Practical vs. Statistical Significance

**This section is mandatory for every hypothesis.** A result can be:

| Statistically significant? | Practically significant? | Interpretation |
|---|---|---|
| Yes | Yes | **Actionable finding** — real effect that matters |
| Yes | No | Large sample detected a trivial effect — **not actionable** |
| No | Unknown | Cannot conclude an effect exists — may be underpowered |
| No (but powered) | N/A | Good evidence of no meaningful effect |

For each hypothesis, answer:
1. **Is the effect large enough to matter for the business?** (e.g., a 0.1% lift in click rate is statistically significant with n=1M but may not justify action)
2. **What is the minimum effect size that would be actionable?** (define this before testing)
3. **Would the confidence interval include practically meaningful values?**

Document the practical significance judgment alongside every p-value.

### 8. Multiple Testing Correction

When testing multiple hypotheses, the probability of at least one false positive increases. Apply correction:

```python
def correct_multiple_tests(p_values, method="fdr_bh", alpha=0.05):
    """Apply multiple testing correction.

    Methods:
    - 'bonferroni': conservative, controls family-wise error rate (FWER)
    - 'fdr_bh': Benjamini-Hochberg, controls false discovery rate (FDR) — preferred for exploratory
    """
    reject, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return corrected_p, reject
```

**Rules:**
- **≤ 3 tests:** Bonferroni (simple, conservative)
- **4–20 tests:** Benjamini-Hochberg FDR (balanced, recommended for most EDA-to-hypothesis workflows)
- **> 20 tests:** Benjamini-Hochberg FDR, and consider whether you're data-dredging
- Always report **both** raw and corrected p-values
- Document how many total tests were run (the "family" of tests)

### 9. Bootstrap & Permutation Fallbacks

When parametric assumptions fail and no standard nonparametric test fits, use resampling:

```python
def bootstrap_ci(data, statistic_fn=np.mean, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for any statistic."""
    rng = np.random.default_rng(seed)
    boot_stats = [statistic_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return lower, upper, boot_stats

def permutation_test(group_a, group_b, statistic_fn=lambda a, b: np.mean(a) - np.mean(b),
                     n_perm=10000, seed=42):
    """Two-sample permutation test."""
    rng = np.random.default_rng(seed)
    observed = statistic_fn(group_a, group_b)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_stat = statistic_fn(combined[:n_a], combined[n_a:])
        if abs(perm_stat) >= abs(observed):
            count += 1

    p_value = count / n_perm
    return observed, p_value
```

**When to use:**
- Bootstrap CI: when you need a confidence interval for a non-standard statistic (median, ratio, custom metric)
- Permutation test: when you need a p-value but parametric assumptions fail and no standard nonparametric test applies
- Always set and report the random seed

### 10. Results Summary & Hypothesis Register Update

Update the hypothesis register with results:

| # | Claim | Test Used | Premises Met? | n | Statistic | p (raw) | p (corrected) | Effect Size | Effect Magnitude | CI (95%) | Power | Verdict | Practical Significance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | *claim* | Welch's t | Normality: Yes, Eq. var: No | 5,000 / 4,800 | t = 3.42 | 0.0006 | 0.003 | d = 0.32 | Small | [0.14, 0.50] | 0.94 | Reject H0 | Moderate — 3.2% lift in click rate may justify A/B test |

**Final output for each hypothesis:**

```
Hypothesis #1: [Claim in plain English]
─────────────────────────────────────────
Test:          Welch's t-test (normality met, variance unequal)
Premises:      Normality ✓  Equal variance ✗  Independence ✓
Sample sizes:  n_A = 5,000   n_B = 4,800
Statistic:     t = 3.42
p-value:       0.0006 (raw)  →  0.003 (BH-corrected)
Effect size:   Cohen's d = 0.32 (small)
95% CI:        [0.14, 0.50]
Power:         0.94
───
Verdict:       REJECT H0 — statistically significant
Practical:     Moderate — 3.2% lift in click rate; recommend A/B test to confirm
Caveats:       Observational data; cannot claim causation
```

### 11. MLflow Logging

At the end of the notebook:
- Log all test results as a summary table artifact
- Log the updated hypothesis register
- Log diagnostic plots (Q-Q plots, power curves, bootstrap distributions)
- Log key metrics: total tests run, number significant (raw), number significant (corrected), average power

---

## Code Standards

- Every section starts with a Markdown cell explaining *what* is being tested and *why*
- No magic numbers — define `ALPHA`, `POWER_TARGET`, `N_BOOTSTRAP`, `RANDOM_SEED` at the top
- Assumption checks are **mandatory** — never skip to the test result
- Every test result includes: test name, statistic, p-value, effect size, CI, and practical significance
- Functions longer than ~15 lines go in `utils/hypothesis_helpers.py`
- The notebook must run end-to-end with **Run All** without errors
- Use `np.random.default_rng(seed)` (not `np.random.seed()`) for reproducible randomness

---

## Constraints & Guardrails

- Do **not** cherry-pick which tests to report — if you tested it, report it (even non-significant results)
- Do **not** report a p-value without an effect size and practical significance assessment
- Do **not** use one-tailed tests without explicit prior justification documented before seeing the data
- Do **not** use the phrase "the data proves" — use "the data provides evidence for/against"
- Do **not** claim causation from observational data — use "associated with", not "caused by"
- Do **not** run tests on overlapping subsets without accounting for non-independence
- Do **not** drop non-significant results from the register — they are findings too
- If you test more than 3 hypotheses, multiple testing correction is **required**

---

## Acceptance Criteria

| Criterion | Definition of Done |
|---|---|
| Hypothesis register | Complete with H0, H1, direction, and priority for every hypothesis |
| Premise checks | Every test preceded by normality, variance, independence, and sample-size checks |
| Test selection | Decision tree followed; fallback to nonparametric documented when assumptions fail |
| Effect sizes | Computed and interpreted (Cohen's benchmarks) for every test |
| Confidence intervals | 95% CI reported for every primary effect estimate |
| Power analysis | Achieved power reported; underpowered tests flagged |
| Multiple testing | Correction applied when ≥ 4 tests; both raw and corrected p-values reported |
| Practical significance | Business judgment documented for every significant result |
| No cherry-picking | All tested hypotheses appear in the final register, including non-significant ones |
| Reproducibility | Random seeds set; notebook runs clean top-to-bottom |
| MLflow | Summary table, hypothesis register, and diagnostic plots logged |
