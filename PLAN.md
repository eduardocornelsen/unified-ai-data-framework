# `data-skills` — Repo Plan

> Status: planning
> Owner: Eduardo Cornelsen
> Last updated: 2026-04-30

## 1. Purpose

`data-skills` is a **standalone, plug-and-play submodule** that turns a Claude Code agent into a senior data science team. It contains:

- **Playbooks** — step-by-step instructions for each phase of a DS project (EDA, hypothesis testing, feature engineering, training, evaluation, inferencing, monitoring, experimentation).
- **Personas** — role-specific system prompts (data analyst, ML engineer, reviewer, etc.) that set tone, focus, and gates.
- **Premises** — reusable checklists of statistical assumptions for tests and models.
- **Checklists** — short "real DS would not skip" guards (leakage, multiple-testing, reproducibility).
- **Templates** — copy-paste scaffolds for problem statements, model cards, experiment designs.
- **Reference implementation** — the ad-click pipeline as a working end-to-end example.

Both playbooks and personas are exposed as **Claude Code skills**, so any project that includes this submodule can invoke them via `Skill(...)`.

---

## 2. Decisions (locked)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Independent submodule** | Versioned separately, reusable across projects. Has its own `.gitignore`, `README.md`, `LICENSE`. No `pyproject.toml` (it's prompts + markdown, not a Python package). |
| 2 | **Generic-by-default, with domain accelerators** | Each playbook is dataset-agnostic. Common business problems (ad-click, churn, forecasting, fraud) get **deeper guided instructions** in `domain_accelerators/`. |
| 3 | **No skeleton-notebook folder** | Skeletons rot. One full **reference implementation** under `examples/reference_implementations/ad_click/`. Inline code recipes live inside playbooks. |
| 4 | **No 9th persona for hypothesis testing** | Inferential rigor is every senior DS's job. Instead: `data_scientist_reviewer` owns the *gate*; `data_analyst` co-authors hypotheses; the **playbook** teaches the method. |
| 5 | **Both playbooks and personas exposed as Claude Code skills** | Skills are the execution surface. Each playbook = `Skill(eda)`, each persona = `Skill(persona-data-analyst)`. Project-local under `.claude/skills/`, ships with the submodule. |
| 6 | **Add `analytics_engineer` persona** (9th persona, distinct from hypothesis testing) | Real, distinct role at top data orgs (popularized by dbt Labs). Owns the modeled-mart / semantic layer / data-test layer that our current `data_engineer` (infra/Spark/Airflow) does not. Without it, every DS playbook assumes clean modeled data exists — that assumption needs an owner. |

---

## 3. Final Repo Structure

```
data-skills/
├── README.md                              # what this is, quickstart, how to add as submodule
├── PLAN.md                                # this file
├── CLAUDE.md                              # default instructions when this dir is a project root
├── LICENSE
├── .gitignore
│
├── .claude/
│   └── skills/                            # makes everything below invocable as Claude Code skills
│       ├── playbooks/
│       │   ├── problem-framing/SKILL.md
│       │   ├── data-contract/SKILL.md
│       │   ├── data-modeling/SKILL.md
│       │   ├── eda/SKILL.md
│       │   ├── hypothesis-testing/SKILL.md
│       │   ├── feature-engineering/SKILL.md
│       │   ├── model-training/SKILL.md
│       │   ├── model-evaluation/SKILL.md
│       │   ├── inferencing/SKILL.md
│       │   ├── monitoring/SKILL.md
│       │   └── experimentation/SKILL.md
│       └── personas/
│           ├── persona-data-scientist-reviewer/SKILL.md
│           ├── persona-data-analyst/SKILL.md
│           ├── persona-data-engineer/SKILL.md
│           ├── persona-analytics-engineer/SKILL.md
│           ├── persona-ml-engineer/SKILL.md
│           ├── persona-product-manager/SKILL.md
│           ├── persona-ux-researcher/SKILL.md
│           ├── persona-frontend-developer/SKILL.md
│           └── persona-qa-tester/SKILL.md
│
├── playbooks/                             # canonical content — SKILL.md files reference these
│   ├── 00_PROBLEM_FRAMING.md             # NEW
│   ├── 01_DATA_CONTRACT.md               # NEW
│   ├── 01b_DATA_MODELING.md              # NEW — owned by analytics_engineer; dimensional modeling, SCD, conformed dims
│   ├── 02_EDA.md                         # existing, expanded + de-domained
│   ├── 03_HYPOTHESIS_TESTING.md          # NEW — the big gap
│   ├── 04_FEATURE_ENGINEERING.md         # existing, expanded + de-domained
│   ├── 05_MODEL_TRAINING.md              # existing, expanded + de-domained
│   ├── 06_MODEL_EVALUATION.md            # NEW — calibration, fairness, error analysis
│   ├── 07_INFERENCING.md                 # existing, expanded + de-domained
│   ├── 08_MONITORING.md                  # NEW — drift, decay, retraining triggers
│   └── 09_EXPERIMENTATION.md             # NEW — A/B design, power, MDE, CUPED
│
├── personas/                              # canonical content
│   ├── _template.md                      # the architet.md meta-template
│   ├── data_scientist_reviewer.md        # gains explicit "statistical rigor gatekeeper" mandate
│   ├── data_analyst.md                   # gains "hypothesis author" responsibility
│   ├── data_engineer.md                  # narrowed to ingestion/infra/DAGs (hands modeling to AE)
│   ├── analytics_engineer.md             # NEW — semantic layer, dbt-style models, marts, data tests
│   ├── ml_engineer.md
│   ├── product_manager.md
│   ├── ux_researcher.md
│   ├── frontend_developer.md
│   └── qa_tester.md
│
├── premises/                              # reusable assumption checklists
│   ├── parametric_tests.md               # normality, independence, equal variance, n>30
│   ├── regression.md                     # linearity, multicollinearity, homoscedasticity, residuals
│   ├── classification.md                 # balance, separation, calibration, cost asymmetry
│   ├── time_series.md                    # stationarity, autocorrelation, seasonality
│   └── causal_inference.md               # SUTVA, ignorability, overlap, exchangeability
│
├── checklists/                            # short non-negotiables
│   ├── data_leakage.md
│   ├── train_test_contamination.md
│   ├── data_quality_tests.md             # NEW — uniqueness, not_null, accepted_values, freshness, ref. integrity (AE)
│   ├── multiple_testing.md               # Bonferroni, BH-FDR, family-wise vs FDR
│   ├── effect_size_reporting.md          # Cohen's d, η², φ, OR, Cramér's V
│   ├── reproducibility.md                # seeds, env, data versioning, MLflow
│   └── stakeholder_handoff.md
│
├── templates/                             # copy-paste artifacts
│   ├── problem_statement.md              # CRISP-DM-style framing
│   ├── experiment_design.md              # H0/H1, MDE, sample size, success metric
│   ├── hypothesis_register.md            # one row per hypothesis, owner, test, status
│   ├── metric_definition.md              # NEW — single-source-of-truth metric spec (AE)
│   ├── model_card.md                     # Google Model Cards format
│   ├── data_card.md
│   └── postmortem.md
│
├── domain_accelerators/                   # deeper guided instructions for common problems
│   ├── ad_click_prediction.md
│   ├── churn_prediction.md
│   ├── demand_forecasting.md
│   ├── fraud_detection.md
│   └── recommender_systems.md
│
├── examples/
│   └── reference_implementations/
│       └── ad_click/                     # full working notebooks: EDA, FE, training, inference
│           ├── README.md
│           ├── eda.ipynb
│           ├── feature_engineering.ipynb
│           ├── model_training.ipynb
│           └── inferencing.ipynb
│
└── archive/                               # provenance, kept out of main paths
    ├── opus_drafts/                      # was skills/opus/
    └── gemini_drafts/                    # was skills/gemini/
```

---

## 4. Gap Analysis — What's Missing in the Current 4 Playbooks

| Existing | Missing |
|---|---|
| `EDA.md` | Hypothesis generation step; assumption checks (normality, independence) before Pearson; multiple-testing correction across feature scans; sample-size adequacy per segment; explicit causal-vs-correlational disclaimer |
| `FEATURE_ENGINEERING.md` | Leakage taxonomy (target / temporal / train-test); stationarity check before time splits; fairness-proxy audit; schema contract from EDA |
| `MODEL_TRAINING.md` | Calibration (Brier, reliability, Platt/Isotonic); fairness slicing (TPR/FPR parity); error analysis / slice discovery; required naive baselines; cost-aware threshold tuning |
| `INFERENCING.md` | Concept drift (label drift, performance decay); shadow mode; rollback runbook; per-row explainability snapshot (SHAP) |
| **Across all** | First-class hypothesis testing; business framing; stakeholder communication; experiment / A-B design; **modeled-mart / semantic-layer ownership** (analytics engineering) |

---

## 5. New Playbooks — Scope Summary

### `00_PROBLEM_FRAMING.md`
Business question → decision being supported → KPI → current baseline → target lift → cost asymmetry → stakeholder map → scope cut-offs. Output: filled `templates/problem_statement.md`.

### `01_DATA_CONTRACT.md`
Schema, freshness SLA, ownership, PII flags, allowed value ranges, drift tolerance. Handoff artifact between data engineer and data scientist.

### `01b_DATA_MODELING.md` *(owned by analytics_engineer)*
Dimensional modeling (star/snowflake), fact-table grain, slowly-changing-dimension types (SCD-0/1/2/3/6), conformed dimensions, surrogate vs natural keys, mart layout (staging → intermediate → marts), dbt-style project conventions, semantic-layer / metric definitions, data tests (uniqueness, not_null, accepted_values, referential integrity, freshness), lineage and column-level documentation. Output: a tested, documented set of marts that EDA and FE can rely on.

### `03_HYPOTHESIS_TESTING.md` *(largest new doc)*
- Stating hypotheses (H0/H1, directional, one/two-tailed)
- Test selection decision tree (Z, t, Welch, Mann-Whitney, Wilcoxon, χ², Fisher, ANOVA, Kruskal-Wallis, Levene, Shapiro)
- **Premise checks for each test** (links to `premises/parametric_tests.md`)
- Sample size & power analysis (`statsmodels.stats.power`)
- Effect size alongside p-value (Cohen's d, η², φ, OR, Cramér's V)
- Multiple testing correction (Bonferroni, BH-FDR)
- Bootstrap / permutation as nonparametric fallbacks
- **Practical vs statistical significance** as required output
- Reporting template: claim → test → assumption checks → n → statistic → p → effect size → CI → conclusion → caveats

### `06_MODEL_EVALUATION.md`
Calibration, fairness slicing, error analysis, baseline comparison, cost-matrix-driven threshold selection.

### `08_MONITORING.md`
Scheduled performance jobs, label-arrival latency, drift alerting, retraining triggers, rollback procedure.

### `09_EXPERIMENTATION.md`
A/B design, power, MDE, sample-ratio mismatch, novelty effects, sequential testing pitfalls, CUPED, switchback for marketplace effects.

---

## 6. Per-Skill Enrichments (existing playbooks)

**EDA** — add Section 0.5 "Problem framing carry-over"; Section 11 "Hypothesis register" (handoff to `03`); make correlation assumption-aware (Pearson only when ≈ normal, else Spearman); χ² with small-cell warning.

**Feature Engineering** — add leakage audit table (per-feature timing); fairness-proxy audit before encoding; temporal-split option when `Timestamp` present.

**Model Training** — required naive/most-frequent baseline; calibration step before champion selection; slice/segment metric table; business-cost threshold (not just F1).

**Inferencing** — shadow mode in flow diagram; SHAP snapshot artifact; rollback runbook; promote drift to `08_MONITORING.md`.

---

## 7. Persona Consolidation

| Action | Source | Destination |
|---|---|---|
| Move | `skills/*.md` (root) | `data-skills/personas/*.md` |
| Move + rename | `skills/architet.md` | `data-skills/personas/_template.md` |
| Move | `skills/opus/`, `skills/gemini/` | `data-skills/archive/` |
| Move out | `skills/projects/ice_gaming/` | `claude-code-data-science/projects/ice_gaming/` (parent repo, not in submodule) |
| Enrich | `data_scientist_reviewer.md` | Add explicit "statistical rigor gatekeeper" focus area |
| Enrich | `data_analyst.md` | Add "hypothesis author / register owner" responsibility |
| Enrich | `data_engineer.md` | Narrow scope to ingestion / infra / DAGs / raw → staging; explicit handoff to analytics_engineer for modeled marts |
| Create | `analytics_engineer.md` | New persona — owns dimensional modeling, semantic/metric layer, dbt-style transforms, marts, data quality tests, lineage |

---

## 8. Claude Code Skills Wiring

Each playbook and persona has a matching `SKILL.md` under `.claude/skills/`. Frontmatter pattern:

```markdown
---
name: eda
description: Use when starting exploratory data analysis on any dataset. Produces a hypothesis register and assumption-aware visualizations. Triggers on prompts mentioning "EDA", "explore data", "data profiling", "what's in this dataset".
---

@playbooks/02_EDA.md
```

```markdown
---
name: persona-data-analyst
description: Adopt this persona when extracting business intelligence from clean data, building dashboards, or framing the "so what" of a finding. Senior Data Analyst persona — thinks in SQL and story.
---

@personas/data_analyst.md
```

The `@path/to/file.md` reference keeps content single-sourced; SKILL.md is just a thin wrapper with the discoverability metadata.

---

## 9. Sequencing (independently mergeable steps)

1. ~~**Restructure** — create folders, move existing files, no content edits~~ **DONE**
2. ~~**Add submodule plumbing** — `README.md`, `.gitignore`, `LICENSE`, top-level `CLAUDE.md`~~ **DONE**
3. ~~**Write `03_HYPOTHESIS_TESTING.md`** — biggest gap, highest leverage~~ **DONE**
4. ~~**Write `00_PROBLEM_FRAMING.md` + `01_DATA_CONTRACT.md` + `01b_DATA_MODELING.md`**~~ **DONE** — also created `templates/problem_statement.md`
5. ~~**Author `analytics_engineer.md` persona**~~ **DONE** + **narrow `data_engineer.md` scope** — deferred to step 6
6. ~~**De-domain + enrich existing 4 playbooks** (EDA, FE, training, inferencing)~~ **DONE**
7. **Write `06_MODEL_EVALUATION.md` + `08_MONITORING.md` + `09_EXPERIMENTATION.md`** ← **NEXT**
8. **Build `premises/`, `checklists/`, `templates/`** by extracting from playbooks (deduplicate)
9. ~~**Wrap everything as Claude Code skills** under `.claude/skills/`~~ **DONE** (initial wiring; new playbooks will be wired as they're created)
10. **Author 1–2 `domain_accelerators/`** (start with `ad_click_prediction.md`, derived from current notebooks)
11. ~~**Move ad-click notebooks** to `examples/reference_implementations/ad_click/`~~ **DONE** *(done early at user request)*
12. **Convert to git submodule** — extract `data-skills/` into its own repo, re-add as submodule

Each step is a separate commit / PR.

---

## 10. Open Questions for Future Iterations

- Versioning strategy for the submodule (semver tags? rolling main?)
- Whether premises and checklists should also be Claude Code skills (probably yes, but lower priority)
- Whether to add a `governance/` folder for ethics/PII review checklists
- Whether to ship language packs (PT-BR translations) for personas
