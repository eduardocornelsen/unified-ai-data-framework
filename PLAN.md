# Unified AI Data Framework — Repo Plan

> Status: active
> Owner: Eduardo Cornelsen
> Last updated: 2026-05-01

## 1. Purpose

`unified-ai-data-framework` is the **Hub** in a Hub/Engine/Product architecture. It is a standalone, plug-and-play skill library that turns a Claude Code agent into a senior data science team. It contains:

- **Playbooks** — step-by-step instructions for each phase of a DS project (problem framing, data contracts, EDA, hypothesis testing, feature engineering, training, evaluation, inferencing, monitoring, experimentation).
- **Personas** — role-specific system prompts (data analyst, analytics engineer, ML engineer, reviewer, etc.) that set tone, focus, and gates.
- **Skills** — a tactical library of 33 modular analytical techniques organized into 6 domains (data quality, documentation, analysis, storytelling, communication, workflow optimization). Sourced from [data-analytics-skills](https://github.com/ai-analyst-lab/data-analytics-skills).
- **Templates** — copy-paste scaffolds for problem statements, model cards, experiment designs.
- **Reference implementation** — the ad-click pipeline as a working end-to-end example.

Playbooks, personas, and the batch-analysis pipeline are exposed as **Claude Code skills** under `.claude/skills/`, invocable via `Skill(...)`.

### Ecosystem Context

This repo is the **Hub (Brain)** in a three-part architecture:
- **Hub** (this repo) — Pure markdown instructions, personas, playbooks, and skills. Zero runtime deps.
- **Engine** (`ai-analyst`) — Python execution backend with database connections, statistical helpers, and slide generation.
- **Product** (`full-funnel-ai-analytics`) — Deployable full-stack application (dbt, Streamlit, FastAPI, MCP servers).

The Hub can be added as a git submodule to either Engine or Product when they need its methodology.

---

## 2. Decisions (locked)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Independent submodule** | Versioned separately, reusable across projects. Has its own `.gitignore`, `README.md`, `LICENSE`. No `pyproject.toml` (it's prompts + markdown, not a Python package). |
| 2 | **Generic-by-default, with domain accelerators** | Each playbook is dataset-agnostic. Common business problems get **deeper guided instructions** in `domain_accelerators/`. |
| 3 | **No skeleton-notebook folder** | Skeletons rot. One full **reference implementation** under `examples/reference_implementations/ad_click/`. Inline code recipes live inside playbooks. |
| 4 | **Both playbooks and personas exposed as Claude Code skills** | Skills are the execution surface. Each playbook = `Skill(eda)`, each persona = `Skill(persona-data-analyst)`. Project-local under `.claude/skills/`. |
| 5 | **Add `analytics_engineer` persona** | Owns the modeled-mart / semantic layer / data-test layer. Without it, every DS playbook assumes clean modeled data exists — that assumption needs an owner. |
| 6 | **Merge `data-analytics-skills` as `skills/` library** | Gives the AI tactical depth — vetted templates, scripts, and SQL blueprints for specific analytical tasks instead of hallucinating structure from pre-training. |
| 7 | **Keep `ai-analyst` and `full-funnel-ai-analytics` as separate repos** | Different artifact types (Python app vs. full-stack product vs. pure markdown), different lifecycles, different consumers. Merging would create an unmanageable monolith. |

---

## 3. Current Repo Structure

```
unified-ai-data-framework/
├── README.md
├── PLAN.md                                # this file
├── CLAUDE.md
├── LICENSE
│
├── .claude/
│   └── skills/                            # invocable Claude Code skills
│       ├── playbooks/
│       │   ├── problem-framing/SKILL.md
│       │   ├── data-contract/SKILL.md
│       │   ├── data-modeling/SKILL.md
│       │   ├── eda/SKILL.md
│       │   ├── hypothesis-testing/SKILL.md
│       │   ├── feature-engineering/SKILL.md
│       │   ├── model-training/SKILL.md
│       │   ├── inferencing/SKILL.md
│       │   └── batch-analysis/SKILL.md
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
│   ├── 00_PROBLEM_FRAMING.md
│   ├── 01_DATA_CONTRACT.md
│   ├── 01b_DATA_MODELING.md
│   ├── 02_EDA.md
│   ├── 03_HYPOTHESIS_TESTING.md
│   ├── 04_FEATURE_ENGINEERING.md
│   ├── 05_MODEL_TRAINING.md
│   └── 07_INFERENCING.md
│
├── personas/                              # canonical content
│   ├── _template.md
│   ├── data_scientist_reviewer.md
│   ├── data_analyst.md
│   ├── data_engineer.md
│   ├── analytics_engineer.md
│   ├── ml_engineer.md
│   ├── product_manager.md
│   ├── ux_researcher.md
│   ├── frontend_developer.md
│   └── qa_tester.md
│
├── skills/                                # tactical analytics library (from data-analytics-skills)
│   ├── 01-data-quality-validation/        # 5 skills
│   ├── 02-documentation-knowledge/        # 7 skills
│   ├── 03-data-analysis-investigation/    # 7 skills
│   ├── 04-data-storytelling-visualization/# 5 skills
│   ├── 05-stakeholder-communication/      # 5 skills
│   └── 06-workflow-optimization/          # 4 skills
│
├── templates/
│   └── problem_statement.md
│
├── scripts/
│   └── validate_skills.py
│
├── assets/
│   └── skill-map.svg
│
└── examples/
    └── reference_implementations/
        └── ad_click/
```

---

## 4. Remaining Work (priority order)

### Next up

| # | Task | Status |
|---|---|---|
| 1 | **Write `06_MODEL_EVALUATION.md`** — calibration, fairness, error analysis, baseline comparison | TODO |
| 2 | **Write `08_MONITORING.md`** — drift, decay, retraining triggers, rollback | TODO |
| 3 | **Write `09_EXPERIMENTATION.md`** — A/B design, power, MDE, CUPED | TODO |
| 4 | **Wire skills for 06, 08, 09** — add `.claude/skills/playbooks/` entries once playbooks exist | TODO |
| 5 | **Build `premises/`** — extract assumption checklists from playbooks (parametric, regression, classification, time series, causal) | TODO |
| 6 | **Build `checklists/`** — leakage, train-test contamination, multiple testing, effect sizes, reproducibility, stakeholder handoff | TODO |
| 7 | **Expand `templates/`** — experiment_design, hypothesis_register, metric_definition, model_card, data_card, postmortem | TODO |
| 8 | **Author `domain_accelerators/`** — start with ad_click_prediction.md, then churn, forecasting, fraud, recommenders | TODO |

### Done

| # | Task | Status |
|---|---|---|
| 1 | Merge `data-analytics-skills` into `skills/` directory | DONE |
| 2 | Create unified README.md and CLAUDE.md | DONE |
| 3 | Write playbooks 00–05, 07 (8 of 11) | DONE |
| 4 | Author all 10 personas (including analytics_engineer) | DONE |
| 5 | Wire `.claude/skills/` for all existing playbooks and personas (18 SKILL.md files) | DONE |
| 6 | Resolve duplicate skills (metric-reconciliation, schema-mapper renamed in domain 02) | DONE |
| 7 | Remove `.docs/` planning artifacts | DONE |

---

## 5. Open Questions

- Versioning strategy (semver tags? rolling main?)
- Whether `premises/` and `checklists/` should also be Claude Code skills
- Whether to add a `governance/` folder for ethics/PII review checklists
- Whether to ship language packs (PT-BR translations) for personas
