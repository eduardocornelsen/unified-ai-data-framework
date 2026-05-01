# Ad-Click Prediction — Reference Implementation

A full end-to-end reference implementation of the `data-skills` playbooks applied to a binary ad-click classification problem on **Databricks Serverless**. This is the canonical "this works" example — Claude regenerates project-specific notebooks from the playbooks; this folder shows what an executed result looks like.

## Pipeline

| Stage | Notebook | Playbook |
|---|---|---|
| Exploratory Data Analysis | `eda.ipynb` | `playbooks/02_EDA.md` |
| Feature Engineering | `feature_engineering.ipynb` | `playbooks/04_FEATURE_ENGINEERING.md` |
| Model Training | `model_training.ipynb` | `playbooks/05_MODEL_TRAINING.md` |
| Inference + Drift | `inferencing.ipynb` | `playbooks/07_INFERENCING.md` |

## Dataset

`ad_10000records.csv` — 10,000 ad impressions with user behavior, demographics, contextual fields, and a binary `Click on Ad` target.

## Layout

```
ad_click/
├── README.md
├── databricks.yml                   # Databricks Asset Bundle config
├── requirements.txt                 # extra pip packages beyond Serverless base
├── ad_10000records.csv              # source dataset
├── eda.ipynb
├── feature_engineering.ipynb
├── model_training.ipynb
├── inferencing.ipynb
└── utils/
    ├── eda_helpers.py
    ├── fe_helpers.py
    └── model_helpers.py
```

## How to run

1. Open the folder in a Databricks workspace (or attach via Databricks Asset Bundles using `databricks.yml`).
2. Run the notebooks in order. Each is idempotent and writes its outputs to Delta + MLflow.
3. Inspect the MLflow experiments for runs, params, metrics, and artifacts.

## Status

Pre-existing executed notebooks. Will be re-validated against the de-domained playbooks during sequencing step 6 (see `data-skills/PLAN.md`). Expect minor edits when the playbooks are generalized.
