# Unified AI Data Framework

An AI-native data science operating system designed to guide LLM agents (like Claude Code) through end-to-end data projects. 

This repository acts as the "Brain" for your AI. It combines high-level strategic project management (Playbooks and Personas) with deep tactical execution tools (Skills and Templates).

---

## 🏗 Architecture & Usage

This framework is divided into three core pillars:

### 1. Personas (`/personas`)
**The "Who"**. AI agents perform better when given strict behavioral constraints. This directory contains profiles for Data Engineers, Product Managers, Data Scientists, etc. 
* **How to use:** Instruct the AI: *"Read `personas/data_analyst.md` and adopt this persona for our session."*

### 2. Playbooks (`/playbooks`)
**The "When"**. A linear set of markdown guides (from `00_PROBLEM_FRAMING` to `07_INFERENCING`) that outline the standard lifecycle of an ML/Data project.
* **How to use:** Instruct the AI: *"We are currently in Phase 3. Read `playbooks/03_HYPOTHESIS_TESTING.md` and guide me through the next steps."*

### 3. Skills Library (`/skills`)
**The "How"**. A massive, granular encyclopedia of specific analytical techniques. It contains markdown templates, Python scripts, and SQL blueprints for exact tasks (e.g., A/B Testing, Cohort Analysis, Data Quality Audits).
* **How to use:** Instruct the AI: *"We need to write an executive summary. Fetch the template from `skills/04-data-storytelling-visualization/executive-summary-generator` and populate it with our findings."*

---

## 🤝 Acknowledgements & Citations

The `skills/` directory in this repository was sourced and adapted from the awesome **[data-analytics-skills](https://github.com/ai-analyst-lab/data-analytics-skills)** repository. 

### Why did we merge it?
Our original framework (`data-skills`) excelled at process management (Playbooks) but lacked tactical depth. When Claude was asked to "do a cohort analysis," it had to hallucinate the structure from its pre-training. 

By importing the `data-analytics-skills` repository into our `skills/` module, we provide the AI with highly vetted, deterministic templates and scripts for specific analytical tasks. 

### How it complements our analysis:
- **Strategy + Tactics:** The Playbooks tell the AI *that* it needs to validate data quality. The Skills library tells the AI *exactly how* to run a null-profiler script to achieve it.
- **Consistency:** By forcing the AI to use the templates inside `skills/`, we guarantee that every A/B test, metric reconciliation, and dashboard specification output by the AI adheres to a standardized, enterprise-grade format.

---

## 🚀 Quick Start for Claude Code

1. Open Claude Code inside this repository.
2. Initialize your project context:
   ```text
   > I want to start a new data project. Please review playbooks/00_PROBLEM_FRAMING.md and help me draft a problem statement using the template in the skills/ directory.
   ```
3. Let the AI guide you through the process, pulling skills dynamically as the analysis deepens.
