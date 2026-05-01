# Problem Statement: [Project Name]

> **Version:** 1.0
> **Date:** [YYYY-MM-DD]
> **Author(s):** [Names and roles]

---

## 1. Business Question

<!-- What specific question does the business need answered? Avoid vague goals like "improve revenue."
     A good question: "Which customer segments have the highest churn risk in the next 30 days, and what intervention reduces it?"
     A bad question: "Help us understand our customers better." -->

**Question:** [Replace with a specific, answerable question]

**Context:** [Why is this question being asked now? What triggered it?]

---

## 2. Decision Being Supported

<!-- What action will the business take based on this analysis? A model that doesn't change a decision is waste. -->

**Decision:** [What will change based on the output?]

**Decision-maker:** [Who has authority to act on the findings?]

**Current process:** [How is the decision made today? (heuristic, gut feel, existing rule)]

---

## 3. Success Metric (KPI)

<!-- One primary metric. It must be measurable, attributable, and timely. -->

**Primary KPI:** [Metric name]

**Definition:** [How it is calculated — formula, data source, aggregation level]

**Measurement cadence:** [Daily / weekly / monthly / per-event]

**Secondary metrics** (optional): [List any secondary metrics that provide supporting signal]

---

## 4. Current Baseline

<!-- What is the current performance on the KPI? Without a baseline, "improvement" is undefined. -->

**Baseline value:** [Current KPI value with date range]

**Baseline source:** [Where this number comes from — dashboard, query, manual calculation]

**Baseline confidence:** [High / Medium / Low — is this number trustworthy?]

---

## 5. Target Lift / Minimum Viable Improvement

<!-- What improvement is worth the cost of building and maintaining a model? -->

**Minimum viable lift:** [Smallest improvement that justifies action, e.g., "5% reduction in churn"]

**Aspirational target:** [Best-case improvement if the model performs well]

**ROI justification:** [Back-of-envelope: what is each unit of improvement worth?]

**Break-even threshold:** [At what lift does the project pay for itself?]

---

## 6. Cost Asymmetry

<!-- Not all errors are equal. Define the cost of each error type. This directly feeds threshold selection in model evaluation. -->

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | True Positive: [consequence] | False Negative: [consequence and cost] |
| **Actually Negative** | False Positive: [consequence and cost] | True Negative: [consequence] |

**Which error is more costly?** [FP / FN / roughly equal]

**Cost ratio estimate:** [e.g., "A false negative costs ~10x more than a false positive"]

---

## 7. Stakeholder Map

| Stakeholder | Role | Needs from this project | Review cadence |
|---|---|---|---|
| [Name/Team] | Decision-maker | [What they need] | [How often] |
| [Name/Team] | Data owner | [What they provide] | [How often] |
| [Name/Team] | Model consumer | [How they use the output] | [How often] |
| [Name/Team] | Methodology approver | [What they review] | [How often] |
| [Name/Team] | Production maintainer | [What they maintain] | [How often] |

---

## 8. Scope & Cut-offs

### In Scope

- [List what this project WILL do]

### Out of Scope

- [List what this project will NOT do — be explicit]

### Timeline

- **Start date:** [YYYY-MM-DD]
- **Target delivery:** [YYYY-MM-DD]
- **Hard deadline:** [YYYY-MM-DD, if any]

### Boundaries

- **Geographic:** [All regions / specific regions]
- **Temporal:** [Date range for training data]
- **Segment:** [All customers / specific segments]

---

## 9. Data Availability Assessment

| Required signal / feature | Available? | Source | Freshness | Quality | PII? |
|---|---|---|---|---|---|
| [Feature name] | Yes / No / Partial | [Table or system] | [Lag] | [High/Med/Low] | [Y/N] |
| [Feature name] | Yes / No / Partial | [Table or system] | [Lag] | [High/Med/Low] | [Y/N] |

**Gaps identified:** [List features needed but not available]

**PII / regulatory concerns:** [List any flags for GDPR, CCPA, etc.]

**Data engineer handoff:** [What needs to be built or sourced before EDA can begin?]

---

## 10. Go / No-Go & Sign-off

**Recommendation:** [GO / NO-GO / CONDITIONAL GO]

**Rationale:** [Why this project should (or should not) proceed]

**Conditions (if conditional):** [What must be resolved before proceeding]

| Role | Name | Decision | Date |
|---|---|---|---|
| Data Analyst | [Name] | [GO / NO-GO] | [Date] |
| Product Manager | [Name] | [GO / NO-GO] | [Date] |
| Data Scientist Reviewer | [Name] | [GO / NO-GO] | [Date] |
