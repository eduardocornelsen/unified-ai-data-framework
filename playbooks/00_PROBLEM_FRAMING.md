# Problem Framing Playbook

## Role

You are a senior Data Analyst and Product Manager jointly responsible for converting a business question into a precise, scoped problem statement. Your output is a filled `templates/problem_statement.md` that the entire downstream pipeline — EDA, hypothesis testing, feature engineering, modeling, evaluation — will anchor on.

A poorly framed problem guarantees a well-executed wrong answer. Spend the time here.

---

## Environment

No compute or coding tools are required for this phase. This is a structured thinking exercise that produces a document, not a notebook. You need:

- Access to stakeholders (or their documented requirements)
- Domain context (industry, competitive landscape, regulatory constraints)
- A data catalog or inventory of available data sources (if one exists)
- The blank template at `templates/problem_statement.md`

---

## Inputs

| Input | Source | Description |
|---|---|---|
| Business question | Stakeholder / user prompt | The raw request — possibly vague, aspirational, or multi-part |
| Domain context | Stakeholder interviews, documentation, industry knowledge | Constraints, regulations, competitive dynamics, prior attempts |
| Data catalog | Data Engineer / data platform (if available) | What data exists, its freshness, quality, and accessibility |

---

## Deliverables

1. **Filled `templates/problem_statement.md`** — all 10 sections complete, specific, and internally consistent
2. **Go / No-Go recommendation** — is this problem solvable with available data and resources?
3. **Handoff summary** — one-paragraph brief for the Data Engineer (what data is needed) and for EDA (what to explore first)

---

## Process Structure

### Step 0. Setup

- Load the blank template from `templates/problem_statement.md`
- Gather any prior context the user has provided (business briefs, Slack threads, prior analyses, dashboards)
- Identify what is known vs. what needs to be discovered through this process
- If the user provided a vague request, prepare clarifying questions for each gap

### Step 1. Articulate the Business Question

Force precision. The goal is to transform a vague goal into a specific, answerable question.

**Bad examples:**
- "Improve revenue" — not a question
- "Understand our customers" — no decision attached
- "Build a churn model" — solution-first, not problem-first

**Good examples:**
- "Which customer segments have the highest churn risk in the next 30 days, and what intervention reduces it?"
- "What is the expected lifetime value of customers acquired through each marketing channel, and should we reallocate spend?"

**Checklist for a well-formed question:**
- [ ] It is a question (not a goal or a directive)
- [ ] It is specific enough to have a testable answer
- [ ] The answer directly informs a decision (see Step 2)
- [ ] It has a time horizon
- [ ] It is scoped to a population or segment

If the business question is actually multiple questions, split them. Each gets its own problem statement. Do **not** bundle.

### Step 2. Map the Decision

Every analysis must support a decision. If no decision changes based on the output, the project is academic exercise — and should be labeled as such or reconsidered.

Document:
- **The decision:** What action will change based on the analysis? (*e.g., "Adjust retention offer targeting from rule-based to model-based"*)
- **The decision-maker:** Who has authority to act? (*e.g., "VP of Customer Success"*)
- **The current process:** How is this decision made today? (*e.g., "All customers in the bottom quartile of engagement get the same email"*)
- **The counterfactual:** What happens if we do nothing? (*e.g., "Churn continues at 8.2% monthly with flat intervention costs"*)

### Step 3. Define the Success Metric (KPI)

Choose **one primary KPI**. Multiple KPIs create ambiguity about what "better" means.

A good KPI is:
- **Measurable** — it can be computed from available data
- **Attributable** — changes in the metric can be linked to the intervention
- **Timely** — it can be measured within the project timeline (a 12-month metric for a 2-week project is a mismatch)
- **Not a vanity metric** — page views, raw counts, or model accuracy without context are not KPIs

Secondary metrics are allowed but must be explicitly labeled as secondary. They provide context but do not drive the optimization target.

Fill in Section 3 of the template with: metric name, formula, data source, measurement cadence, and any secondary metrics.

### Step 4. Establish the Baseline

What is the current performance on the KPI? Without a baseline, "improvement" is undefined.

- Pull the current value from a dashboard, query, or manual calculation
- Document the date range and source
- Rate the confidence: High (automated dashboard, audited), Medium (ad-hoc query, reasonable assumptions), Low (estimate, derived)
- If **no baseline exists**, define how to measure one. The first project milestone may be "establish a reliable baseline" rather than "build a model"

### Step 5. Set the Target Lift / Minimum Viable Improvement

Not all improvements are worth pursuing. A 0.1% lift in conversion rate is statistically detectable with enough data but may not justify the engineering and maintenance cost of a production model.

Define:
- **Minimum viable lift** — the smallest improvement that justifies the project's cost
- **Aspirational target** — best-case outcome
- **ROI justification** — back-of-envelope: what is each unit of improvement worth in dollars, hours, or customer outcomes?
- **Break-even threshold** — at what lift does the project pay for itself?

This forces an honest conversation about whether the project is worth doing *before* any data is touched.

### Step 6. Assess Cost Asymmetry

Not all errors are equal. The cost matrix defines the relative penalty of false positives vs. false negatives. This directly feeds threshold selection during model evaluation — it is **not** optional.

Fill in the cost matrix in the template:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | TP: [benefit] | FN: [cost of missing] |
| **Actually Negative** | FP: [cost of false alarm] | TN: [benefit] |

Examples by domain:
- **Fraud detection:** FN (missed fraud) costs $$$; FP (blocking a legitimate transaction) costs customer trust
- **Medical screening:** FN (missed disease) can be fatal; FP (unnecessary follow-up) costs time and anxiety
- **Ad targeting:** FP (irrelevant ad) costs a click; FN (missed opportunity) costs potential revenue
- **Churn prediction:** FN (lost customer) costs lifetime value; FP (unnecessary retention offer) costs the discount

State which error is more costly and estimate the ratio. A 10:1 FN:FP cost ratio produces very different model behavior than a 1:1 ratio.

### Step 7. Map Stakeholders

Identify every person or team who touches this project. Use the stakeholder table in the template:

| Stakeholder | Role | Needs from this project | Review cadence |
|---|---|---|---|
| Decision-maker | Acts on output | Actionable recommendations | Weekly/at delivery |
| Data owner | Provides data access | Clear data requirements | At start / ad-hoc |
| Methodology approver | Reviews statistical rigor | Sound methodology, reproducibility | At gate reviews |
| Model consumer | Uses model output | Reliable predictions, SLA guarantees | Ongoing |
| Production maintainer | Keeps model running | Documentation, monitoring alerts, runbooks | At handoff |

Missing a stakeholder here means surprises later. Better to over-identify than under-identify.

### Step 8. Define Scope & Cut-offs

Scope creep is the most common failure mode in data science projects. Prevent it with explicit boundaries.

**In Scope:** List what this project WILL do. Be specific — "build a churn prediction model for B2B SaaS customers in North America using the last 12 months of behavior data."

**Out of Scope:** List what this project will NOT do. Be explicit — "real-time scoring, European customer segments, causal analysis of churn drivers." Out-of-scope items are not rejected — they are deferred.

**Timeline:** Start date, target delivery, hard deadline (if any). If no deadline exists, state that — it affects prioritization.

**Boundaries:** Geographic, temporal, and segment constraints. What subset of the data universe is in play?

### Step 9. Data Availability Assessment

Cross-reference the features and signals implied by the business question against the data catalog. For each required signal:

| Required signal | Available? | Source | Freshness | Quality | PII? |
|---|---|---|---|---|---|
| *e.g., login frequency* | Yes | `events.user_sessions` | 1-hour lag | High | No |
| *e.g., support ticket sentiment* | Partial | `support.tickets` (no NLP scores) | Daily | Medium | Yes (PII in text) |

Identify:
- **Gaps:** Features that are needed but do not exist. These may require data engineering work (handoff to `01_DATA_CONTRACT.md`) or may reduce the project scope.
- **PII / regulatory concerns:** Flag columns that contain personally identifiable information, health data, financial records, or anything subject to GDPR, CCPA, HIPAA, or other regulations. Document the handling plan (anonymize, aggregate, exclude).
- **Data engineer handoff:** Summarize what the data engineer needs to provide or build before EDA can begin.

### Step 10. Go / No-Go & Sign-off

Assess whether this project should proceed. The answer is not always "go."

**GO criteria (all must be met):**
- [ ] Business question is specific and decision-linked
- [ ] KPI is defined with a measurable baseline
- [ ] Data is available (or can be sourced within timeline)
- [ ] Minimum viable lift is achievable with available data (reasonable belief)
- [ ] Stakeholders are identified and aligned

**NO-GO triggers (any one blocks):**
- No clear decision attached to the analysis
- No measurable KPI or no baseline
- Critical data is unavailable and cannot be sourced in time
- The minimum viable lift is unrealistic given data quality
- Stakeholder misalignment on goals or scope

**CONDITIONAL GO:** The project can proceed if specific conditions are met. List the conditions and the owner for each.

Fill in the sign-off table in the template.

---

## Documentation Standards

- Every assertion must be supported by evidence (data, stakeholder quote, prior analysis) or explicitly labeled as an assumption
- Write for a stakeholder who is not a data scientist — no jargon without definition
- Use the template structure exactly — do not skip sections or reorder them
- If a section cannot be filled, explain why and what would be needed to fill it
- Quantify whenever possible — "significant churn problem" is vague; "8.2% monthly churn rate, 3x industry average" is useful

---

## Constraints & Guardrails

- Do **NOT** skip the cost asymmetry assessment — it directly affects model evaluation and threshold selection downstream
- Do **NOT** define more than one primary KPI — secondary metrics are fine, but the optimization target must be singular
- Do **NOT** assume data availability — verify with the data catalog or data engineer
- Do **NOT** proceed past this phase if the business question cannot be stated as a specific, testable question with a decision attached
- Do **NOT** scope a project that cannot be baselined — "improve X" without a current X is undefined
- Do **NOT** conflate correlation with causation in the problem statement — if the goal is causal inference, say so explicitly and plan the methodology accordingly (see `09_EXPERIMENTATION.md`)

---

## Acceptance Criteria

| Criterion | Definition of Done |
|---|---|
| Business question | Stated as a specific, answerable question with a decision attached |
| KPI defined | One primary metric with formula, data source, measurement cadence, baseline, and target |
| Cost asymmetry | FP/FN costs documented; asymmetry ratio estimated; feeds downstream threshold selection |
| Scope locked | In-scope and out-of-scope lists present; timeline stated; boundaries defined |
| Data feasibility | Data catalog cross-referenced; gaps identified; PII flagged with handling plan |
| Sign-off | Template fully filled; go/no-go stated with rationale; stakeholder table populated |

---

## Downstream Handoffs

| Recipient | What they receive | Playbook |
|---|---|---|
| Data Engineer | Data requirements, gaps, and PII flags from Step 9 | `01_DATA_CONTRACT.md` |
| Analytics Engineer | KPI definitions and required dimensions from Steps 3 and 7 | `01b_DATA_MODELING.md` |
| Data Analyst | Business question, hypothesis seeds, and scope from Steps 1 and 8 | `02_EDA.md` |
| Data Scientist Reviewer | Full problem statement for methodology gate review | Any downstream playbook |
