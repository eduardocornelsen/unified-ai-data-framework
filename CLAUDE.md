# Unified AI Data Framework — agent instructions

This directory is the core operating system and skill library for your data science work.

- **Playbooks** in `playbooks/` are the source of truth for *when* and *what* to do. Always open the relevant playbook (e.g., `00_PROBLEM_FRAMING`) to understand the current project phase.
- **Personas** in `personas/` define *who* you are acting as. Adopt the relevant persona via `Skill(persona-...)` to set tone, focus, and responsibilities.
- **Skills** in `skills/` provide *how* to execute tasks. This is your tactical library. Use it to pull standardized analytical templates, Python scripts, and SQL blueprints (e.g., for cohort analysis or data quality audits). Do not paraphrase analytical structures from memory; use these blueprints.
- **Templates** in `templates/` and `skills/*/assets/` are copy-paste artifacts.
- **Reference implementations** in `examples/` show what an executed output looks like.

The `.claude/skills/` directory exposes core tasks as invocable Claude Code skills. Prefer `Skill(...)` over `Read`-then-paraphrase.

For the current development plan and roadmap, see [`PLAN.md`](./PLAN.md).
