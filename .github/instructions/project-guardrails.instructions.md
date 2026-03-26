---
description: "Use when generating or editing code/docs in this project. Enforces math formatting, project-root-only file access, CARLA API compliance via python_api.md, and planning document logging with research-notebook-logger."
name: "Project Guardrails"
applyTo: "**"
---

# Project Guardrails

- Math formatting is mandatory in generated content:
  - Inline math must use `$...$`.
  - Display math must use `$$...$$`.

- File operations must stay inside the project root:
  - Never read from or write to paths outside this repository.
  - Assume the project root is the base for all operations.
  - Use relative paths rooted at the project directory.

- CARLA simulator code must follow local CARLA API docs:
  - For any CARLA-related code, consult `python_api.md` at the project root.
  - Keep behavior, signatures, and usage patterns compliant with that API reference.

- Planning and research logging policy:
  - For planning documents, use the `research-notebook-logger` agent when appropriate.
  - Save planning/research logs in `.reports/` at the project root.
  - Use clear, descriptive filenames that reflect topic and date.