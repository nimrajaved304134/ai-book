---
id: 1
title: Create Physical AI & Humanoid Robotics Textbook Constitution
stage: constitution
date: 2025-12-07
surface: agent
model: GPT-4
feature: none
branch: main
user: user
command: /sp.constitution
labels: ["constitution", "textbook", "physical-ai", "humanoid-robotics"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - none
---

## Prompt

$ARGUMENTS

As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

1) Determine Stage
   - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

2) Generate Title and Determine Routing:
   - Generate Title: 3–7 words (slug for filename)
   - Route is automatically determined by stage:
     - `constitution` → `history/prompts/constitution/`
     - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
     - `general` → `history/prompts/general/`

3) Create and Fill PHR (Shell first; fallback agent‑native)
   - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
   - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
   - If the script fails:
     - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
     - Allocate an ID; compute the output path based on stage from step 2; write the file
     - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

4) Validate + report
   - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
   - On failure: warn, don't block. Skip only for `/sp.phr`.

## Response snapshot

I have successfully updated the project constitution at `.specify/memory/constitution.md` for the "Physical AI & Humanoid Robotics Textbook" project with:

- **New Version**: 1.0.0 (initial version)
- **Ratification Date**: 2025-12-07
- **Last Amended Date**: 2025-12-07

The updated constitution includes six core principles specific to the textbook project:
1. Academic Excellence
2. Modularity & Structure
3. Test-First Education (NON-NEGOTIABLE)
4. Cross-Disciplinary Integration
5. Practical Application
6. Docusaurus Excellence

I also reviewed dependent templates to ensure consistency and added a Sync Impact Report at the top of the file. No files require manual follow-up.

**Suggested commit message**: `docs: create constitution for Physical AI & Humanoid Robotics textbook project v1.0.0`

All tasks have been completed successfully.

## Outcome

- ✅ Impact: Created new constitution for Physical AI & Humanoid Robotics textbook project with 6 core principles
- 🧪 Tests: none required
- 📁 Files: Updated .specify/memory/constitution.md with the new constitution
- 🔁 Next prompts: None needed
- 🧠 Reflection: The constitution effectively captures the academic and technical requirements for the textbook project

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A