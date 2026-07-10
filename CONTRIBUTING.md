# Contributing Guide

Thanks for your interest in the project! Contributions of any size are
welcome — bug reports, doc fixes, new experiments, frontend polish.

## Getting set up

```bash
git clone https://github.com/AlirezaMoazen/Multi-Agent-Rescue-Intelligence-Platform.git
cd Multi-Agent-Rescue-Intelligence-Platform
pip install -e ".[dev]"                 # Python package + pytest + ruff
pip install torch --index-url https://download.pytorch.org/whl/cpu  # deep RL / MoE
```

Or use Docker for everything: `docker compose up --build viz`.

## Workflow

1. **Branch** from `main`:
   ```bash
   git checkout -b your-feature-name
   ```
2. **Make your change.** Keep it focused — one topic per branch.
3. **Check it passes what CI runs:**
   ```bash
   ruff check src tests scripts
   pytest
   ```
   If you touched the frontend, also make sure it builds:
   ```bash
   cd src/rescue_sim/visualization/frontend && npm run build
   ```
4. **Open a merge/pull request** describing what you changed and how you
   tested it. A maintainer reviews it before merge — please don't merge your
   own unreviewed changes.

## Guidelines

- Match the style of the surrounding code; `ruff` (line length 100) is the
  source of truth for Python formatting issues.
- Add or update tests under `tests/` for behavior changes.
- Update the relevant docs (`README.md`, `docs/architecture.md`) when you
  change how something works or is run.
- Retrained checkpoints (`checkpoints/*.pt`) should only be committed when
  the training change that produced them is part of the same request.
