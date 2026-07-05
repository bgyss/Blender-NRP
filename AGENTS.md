# AGENTS.md

This repository's agent guidance is maintained in [CLAUDE.md](CLAUDE.md).
Read that file before making changes, and treat it as the source of truth for
commands, architecture, compatibility contracts, and repo-specific conventions.

## Codex Notes

- Keep changes aligned with the contracts and validation tiers documented in
  [CLAUDE.md](CLAUDE.md).
- Prefer repo-relative paths in docs and handoffs.
- Keep generated artifacts under ignored output directories such as `build/` and
  `dist/`; do not commit caches, model files, previews, reports, or package zips.
- When behavior changes, run the narrowest relevant check from [CLAUDE.md](CLAUDE.md)
  and report any checks that could not be run.
