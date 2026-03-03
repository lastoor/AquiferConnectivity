# Agent Guide

## Python Environment

Use the local Conda environment `fmm_env` for all Python commands in this repository.

Preferred command pattern:

```bash
./scripts/run_in_fmm_env.sh <command>
```

Example:

```bash
./scripts/run_in_fmm_env.sh python ./scripts/run_fmm.py
```

## FlowSimulation Dependency

This project depends on the sibling project at `../../FlowSimulation`.

- `./scripts/run_in_fmm_env.sh` adds both `./src` and `../../FlowSimulation` to `PYTHONPATH` before running commands.
- Optional override:

```bash
FLOWSIM_PATH=/abs/path/to/FlowSimulation ./scripts/run_in_fmm_env.sh python <command>
```

## Local Module Layout

Keep FMM-specific logic in `./src/fmm_core`:

- `fmm.py`: fast marching and source-target path extraction.
- `metrics.py`: path distance metrics and resampling helpers.

## Documentation Maintenance

When making repo changes, keep both documentation files up to date:

- `./README.md`: update usage, environment commands, and dependency notes when behavior or setup changes.
- `./agent/log.md`: append a dated bullet summary of what changed and why.

For every non-trivial change, update `./agent/log.md` in the same edit set.

## Validation Commands

Use these checks after major algorithm or interface changes:

```bash
PYTHONPATH=../../FlowSimulation:./src conda run -n fmm_env python -m py_compile scripts/run_fmm.py src/fmm_core/*.py
PYTHONPATH=../../FlowSimulation:./src conda run -n fmm_env pytest -q
```
