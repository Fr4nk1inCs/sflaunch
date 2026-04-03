# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Python 3.12, managed with `uv`. The venv is already at `.venv/`.

```bash
uv sync              # install dependencies
uv sync --dev        # include dev dependencies (pre-commit)
```

## Running

The main entry point is the `sfmegarun` CLI, registered in `pyproject.toml`:

```bash
uv run sfmegarun --cluster <cluster.yaml> --pretrain <megatron.yaml>
# short aliases: -c for cluster, -p/-m for pretrain
```

Generate JSON schemas for config files:

```bash
uv run python scripts/generate_schemas.py --output <dir>
```

## Architecture

The sole implemented runner is `sfmegarun` (`src/sfutils/runners/megarun.py`). Its flow:

1. **Parse CLI args** (`CliArgs` extends `BaseCliArgs`) — accepts cluster and megatron job YAML paths, optional `--nnodes`/`--nproc-per-node` overrides, `--port`, and `--tmux`/`--no-tmux`.
2. **Load configs** — `ClusterConfig` (`schemas/cluster.py`) describes nodes (IP, SSH target, GPU count), working dir, env setup command, output dir, and script path. `MegatronJob` (`schemas/jobs/megatron.py` → `CliBasedJob`) holds job name, env vars, and `argv` (CLI args for the training script).
3. **Render script** — `templates/torchrun.py` fills a bash template that calls `torchrun` with the distributed parameters, sources env setup, exports env vars, and tees stdout/stderr to a per-node log file in the output directory.
4. **Create output directory** — `utils/OutputDirectory` creates `output_dir/<job_name>/<date>/<timestamp>/`, writes `config.yaml` (snapshot of both configs) and `run.sh`.
5. **Launch** — iterates over nodes, SSHes into each (`ssh -t <ssh_target> -- bash run.sh <rank>`). Launcher is either tmux (opens one window per rank in the current session) or multiprocessing.

### Config schemas

| File | Schema class | Purpose |
|------|-------------|---------|
| `schemas/cluster.py` | `ClusterConfig` | Cluster topology and paths |
| `schemas/jobs/base.py` | `CliBasedJob` | Base job: `name`, `env`, `argv` |
| `schemas/jobs/megatron.py` | `MegatronJob` | Megatron-LM job (currently just `CliBasedJob`) |
| `schemas/cli_args.py` | `BaseCliArgs` | Shared CLI base with `--log-level` |

Pydantic v2 + pydantic-settings `CliApp` is used for CLI parsing. All CLI flags use kebab-case (configured in `BaseCliArgs`).

### Adding a new runner

Follow the pattern in `runners/megarun.py`: define a `CliArgs(BaseCliArgs)` with runner-specific fields, implement `cli_cmd()`, and expose a `main()` that calls `CliApp.run(CliArgs)`. Register the entry point in `pyproject.toml` under `[project.scripts]`.
