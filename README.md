# sflaunch

Utilities for LLM system research and development.

## Installation

```bash
pip install sflaunch
```

## sf-megarun

`sf-megarun` launches a distributed [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) training job across multiple nodes over SSH. It renders a `torchrun` launch script, saves it alongside a config snapshot to a timestamped output directory, then SSHes into each node to execute it.

### Usage

```bash
sf-megarun --cluster cluster.yaml --pretrain megatron.yaml
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--cluster` | `-c` | required | Path to cluster config YAML |
| `--pretrain` | `-p`, `-m` | required | Path to Megatron-LM job config YAML |
| `--port` | | random in `[30000, 49999]` | Master node port for distributed communication |
| `--nnodes` | | cluster default | Override number of nodes (must not exceed nodes in cluster config) |
| `--nproc-per-node` | | cluster default | Override GPUs per node (must not exceed each node's `num_gpus`) |
| `--tmux` / `--no-tmux` | | auto-detected | Open each rank in a tmux window. Defaults to true iff `$TMUX` is set; passing `--tmux` outside a tmux session fails fast |
| `--daemon` / `--no-daemon` | | `--no-daemon` | When `--no-tmux` is set, detach launched SSH processes so they survive `sf-megarun` exiting. Mutually exclusive with `--tmux` |
| `--log-level` | | `INFO` | Logging level |

Launch modes:

- **tmux** (default inside tmux): each rank opens in its own tmux window named `rank-<N>`. Closes on key press after the run.
- **process, foreground** (`--no-tmux`): each rank is a subprocess; `sf-megarun` blocks until all of them exit.
- **process, daemon** (`--no-tmux --daemon`): each rank is detached into its own session; `sf-megarun` exits immediately and the SSH/training processes keep running.

Heterogeneous clusters (nodes with different `num_gpus`) are rejected — torchrun requires a uniform `--nproc_per_node`. Pin a value with `--nproc-per-node` if needed.

### Cluster config

```yaml
# cluster.yaml
nodes:
  - ip_addr: 10.0.0.1
    ssh_target: node1      # defaults to ip_addr if omitted
    num_gpus: 8            # default: 8
  - ip_addr: 10.0.0.2
    ssh_target: node2
    num_gpus: 8

working_dir: /path/to/working/dir   # must exist on each node
script: /path/to/train_script.py    # path to the Megatron training script
output_dir: /path/to/output         # base dir for logs and run artifacts
env_setup: "source /path/to/venv/bin/activate"  # optional
```

### Megatron job config

```yaml
# megatron.yaml
name: my-pretrain-job

env:
  CUDA_DEVICE_MAX_CONNECTIONS: "1"
  NCCL_DEBUG: "INFO"

argv:
  - --num-layers="32"
  - --hidden-size="4096"
  # ... other Megatron-LM arguments
```

### Output

Each run creates a directory at `<output_dir>/<job_name>/<date>/<timestamp>/` containing:

- `config.yaml` - snapshot of the cluster and job configs used
- `run.sh` - the generated torchrun launch script
- `node-<rank>.log` - stdout/stderr from each node (written on the node)

## Miscellaneous

### Generating JSON schemas

To get JSON schemas for the config files (useful for editor validation):

```bash
python scripts/generate_schemas.py --output <dir>
```

This writes `cluster.schema.json` and `megatron-job.schema.json` to the specified directory. You can then reference these in your editor for YAML validation and autocompletion.
