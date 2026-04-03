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
| `--port` | | `29500` | Master node port for distributed communication |
| `--nnodes` | | cluster default | Override number of nodes (must not exceed nodes in cluster config) |
| `--nproc-per-node` | | cluster default | Override GPUs per node |
| `--tmux` / `--no-tmux` | | `--tmux` | Open each rank in a tmux window (requires an active tmux session) |
| `--log-level` | | `INFO` | Logging level |

When `--tmux` is active and a tmux session is detected, each node rank is launched in a separate tmux window named `rank-<N>`. Otherwise, each rank is launched as a subprocess.

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
