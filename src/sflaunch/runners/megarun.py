import random
import shlex
from typing import Optional, Self, TypeVar

import yaml
from pydantic import BaseModel
from pydantic.aliases import AliasChoices
from pydantic.fields import Field
from pydantic.functional_validators import model_validator
from pydantic.types import FilePath, PositiveInt
from pydantic_settings import CliApp

from sflaunch.launchers import join_launched, make_launcher
from sflaunch.schemas.cli_args import BaseCliArgs
from sflaunch.schemas.cluster import ClusterConfig
from sflaunch.schemas.jobs.megatron import MegatronJob
from sflaunch.templates.torchrun import render_script as render_torchrun_script
from sflaunch.utils import OutputDirectory
from sflaunch.utils.log import get_logger, setup_logging
from sflaunch.utils.tmux import get_session


def _build_ssh_command(ssh_target: str, script_path: str) -> str:
    return f"< {shlex.quote(script_path)} ssh {shlex.quote(ssh_target)} -- bash -s --"


_M = TypeVar("_M", bound=BaseModel)

logger = get_logger(__name__)


def load_model(path: FilePath, model_cls: type[_M]) -> _M:
    with path.open("r") as f:
        data = yaml.safe_load(f)

    return model_cls.model_validate(data)


def render_script(
    cluster: ClusterConfig,
    job: MegatronJob,
    output_dir: OutputDirectory,
    master_port: int,
) -> str:
    gpu_counts = {node.num_gpus for node in cluster.nodes}
    if len(gpu_counts) > 1:
        raise ValueError(
            f"Heterogeneous num_gpus across nodes ({sorted(gpu_counts)}); "
            "torchrun requires a uniform --nproc_per_node. "
            "Pass --nproc-per-node to pin a value."
        )

    return render_torchrun_script(
        env_vars=job.env,
        script=cluster.script,
        argv=job.argv,
        working_dir=cluster.working_dir,
        log_dir=output_dir.base,
        env_setup=cluster.env_setup or "",
        nnodes=len(cluster.nodes),
        nproc_per_node=gpu_counts.pop(),
        master_addr=cluster.nodes[0].ip_addr.exploded,
        master_port=master_port,
    )


class CliArgs(BaseCliArgs):
    nnodes: Optional[PositiveInt] = Field(
        None,
        description="`--nnodes` passed to torchrun, overrides cluster config",
    )

    nproc_per_node: Optional[PositiveInt] = Field(
        None,
        description="`--nproc-per-node` passed to torchrun, overrides cluster config",
    )

    port: int = Field(
        default_factory=lambda: random.randint(30000, 49999),
        ge=1,
        le=65535,
        description="Port for master node communication "
        "(random in [30000, 49999] by default to avoid collisions)",
    )

    cluster: FilePath = Field(
        ...,
        validation_alias=AliasChoices("c", "cluster"),
        description="Path to the cluster configuration file (YAML)",
    )

    pretrain: FilePath = Field(
        ...,
        validation_alias=AliasChoices("p", "m", "pretrain", "megatron"),
        description="Path to the megatron configuration file (YAML)",
    )

    tmux: bool = Field(
        default_factory=lambda: get_session() is not None,
        description="Whether to attach to current tmux session when running the script "
        "(defaults to True iff invoked from inside a tmux session)",
    )

    daemon: bool = Field(
        False,
        description="Whether to run launched processes as daemons (only if not using tmux)",
    )

    @model_validator(mode="after")
    def check_tmux_daemon(self) -> Self:
        if self.tmux and self.daemon:
            raise ValueError("Cannot use --daemon together with --tmux")
        if self.tmux and get_session() is None:
            raise ValueError(
                "--tmux requested but no tmux session detected "
                "(env var TMUX is unset). Run sf-megarun from inside tmux, "
                "or pass --no-tmux."
            )
        return self

    def cli_cmd(self) -> None:
        setup_logging(self.log_level)

        logger.debug("Launching megarun with args: %s", self.model_dump())

        try:
            cluster = load_model(self.cluster, ClusterConfig)
            self._override_cluster_config(cluster)

            logger.debug("Using cluster config: %s", cluster.model_dump())
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cluster config from {self.cluster}"
            ) from e

        try:
            megatron_job = load_model(self.pretrain, MegatronJob)

            logger.debug("Using megatron config: %s", megatron_job.model_dump())
        except Exception as e:
            raise RuntimeError(
                f"Failed to load megatron config from {self.pretrain}"
            ) from e

        output_dir = OutputDirectory.make(cluster.output_dir / megatron_job.name)

        config = {
            "cluster": cluster.model_dump(mode="json"),
            "job": megatron_job.model_dump(mode="json"),
        }
        script_content = render_script(cluster, megatron_job, output_dir, self.port)

        output_dir.write_config(config)
        script_path = output_dir.write_script(script_content)

        launcher = make_launcher(self.tmux, self.daemon)
        for rank, node in enumerate(cluster.nodes):
            command = _build_ssh_command(
                node.ssh_target, script_path.absolute().as_posix()
            )
            logger.info(
                "Launching rank %d on node %s: %s %d",
                rank,
                node.ssh_target,
                command,
                rank,
            )
            launcher(command, rank)

        join_launched()

    def _override_cluster_config(self, config: ClusterConfig) -> None:
        if self.nnodes is not None:
            if len(config.nodes) < self.nnodes:
                raise ValueError(
                    f"Requested number of nodes ({self.nnodes}) exceeds "
                    f"available nodes in config ({len(config.nodes)})"
                )

            if len(config.nodes) > self.nnodes:
                dropped = [n.ssh_target for n in config.nodes[self.nnodes :]]
                logger.warning(
                    "Using first %d of %d configured nodes; dropping: %s",
                    self.nnodes,
                    len(config.nodes),
                    dropped,
                )
                config.nodes = config.nodes[: self.nnodes]

        if self.nproc_per_node is not None:
            for node in config.nodes:
                if node.num_gpus < self.nproc_per_node:
                    raise ValueError(
                        f"Requested number of GPUs per node ({self.nproc_per_node}) "
                        f"exceeds available GPUs on node {node.ssh_target} ({node.num_gpus})"
                    )

                node.num_gpus = self.nproc_per_node


def main() -> None:
    CliApp.run(CliArgs)


if __name__ == "__main__":
    main()
