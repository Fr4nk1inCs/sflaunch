from typing import Optional, Self, TypeVar

import yaml
from pydantic import BaseModel
from pydantic.aliases import AliasChoices
from pydantic.fields import Field
from pydantic.functional_validators import model_validator
from pydantic.types import FilePath, PositiveInt
from pydantic_settings import CliApp

from sfutils.launchers import join_launched, make_launcher
from sfutils.schemas.cli_args import BaseCliArgs
from sfutils.schemas.cluster import ClusterConfig
from sfutils.schemas.jobs.megatron import MegatronJob
from sfutils.templates.torchrun import render_script as render_torchrun_script
from sfutils.utils import OutputDirectory
from sfutils.utils.log import get_logger, setup_logging

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
    return render_torchrun_script(
        env_vars=job.env,
        script=cluster.script,
        argv=job.argv,
        working_dir=cluster.working_dir,
        log_dir=output_dir.base,
        env_setup=cluster.env_setup or "",
        nnodes=len(cluster.nodes),
        nproc_per_node=min(node.num_gpus for node in cluster.nodes),
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
        description="`--nproc_per_node` passed to torchrun, overrides cluster config",
    )

    port: PositiveInt = Field(
        29500,
        description="Port for master node communication",
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

    try_use_tmux: bool = Field(
        True,
        validation_alias=AliasChoices("tmux", "try-use-tmux"),
        description="Whether to attach to current tmux session when running the script",
    )

    daemon: bool = Field(
        default_factory=lambda data: not data["try_use_tmux"],
        description="Whether to run launched processes as daemons (only if not using tmux)",
    )

    @model_validator(mode="after")
    def check_tmux_daemon(self) -> Self:
        if self.try_use_tmux and self.daemon:
            raise ValueError("Cannot use daemon mode when try_use_tmux is True")
        return self

    def cli_cmd(self) -> None:
        setup_logging(self.log_level)

        logger.debug("Lauching megarun with args: %s", self.model_dump())

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

        launcher = make_launcher(self.try_use_tmux, self.daemon)
        for rank, node in enumerate(cluster.nodes):
            command = (
                f"ssh -t {node.ssh_target} -- bash {script_path.absolute().as_posix()}"
            )
            logger.info(
                "Launching rank %d on node %s: %s", rank, node.ssh_target, command
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
