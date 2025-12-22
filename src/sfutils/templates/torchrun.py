from string import Template
from textwrap import dedent
from typing import Dict, List, TypedDict, Unpack

from pydantic.types import DirectoryPath, FilePath

from sfutils.utils.formatters import argv2cmd, fmt_env_vars

SCRIPT_TEMPLATE = Template(
    dedent(
        """
        #!/usr/bin/env bash

        cd ${working_dir}

        ${env_setup}

        _NODE_RANK=${1:-0}

        torchrun \\
            --nnodes=${nnodes} \\
            --nproc_per_node=${nproc_per_node} \\
            --node_rank=${_NODE_RANK} \\
            --master_addr=${master_addr} \\
            --master_port=${master_port} \\
            ${command} \\
            2>&1 | tee ${log_dir}/node-${_NODE_RANK}.log

        read -n 1 -s -r -p "Press any key to continue"
        """
    ).strip()
)


class RenderScriptKwargs(TypedDict):
    env_setup: str
    nnodes: int
    nproc_per_node: int
    master_addr: str
    master_port: int


def render_script(
    *,
    env_vars: Dict[str, str],
    script: FilePath,
    argv: List[str],
    working_dir: DirectoryPath,
    log_dir: DirectoryPath,
    **kwargs: Unpack[RenderScriptKwargs],
) -> str:
    kwargs["env_setup"] = f"{kwargs['env_setup']}\n\n{fmt_env_vars(env_vars)}"

    return SCRIPT_TEMPLATE.safe_substitute(
        command=argv2cmd(script.absolute().as_posix(), argv, indent=4),
        working_dir=working_dir.absolute().as_posix(),
        log_dir=log_dir.absolute().as_posix(),
        **kwargs,
    )
