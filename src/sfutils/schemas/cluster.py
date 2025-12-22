from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic.networks import IPvAnyAddress
from pydantic.types import DirectoryPath, FilePath


class Node(BaseModel):
    ip_addr: IPvAnyAddress = Field(
        ...,
        description="IP address of the node for inter-node communication",
    )

    ssh_target: str = Field(
        default_factory=lambda data: data["ip_addr"],
        description="SSH target (address or hostname) for connecting to the node",
    )

    num_gpus: int = Field(
        8,
        description="Number of GPUs available on this node",
    )


class ClusterConfig(BaseModel):
    nodes: List[Node] = Field(
        ...,
        description="List of nodes participating in the distributed job",
    )

    working_dir: DirectoryPath = Field(
        ...,
        description="Working directory on each node where the job will be executed",
    )

    env_setup: Optional[str] = Field(
        None,
        description="Command to activate the environment (e.g., source venv/bin/activate)",
    )

    output_dir: DirectoryPath = Field(
        ...,
        description="Base directory to store logs/scripts/config from the job. "
        "The final directory will be output_dir/name/date/time",
    )

    script: FilePath = Field(
        ...,
        description="Path to the script to be executed",
    )
