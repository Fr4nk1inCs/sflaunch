from typing import Dict, List

from pydantic import BaseModel, Field


class CliBasedJob(BaseModel):
    name: str = Field(
        ...,
        description="Name of the job",
    )

    env: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to set for the job",
    )

    argv: List[str] = Field(
        ...,
        description="Command-line arguments to pass to the script",
    )
