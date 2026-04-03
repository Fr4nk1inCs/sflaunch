import json

from pydantic.aliases import AliasChoices
from pydantic.fields import Field
from pydantic.types import DirectoryPath
from pydantic_settings import BaseSettings

from sflaunch.schemas.cluster import ClusterConfig
from sflaunch.schemas.jobs.megatron import MegatronJob

MAPPING = {
    "cluster": ClusterConfig,
    "megatron-job": MegatronJob,
}


class CliArgs(BaseSettings, cli_parse_args=True):
    output: DirectoryPath = Field(
        ...,
        validation_alias=AliasChoices("output", "o"),
        description="Directory to save generated schema files",
    )


if __name__ == "__main__":
    args = CliArgs()

    for name, model in MAPPING.items():
        schema = model.model_json_schema()
        with open(args.output / f"{name}.schema.json", "w") as f:
            json.dump(schema, f, indent=4)
