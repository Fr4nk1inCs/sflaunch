from typing import Literal

from pydantic.fields import Field
from pydantic.main import BaseModel
from pydantic_settings import SettingsConfigDict

LOGGING_LEVELS = Literal[
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "NOTSET",
]


class BaseCliArgs(BaseModel):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case="all",
        cli_implicit_flags=True,
        cli_hide_none_type=True,
        cli_enforce_required=True,
    )

    log_level: str = Field(
        "INFO",
        description="Logging level for the application",
    )
