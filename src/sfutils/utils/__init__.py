import datetime
from dataclasses import dataclass
from typing import Any, Dict

import yaml
from pydantic.types import DirectoryPath, FilePath

from sfutils.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class OutputDirectory:
    base: DirectoryPath

    @classmethod
    def make(cls, base: DirectoryPath) -> "OutputDirectory":
        date = datetime.date.today().isoformat()
        timestamp = datetime.datetime.now().strftime("%H-%M-%S")

        logger.info("Current date: %s, timestamp: %s", date, timestamp)

        dir = base / date / timestamp
        dir.mkdir(parents=True, exist_ok=True)

        logger.info("Created output directory at: %s", dir.absolute().as_posix())

        return cls(base=dir)

    def write_config(self, config: Dict[str, Any]) -> FilePath:
        config_path = self.base / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(config, f)
        logger.info("Wrote config to: %s", config_path.absolute().as_posix())
        return config_path

    def write_script(self, content: str, extension: str = ".sh") -> FilePath:
        script_path = self.base / f"run{extension}"
        with script_path.open("w") as f:
            f.write(content)
        script_path.chmod(0o755)
        logger.info("Wrote script to: %s", script_path.absolute().as_posix())
        return script_path
