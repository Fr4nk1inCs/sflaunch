import shlex
from typing import Dict, List


def fmt_env_vars(env_vars: Dict[str, str]) -> str:
    return "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in env_vars.items()
    )


def argv2cmd(arg0: str | None, argv: List[str], indent: int = 0) -> str:
    delimiter = " \\\n" + " " * indent
    if arg0 is not None:
        argv = [arg0] + argv
    return delimiter.join(argv)
