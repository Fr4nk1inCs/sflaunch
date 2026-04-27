import subprocess
from functools import partial
from typing import List, Protocol

from libtmux import Session

from sflaunch.utils.log import get_logger
from sflaunch.utils.tmux import get_session

logger = get_logger(__name__)


class Launcher(Protocol):
    def __call__(self, command: str, rank: int):
        """The executed command would be `{command} {rank}`"""
        ...


def tmux_launcher(command: str, rank: int, *, session: Session) -> None:
    run_and_persist = (
        f"{command} {rank}; bash -c \"read -n 1 -s -r -p 'Press any key to exit...'\""
    )
    window = session.new_window(attach=False, window_shell=run_and_persist)
    window.rename_window(f"rank-{rank}")


_processes_to_join: List[subprocess.Popen] = []


def process_launcher(command: str, rank: int, *, daemon: bool = False) -> None:
    full_command = f"{command} {rank}"
    if daemon:
        process = subprocess.Popen(
            full_command,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        logger.info("Detached rank %d as pid %d", rank, process.pid)
        return

    process = subprocess.Popen(full_command, shell=True)
    global _processes_to_join
    _processes_to_join.append(process)


def make_launcher(tmux: bool = False, daemon: bool = False) -> Launcher:
    if tmux:
        session = get_session()
        if session is None:
            raise RuntimeError("tmux launcher requested but no tmux session detected")
        logger.info("Using tmux launcher.")
        return partial(tmux_launcher, session=session)
    logger.info("Using process launcher.")
    return partial(process_launcher, daemon=daemon)


def join_launched() -> None:
    global _processes_to_join
    if not _processes_to_join:
        return
    logger.info("Waiting for launched processes to finish...")
    for process in _processes_to_join:
        process.wait()
