import os
from functools import partial
from multiprocessing import Process
from typing import List, Protocol

from libtmux import Session

from sfutils.utils.log import get_logger
from sfutils.utils.tmux import get_session

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


_processes_to_join: List[Process] = []


def process_launcher(command: str, rank: int, *, daemon: bool = False) -> None:
    process = Process(target=lambda: os.system(f"{command} {rank}"))
    process.daemon = daemon
    process.start()

    if not daemon:
        global _processes_to_join
        _processes_to_join.append(process)


def make_launcher(try_use_tmux: bool = False, daemon: bool = False) -> Launcher:
    if try_use_tmux:
        session = get_session()
        if session is not None:
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
        process.join()
