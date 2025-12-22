import os
from typing import Optional

from libtmux import Server, Session
from libtmux._internal.query_list import MultipleObjectsReturned, ObjectDoesNotExist

from sfutils.utils.log import get_logger

logger = get_logger(__name__)


def get_session() -> Optional[Session]:
    TMUX = os.environ.get("TMUX")
    if TMUX is None:
        logger.warning("Not running inside a tmux session.")
        return

    socket, _, session_id = TMUX.split(",")
    logger.info(f"Detected tmux session id: {session_id}")

    server = Server(socket_path=socket)
    try:
        session = server.sessions.get(session_id=f"${session_id}")
        if not session:
            logger.warning(f"Tmux session with id '${session_id}' does not exist.")
        return session
    except ObjectDoesNotExist:
        logger.warning(f"Tmux session with id '${session_id}' does not exist.")
    except MultipleObjectsReturned:
        logger.warning(f"Multiple tmux sessions found with the id '${session_id}'.")
