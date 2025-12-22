import logging

_LOGGING_SETUP: bool = False


def setup_logging(level: int | str | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)5s] %(message)s",
        level=level or logging.INFO,
        force=True,
    )

    global _LOGGING_SETUP
    _LOGGING_SETUP = True


def get_logger(name: str) -> logging.Logger:
    if not _LOGGING_SETUP:
        setup_logging()

    return logging.getLogger(name)
