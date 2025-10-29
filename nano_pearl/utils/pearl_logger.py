import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

def get_logger(name="PEARL", level=logging.INFO):
    logging.disable(logging.NOTSET)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console = Console(force_terminal=True, theme=Theme({
            "logging.level.info": "green",
        }))

        h = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
            omit_repeated_times=False,
            log_time_format="%H:%M:%S",
            markup=True,
        )
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
        logger.propagate = False

        _orig_info = logger.info
        def _info(msg, *args, color=None, **kwargs):
            if color:
                msg = f"[{color}]{msg}[/]"
            return _orig_info(msg, *args, **kwargs)
        logger.info = _info
    return logger

logger = get_logger()


def get_model_name(model_path: str) -> str:
    l = model_path.split("/")
    for s in l:
        if s.startswith("models--"):
            return s
    logger.warning(f"Model Name Not Found: {model_path}")
    return model_path
