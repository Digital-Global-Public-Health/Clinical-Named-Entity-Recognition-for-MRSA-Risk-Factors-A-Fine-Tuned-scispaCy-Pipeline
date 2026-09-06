# src/utils_logging.py
"""Logging utilities for the MRSA NLP NER-based pipeline."""

import logging
import functools
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("mrsa_nlp.ner")

LOG_RUN_DIR: Optional[Path] = None  # cache the active run directory


def configure_logging(
    level: str = "INFO",
    run_name: Optional[str] = None,
    base_dir: Path = Path("outputs"),
) -> Path:
    """
    Configure root logging once per process.

    Creates:
      - A console (stderr) handler
      - A file handler at outputs/<run_name>_<timestamp>/run.log

    Parameters
    ----------
    level : str
        Python logging level string (DEBUG, INFO, WARNING, ERROR).
    run_name : str, optional
        Prefix for the timestamped run directory. Defaults to "session".
    base_dir : Path
        Base directory for all run output folders.

    Returns
    -------
    Path
        The newly created (or previously cached) run directory.

    Notes
    -----
    Calling this function more than once in the same process is a no-op —
    the cached run directory is returned unchanged.
    """
    global LOG_RUN_DIR

    root = logging.getLogger()
    if root.handlers and LOG_RUN_DIR is not None:
        numeric = getattr(logging, level.upper(), logging.INFO)
        root.setLevel(numeric)
        logger.info("Logging already configured; reusing run dir %s", LOG_RUN_DIR)
        return LOG_RUN_DIR

    if run_name is None:
        run_name = "session"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    numeric = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric)

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(numeric)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(log_path)
    fh.setLevel(numeric)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    LOG_RUN_DIR = run_dir
    logger.info("Logging initialized: level=%s, file=%s", level.upper(), log_path)
    return run_dir


def make_run_dir(
    prefix: str = "run",
    base_dir: Path = Path("outputs"),
) -> Tuple[str, Path]:
    """
    Create a generic timestamped run directory: outputs/<prefix>_<YYYYMMDD-HHMMSS>.

    Parameters
    ----------
    prefix : str
        Directory name prefix (e.g. "cohort_builder", "ner_training").
    base_dir : Path
        Parent output directory.

    Returns
    -------
    Tuple[str, Path]
        (run_id, run_dir) where run_id is the directory name and run_dir is
        the full Path.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{prefix}_{ts}"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def save_config_snapshot(cfg: Dict[str, Any], run_dir: Path, fname: str = "config.yaml") -> None:
    """
    Persist a YAML snapshot of a config dictionary into the run directory.

    Parameters
    ----------
    cfg : dict
        Configuration key-value pairs (must be YAML-serialisable).
    run_dir : Path
        Destination directory (must already exist).
    fname : str
        Output filename; defaults to "config.yaml".
    """
    import yaml

    def _yaml_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: _yaml_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_yaml_safe(item) for item in value]
        return value

    path = run_dir / fname
    with open(path, "w") as f:
        yaml.safe_dump(_yaml_safe(cfg), f, sort_keys=False)
    logger.info("Saved config snapshot → %s", path)


def log_timing(fn):
    """
    Decorator that logs the start, completion time, and any exception of a function.

    Logs the output shape if the return value has a `.shape` attribute
    (e.g. pandas DataFrames).
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        fn_name = fn.__name__
        logger.info("▶  %s: start", fn_name)
        logger.debug("%s args=%s kwargs=%s", fn_name, args, kwargs)
        ts = datetime.now().timestamp()
        try:
            out = fn(*args, **kwargs)
            dt = datetime.now().timestamp() - ts
            if hasattr(out, "shape"):
                try:
                    shape = out.shape
                except Exception:
                    shape = "n/a"
                logger.info("✔  %s: done in %.2fs (shape=%s)", fn_name, dt, shape)
            else:
                logger.info("✔  %s: done in %.2fs", fn_name, dt)
            return out
        except Exception:
            logger.exception("✘  %s: failed", fn_name)
            raise
    return wrapper
