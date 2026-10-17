"""Locate the dataset and project directories without hardcoded machine paths."""

import os
from pathlib import Path

from fusiontrace import PROJECT_ROOT

DATASET_ENV_VAR = "LAVDF_ROOT"

DEFAULT_DATASET_CANDIDATES = [
    PROJECT_ROOT / "datasets" / "LAV-DF" / "LAV-DF",
    Path(r"C:\AI_DATA\LAV-DF\LAV-DF"),
    Path("/mnt/c/AI_DATA/LAV-DF/LAV-DF"),
]


def resolve_dataset_root(configured=None):
    """First existing directory among: config value, $LAVDF_ROOT, known defaults."""
    candidates = []
    if configured:
        candidates.append(Path(configured))
    if os.environ.get(DATASET_ENV_VAR):
        candidates.append(Path(os.environ[DATASET_ENV_VAR]))
    candidates.extend(DEFAULT_DATASET_CANDIDATES)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    tried = "\n".join(f"  {c}" for c in candidates)
    raise FileNotFoundError(
        "LAV-DF dataset directory not found. Set data.dataset_root in the "
        f"config or the {DATASET_ENV_VAR} environment variable. Tried:\n{tried}"
    )


def project_path(value):
    """Resolve a path relative to the project root unless it is absolute."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
