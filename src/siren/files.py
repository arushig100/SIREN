"""Read the unreadable-scenes list, hash files, parse counts and expand image paths."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


def read_unreadable_scenes(path: str | Path) -> frozenset[str]:
    """Scene IDs in an unreadable-scenes file such as assets/metadata/unreadable_scenes.json.

    The file lists validation scenes whose images cannot be read; training and
    calibration skip them. Test scenes are never skipped."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("split") != "val":
        raise ValueError(f"{path}: unreadable scenes must be validation scenes (split 'val')")
    ids = data.get("excluded_chip_ids")
    if (not isinstance(ids, list) or not ids
            or not all(isinstance(i, str) and i and i == i.strip() for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError(f"{path}: excluded_chip_ids must be a list of unique scene IDs")
    return frozenset(ids)


def expand_data_path(value: str, data_root: str | Path | None = None) -> str:
    """Replace ${DM3_ROOT} in a split-file path with data_root or the DM3_ROOT variable,
    then expand any other environment variables.

    An empty value stays empty. Raise ValueError if ${DM3_ROOT} is present and no root is set."""
    if not value:
        return value
    root = str(data_root) if data_root is not None else os.environ.get("DM3_ROOT")
    if "${DM3_ROOT}" in value:
        if not root:
            raise ValueError("Set DM3_ROOT to the downloaded DisasterM3 folder (see EXPERIMENTS.md)")
        value = value.replace("${DM3_ROOT}", root)
    return os.path.expandvars(value)


def require_file(path: str) -> str:
    """Return path if it is an existing file; otherwise say how to fix DM3_ROOT."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Image not found: {path}. Check that DM3_ROOT points to the "
                                "DisasterM3 folder with DisasterM3_Bench/ and DisasterM3_Instruct/")
    return path


def sha256(path: str | Path) -> str:
    """SHA-256 of a file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer_count(value: str | float) -> int:
    """Parse a count: '2' and '2.0' give 2; Booleans, negatives and fractions are rejected."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid count: {value!r}") from exc
    if isinstance(value, bool) or not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"Invalid count: {value!r}")
    return int(number)
