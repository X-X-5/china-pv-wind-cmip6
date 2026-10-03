"""Centralized project-path resolution.

All active scripts import this module to locate inputs and outputs instead of
hard-coding ``<project-root>\\...`` paths or assuming they live inside
specific legacy working directories.

Every path in ``config/project_paths.json`` is stored relative to the project
root (the directory that contains this ``scripts/`` directory).  Scripts may
still override any path via their command-line arguments.
"""
from __future__ import annotations

import json
from pathlib import Path

# scripts/common/project_paths.py -> parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "project_paths.json"

with CONFIG_PATH.open("r", encoding="utf-8") as _fh:
    _CONFIG = json.load(_fh)


def get_path(name: str) -> Path:
    """Resolve a named path from the central configuration.

    ``name`` is one of the keys under ``paths`` in ``project_paths.json``
    (e.g. ``"data_interim_cmip6_china_clipped"``), or the special key
    ``"china_shapefile"``.
    """
    if name == "china_shapefile":
        return (PROJECT_ROOT / _CONFIG["china_shapefile"]).resolve()
    return (PROJECT_ROOT / _CONFIG["paths"][name]).resolve()


def project_root() -> Path:
    """Return the resolved project root directory."""
    return PROJECT_ROOT.resolve()


def models() -> list[str]:
    """Return the canonical 17-model list."""
    return list(_CONFIG["models"])


def scenarios() -> list[str]:
    """Return the canonical 3-scenario list."""
    return list(_CONFIG["scenarios"])


def variables() -> list[str]:
    """Return the canonical 3-variable list."""
    return list(_CONFIG["variables"])
