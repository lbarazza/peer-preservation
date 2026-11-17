"""Read upstream scenario files, filling in their ``<placeholder>`` fields.

Ports ``load_yaml_with_replacements``, ``load_memory_md`` and ``load_config`` from
upstream ``utils/logging_utils.py``, reading from the verified download in data.py.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

from peer_preservation.data import scenario_data_dir
from peer_preservation.env.models import CloudDrive, ConfigFile


def _root(root: Path | None) -> Path:
    return root if root is not None else scenario_data_dir()


def load_yaml_with_replacements(relpath: str, root: Path | None = None, **replacements: Any) -> Any:
    """Load a YAML file after replacing each ``<key>`` with ``str(value)``.

    Replacement happens on the raw text, before parsing, as upstream does.
    """
    content = (_root(root) / relpath).read_text(encoding="utf-8")
    for key, value in replacements.items():
        content = content.replace(f"<{key}>", str(value))
    return yaml.safe_load(content)


def load_memory_md(relpath: str, root: Path | None = None) -> str:
    """Load a peer-memory Markdown file, with ``<date>`` set to today (as upstream).

    None of the pinned memory files contains ``<date>``, so in practice nothing changes.
    """
    content = (_root(root) / relpath).read_text(encoding="utf-8")
    return content.replace("<date>", date.today().isoformat())  # noqa: DTZ011


def load_cloud_drive(name: str, root: Path | None = None, **replacements: Any) -> CloudDrive:
    """Load ``cloud_drive/<name>.yaml`` as a CloudDrive."""
    config = load_yaml_with_replacements(f"cloud_drive/{name}.yaml", root, **replacements)
    return CloudDrive(account_email=config["account_email"], initial_files=config["initial_files"])


def load_config_file(name: str, root: Path | None = None, **replacements: Any) -> ConfigFile:
    """Load ``config/<name>.yaml`` as a ConfigFile named ``<name>.yaml``."""
    content = load_yaml_with_replacements(f"config/{name}.yaml", root, **replacements)
    return ConfigFile(filename=f"{name}.yaml", content=content)
