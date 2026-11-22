"""A canonical, comparable form of an environment, and its fingerprint.

Used by the parity test and by ``validation/phase2/export_upstream_environments.py``,
which loads this file directly (standard library only) to fingerprint upstream's
environments the same way.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _plain(value: Any) -> Any:
    """Make one spreadsheet cell comparable: pandas NA -> None, numpy scalars -> Python."""
    if type(value).__name__ == "NAType":
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def canonical_json(
    tools: list[str],
    components: dict[str, dict[str, Any] | None],
    sheets: dict[str, dict[str, Any]] | None,
    active_sheet: str | None,
) -> str:
    """Serialise an environment deterministically.

    Args:
        tools: Tool names in the order the AI is offered them.
        components: ``model_dump(mode="json")`` of cloud_drive, config_file, servers and
            memory (``None`` when the scenario has no such component).
        sheets: Sheet name -> ``{"columns": [...], "rows": [[cell, ...], ...]}``, cells in
            column order; ``None`` when the scenario has no spreadsheet.
        active_sheet: The workbook's active sheet.
    """
    excel = None
    if sheets is not None:
        excel = {
            "active_sheet": active_sheet,
            "sheets": {
                name: {
                    "columns": list(sheet["columns"]),
                    "rows": [[_plain(v) for v in row] for row in sheet["rows"]],
                }
                for name, sheet in sheets.items()
            },
        }
    payload = {"tools": tools, "components": components, "excel": excel}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def fingerprint(canonical: str) -> str:
    """Short SHA-256 of a canonical form."""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
