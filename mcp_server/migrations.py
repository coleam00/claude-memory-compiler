"""List StarBot SQL migrations and compute the next migration number."""

from __future__ import annotations

import re
from pathlib import Path

from mcp_server import settings

_NUM = re.compile(r"^(\d{3})_.*\.sql$")


def list_migrations(migrations_dir: Path | None = None) -> dict:
    """Return ordered NNN_*.sql migration files plus latest/next numbers."""
    directory = migrations_dir or settings.MIGRATIONS_DIR
    if not directory.exists():
        return {"files": [], "latest_number": None, "next_number": "000",
                "error": f"migrations dir not found: {directory}"}

    numbered: list[tuple[int, str]] = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        m = _NUM.match(entry.name)
        if m:
            numbered.append((int(m.group(1)), entry.name))

    numbered.sort()
    files = [name for _, name in numbered]
    latest = numbered[-1][0] if numbered else None
    next_num = f"{(latest + 1):03d}" if latest is not None else "000"
    return {"files": files, "latest_number": latest, "next_number": next_num}
