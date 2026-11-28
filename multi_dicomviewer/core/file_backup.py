"""Keep the previous version of a file before it is overwritten.

The three analysis files (.FullLv.json / .corotree.json / .territory.json)
keep ONE previous version as ``<name>.bak`` (user 2026-11-28) — a
mistaken overwrite can be undone by hand (rename the .bak back). The .bak
is replaced on every save; a folder scan for *.json never picks it up.
"""

from __future__ import annotations

import os
import shutil


def backup_path(path: str) -> str:
    """``<path>.bak`` (e.g. case.corotree.json → case.corotree.json.bak)."""
    return str(path) + ".bak"


def backup_existing(path: str) -> str:
    """Copy *path* (when it exists) to ``<path>.bak`` before it is written.
    Returns the .bak path, or "" when there was nothing to keep / the copy
    failed (a save never fails because of its backup)."""
    try:
        if not path or not os.path.isfile(path):
            return ""
        dst = backup_path(path)
        shutil.copy2(path, dst)
        return dst
    except OSError:
        return ""
