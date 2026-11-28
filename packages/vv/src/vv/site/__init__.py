"""What every verification and validation site shares: its assets and the records its pages fetch.

A project's site is its own page (or pages) and index builder. This gives it
the shared stylesheet and script (``vv.css``: colour tokens with dark mode,
cards, filters, tiles, the results table and detail panels, prose pages;
``vv.js``: formatting, tooltips, record fetching, Plotly theming and
``vv.explorer``, the filterable, sortable benchmark table), and writes each
published record into the site as its exact published bytes::

    from vv import site
    site.copy_assets(out)                       # out/vv.css, out/vv.js
    items = site.item_records(root, run, out)   # out/records/<blob>.json
    index = {"run": site.run_summary(run), ...}
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from ..publish import Run, record_bytes

ASSETS = ("vv.css", "vv.js")
"""The shared files a page links, relative to the site root."""


def asset_dir() -> Path:
    """The folder the shared assets are installed in."""
    return Path(__file__).resolve().parent


def copy_assets(site: Path | str) -> list[Path]:
    """Copy the shared assets into ``site``; return where they went."""
    site = Path(site)
    site.mkdir(parents=True, exist_ok=True)
    out = []
    for name in ASSETS:
        out.append(Path(shutil.copyfile(asset_dir() / name, site / name)))
    return out


def run_summary(run: Optional[Run]) -> Optional[dict]:
    """What a page says about a run: its tag, date, codes, models and libraries."""
    if run is None:
        return None
    return {"tag": run.tag, "date": run.date, "codes": list(run.codes), "models": list(run.models),
            "libraries": list(run.libraries)}


def record_path(entry) -> str:
    """Where a published record is filed in the site, by its git blob id."""
    return f"records/{entry.blob}.json"


def item_records(root: Path | str, run: Run, site: Path | str | None = None) -> dict:
    """Every record of ``run``, by ``(model, library, item)`` and then label.

    Each value is ``{"file": "records/<blob>.json", "record": <the parsed
    record>}``. The record is its exact published bytes, read back from git
    under ``root`` if it has been replaced since (:func:`vv.publish.record_bytes`),
    and when ``site`` is given those bytes are written there at ``file``, for
    the page to fetch.
    """
    root = Path(root)
    out: dict = {}
    for e in run.entries:
        data = record_bytes(root, e)
        path = record_path(e)
        if site is not None:
            dest = Path(site) / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        out.setdefault((e.model, e.library, e.item), {})[e.label] = {"file": path, "record": json.loads(data)}
    return out
