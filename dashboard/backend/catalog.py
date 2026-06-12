"""Discovery: scan ``RESULTS_ROOT`` for result bundles and expose their facets.

A *dataset* is any directory under ``RESULTS_ROOT`` containing a ``results.pkl``.
Each result's label (``level/contrast_proc[/band[/...]]``) is parsed into facets
that power the dropdowns. ``dataset_dir`` resolves a dataset id back to a path,
guarding against traversal outside ``RESULTS_ROOT``.
"""

from __future__ import annotations

from pathlib import Path

from . import config
from .loaders import get_results


def parse_label(label: str) -> dict:
    """Split an April-style label into facet fields (tolerant of other shapes)."""
    out: dict[str, str | None] = {
        "level": None,
        "contrast": None,
        "processing": None,
        "band": None,
        "subject": None,
    }
    if not label:
        return out
    parts = label.split("/")
    out["level"] = parts[0]

    def split_proc(tag: str) -> tuple[str | None, str | None]:
        if "_" in tag:
            contrast, processing = tag.split("_", 1)
            return contrast, processing
        return None, tag

    level = parts[0]
    if level == "global" and len(parts) >= 2:
        out["contrast"], out["processing"] = split_proc(parts[1])
    elif level == "band" and len(parts) >= 3:
        out["contrast"], out["processing"] = split_proc(parts[1])
        out["band"] = parts[2]
    elif level == "patient" and len(parts) >= 4:
        out["subject"] = parts[1]
        out["contrast"], out["processing"] = split_proc(parts[2])
        out["band"] = parts[3]
    return out


def dataset_dir(dataset_id: str) -> Path | None:
    """Resolve a dataset id to its directory, or ``None`` if invalid/outside root."""
    root = config.RESULTS_ROOT.resolve()
    candidate = (root if dataset_id in (".", "") else root / dataset_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None  # path traversal attempt
    if not (candidate / "results.pkl").is_file():
        return None
    return candidate


_FACET_KEYS = ("level", "contrast", "processing", "band", "subject")


def list_datasets() -> list[dict]:
    """All discoverable datasets with their items and aggregated facets."""
    root = config.RESULTS_ROOT
    datasets: list[dict] = []
    if not root.is_dir():
        return datasets

    for pkl in sorted(root.rglob("results.pkl")):
        directory = pkl.parent
        rel = directory.relative_to(root)
        ds_id = "." if str(rel) == "." else str(rel).replace("\\", "/")
        try:
            rc = get_results(directory)
        except Exception as exc:  # noqa: BLE001 - surface, don't crash discovery
            datasets.append({"id": ds_id, "error": str(exc), "items": []})
            continue

        items: list[dict] = []
        facets: dict[str, set] = {k: set() for k in (*_FACET_KEYS, "filter")}
        for r in rc:
            meta = parse_label(r.label or "")
            filters = list(r.network_analyses.keys())
            items.append(
                {"label": r.label, **meta, "filters": filters, "error": r.error}
            )
            for k in _FACET_KEYS:
                if meta.get(k):
                    facets[k].add(meta[k])
            facets["filter"].update(filters)

        datasets.append(
            {
                "id": ds_id,
                "path": str(directory),
                "n_results": len(rc),
                "items": items,
                "facets": {k: sorted(v) for k, v in facets.items()},
            }
        )
    return datasets
