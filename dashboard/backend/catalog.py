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
    """Resolve a dataset id to its directory, or ``None`` if invalid/outside root.

    The results root itself is **not** a dataset (only named sub-directories are;
    see :func:`list_datasets`), so an empty / ``"."`` id — or any id that resolves
    back to the root, e.g. ``"april/.."`` — is rejected, as is path traversal.
    """
    root = config.RESULTS_ROOT.resolve()
    if dataset_id in (".", ""):
        return None
    candidate = (root / dataset_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None  # path traversal attempt
    if candidate == root:
        return None  # resolved back to the root (not a named dataset)
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
        # Only named sub-directories are datasets. A bundle sitting directly at
        # the results root has no dataset/category name — it surfaced as the
        # confusing "(root)" entry and breaks the Dataset/Category structure — so
        # skip it. Move such a bundle into a named subfolder to surface it.
        if str(rel) == ".":
            continue
        ds_id = str(rel).replace("\\", "/")
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
