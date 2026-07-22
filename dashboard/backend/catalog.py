"""Discovery: scan ``RESULTS_ROOT`` for result bundles and expose their facets.

A *dataset* is any directory under ``RESULTS_ROOT`` containing a ``results.pkl``.
Each result's label (``level/contrast_proc[/band[/...]]``) is parsed into facets
that power the dropdowns. ``dataset_dir`` resolves a dataset id back to a path,
guarding against traversal outside ``RESULTS_ROOT``.
"""

from __future__ import annotations

from pathlib import Path

from . import config
from .facets import parse_variant
from .loaders import get_results


def parse_label(label: str) -> dict:
    """Split a pipeline label into the facet fields that power the selectors.

    Labels are ``level/<variant>[/<band>]`` or, for patients,
    ``patient/<subject>/<variant>[/<band>]``, where ``<variant>`` is
    ``[<task>_]<token>``. The variant is decomposed into three axes via
    :func:`facets.parse_variant`: **task** (co2/rest — a genuine prefix only),
    **contrast** (modality parsed from the token: bold/vaso/cbf) and
    **processing** (the pipeline: clean/optcom/optcomMIRdenoised/bpf/raw/…). So a
    kw token like ``clean_kwCBF4D`` becomes task=None, contrast=cbf,
    processing=clean, while ``co2_bpfVASO`` becomes task=co2, contrast=vaso,
    processing=bpf.
    """
    out: dict[str, str | None] = {
        "level": None,
        "task": None,
        "contrast": None,
        "processing": None,
        "band": None,
        "subject": None,
    }
    if not label:
        return out
    parts = label.split("/")
    level = parts[0]
    out["level"] = level

    if level == "global" and len(parts) >= 2:
        out["task"], out["contrast"], out["processing"] = parse_variant(parts[1])
    elif level == "band" and len(parts) >= 3:
        out["task"], out["contrast"], out["processing"] = parse_variant(parts[1])
        out["band"] = parts[2]
    elif level == "patient" and len(parts) >= 3:
        out["subject"] = parts[1]
        out["task"], out["contrast"], out["processing"] = parse_variant(parts[2])
        if len(parts) >= 4:
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


_FACET_KEYS = ("level", "task", "contrast", "processing", "band", "subject")


def _known_dataset_ids() -> set[str]:
    """Raw-dataset ids the Signal tab offers (result tabs are a subset of these)."""
    from .timeseries import _dataset_id, _raw_data_dirs

    return {_dataset_id(p) for p in _raw_data_dirs()}


def list_datasets() -> list[dict]:
    """All discoverable datasets with their items and aggregated facets.

    A dataset is a named sub-directory of the results root containing a
    ``results.pkl``, whose **top-level** folder matches one of the known raw
    datasets (so the result-tab dataset list mirrors the Signal tab and excludes
    stray/legacy bundles). The filter is skipped when no raw datasets are present
    (e.g. a results-only checkout), so nothing is hidden spuriously.
    """
    root = config.RESULTS_ROOT
    datasets: list[dict] = []
    if not root.is_dir():
        return datasets

    known = _known_dataset_ids()
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
        top = ds_id.split("/")[0]
        if known and top not in known:
            continue  # not one of the recognised raw datasets
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
