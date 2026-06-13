"""Signal-tab routes: raw ROI timecourses (carpet / channel / EMD).

``GET /api/signal/catalog`` powers the Signal selectors; ``GET /api/signal/{kind}``
serialises one of ``heatmap`` / ``channel`` / ``emd`` for a
``(subject, contrast, processing[, channel])`` selection. The raw array is loaded
here (memoised) and handed to the pure serializers.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .. import timeseries
from ..serializers import signal as signal_ser

router = APIRouter()


@router.get("/signal/catalog")
def signal_catalog() -> dict:
    """Available raw timecourses + facets (subjects, contrasts, processings)."""
    return timeseries.signal_catalog()


@router.get("/signal/cohort_bands")
def signal_cohort_bands(contrast: str, processing: str) -> dict:
    """Cohort IMF-frequency histogram + data-driven s5/s4/s* band ranges.

    Pools EMD over every ROI of every subject for ``(contrast, processing)`` —
    cached, but the first call takes a few seconds.
    """
    cohort = timeseries.cohort_bands(contrast, processing)
    if cohort is None:
        raise HTTPException(status_code=404, detail="no timecourses for this contrast/processing")
    return signal_ser.cohort_bands_spec(cohort)


@router.get("/signal/{kind}")
def signal_plot(
    kind: str,
    subject: str,
    contrast: str,
    processing: str,
    channel: int = Query(default=0),
    max_imfs: int = Query(default=10),
) -> dict:
    """Serialise *kind* for one raw-timecourse selection.

    ``kind`` matches the spec/builder name: ``signal_heatmap`` (carpet),
    ``signal_channel`` (single timecourse), ``signal_emd`` (IMF decomposition).
    """
    names = timeseries.region_names()

    if kind == "signal_emd":
        sift = timeseries.sift_channel(subject, contrast, processing, channel, max_imfs)
        if sift is None:
            raise HTTPException(status_code=404, detail="unknown timecourse selection")
        return signal_ser.emd_spec(sift, names)

    if kind == "signal_bands":
        recon = timeseries.band_reconstruction(subject, contrast, processing, channel)
        if recon is None:
            raise HTTPException(status_code=404, detail="no band scheme for this selection")
        return signal_ser.band_reconstruction_spec(recon, names)

    ts = timeseries.get_timecourses(subject, contrast, processing)
    if ts is None:
        raise HTTPException(status_code=404, detail="unknown timecourse selection")
    if kind == "signal_heatmap":
        return signal_ser.heatmap_spec(ts, names)
    if kind == "signal_channel":
        return signal_ser.channel_spec(ts, names, channel=channel)
    raise HTTPException(status_code=404, detail=f"unknown signal kind {kind!r}")
