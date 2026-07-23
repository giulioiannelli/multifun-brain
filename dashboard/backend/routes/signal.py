"""Signal-tab routes: raw ROI timecourses (carpet / channel / EMD / bands).

``GET /api/signal/catalog`` powers the Signal selectors (incl. the Dataset
dropdown); ``GET /api/signal/{kind}`` serialises one view for a
``(dataset, subject, contrast, processing[, channel, scheme])`` selection. The
raw array is loaded here (memoised) and handed to the pure serializers.

``contrast`` is optional: the older ``kw…`` datasets have no co2/rest, so a
request may omit it (the client drops empty params), in which case it resolves
the single-condition file.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .. import timeseries
from ..serializers import signal as signal_ser

router = APIRouter()


@router.get("/signal/catalog")
def signal_catalog(dataset: str = Query(default="")) -> dict:
    """Available raw timecourses + facets for *dataset* (+ the dataset list)."""
    return timeseries.signal_catalog(dataset or None)


@router.get("/signal/cohort_bands")
def signal_cohort_bands(
    contrast: str,
    processing: str,
    dataset: str = Query(default=""),
    scheme: str = Query(default="canonical"),
    subject: str = Query(default=""),
) -> dict:
    """Cohort IMF-frequency histogram + the s5/s4/s* band ranges (Hz).

    Pools EMD over every ROI of every subject for ``(contrast, processing)`` —
    cached, but the first call takes a few seconds. ``scheme`` is ``canonical``
    (fixed slow-oscillation edges, default) or ``data_driven``. A non-empty
    ``subject`` restricts the pool to that one subject (per-subject spectrum);
    omit it for the whole-cohort spectrum.
    """
    cohort = timeseries.cohort_bands(dataset or None, contrast, processing, scheme, subject or "")
    if cohort is None:
        raise HTTPException(status_code=404, detail="no timecourses for this contrast/processing")
    return signal_ser.cohort_bands_spec(cohort)


@router.get("/signal/{kind}")
def signal_plot(
    kind: str,
    subject: str,
    processing: str,
    dataset: str = Query(default=""),
    contrast: str = Query(default=""),
    channel: int = Query(default=0),
    max_imfs: int = Query(default=10),
    scheme: str = Query(default="canonical"),
    band: str = Query(default=""),
) -> dict:
    """Serialise *kind* for one raw-timecourse selection.

    ``kind`` matches the spec/builder name: ``signal_heatmap`` (carpet),
    ``signal_channel`` (single timecourse), ``signal_emd`` (IMF decomposition),
    ``signal_bands`` (per-band reconstruction; honours ``scheme``). For
    ``signal_heatmap``, a non-empty ``band`` (``s5``/``s4``/``sstar``) shows the
    band-reconstructed carpet instead of the raw broadband one.
    """
    ds = dataset or None
    names = timeseries.region_names(ds)

    if kind == "signal_emd":
        sift = timeseries.sift_channel(ds, subject, contrast, processing, channel, max_imfs)
        if sift is None:
            raise HTTPException(status_code=404, detail="unknown timecourse selection")
        return signal_ser.emd_spec(sift, names)

    if kind == "signal_bands":
        recon = timeseries.band_reconstruction(
            ds, subject, contrast, processing, channel, scheme
        )
        if recon is None:
            raise HTTPException(status_code=404, detail="no band scheme for this selection")
        return signal_ser.band_reconstruction_spec(recon, names)

    if kind == "signal_heatmap":
        if band and band != "full":
            ts = timeseries.band_carpet(ds, subject, contrast, processing, band)
        else:
            ts = timeseries.get_timecourses(ds, subject, contrast, processing)
        if ts is None:
            raise HTTPException(status_code=404, detail="unknown timecourse selection")
        return signal_ser.heatmap_spec(ts, names, band=band or "full")

    ts = timeseries.get_timecourses(ds, subject, contrast, processing)
    if ts is None:
        raise HTTPException(status_code=404, detail="unknown timecourse selection")
    if kind == "signal_channel":
        return signal_ser.channel_spec(ts, names, channel=channel)
    raise HTTPException(status_code=404, detail=f"unknown signal kind {kind!r}")
