"""Variant-token facet parsing shared by both catalog systems (Signal + results).

A processing *token* — the AFNI ``desc`` tag or the older ``kw`` modality tag
(e.g. ``bpfBOLD``, ``clean_kwCBF4D``, ``kwoptcomMIRDenoised_bold``) — encodes two
independent facets:

* the imaging **contrast** / modality: ``bold`` (oxygenation), ``vaso`` (blood
  volume), or ``cbf`` (ASL flow);
* the **processing** pipeline: ``clean`` / ``optcom`` / ``optcomMIRdenoised`` /
  ``bpf`` / ``raw`` / a reconstruction id (``furN`` / ``fcurN``) / a space
  (``MNI152``).

The experimental **task** (``co2`` / ``rest``) is carried separately — it is a
BIDS ``task-`` field in the filename / a label prefix, *not* inside this token.
These helpers split a token into ``(contrast, processing)`` so the Signal tab and
the result tabs expose the same three axes (task · contrast · processing).

The mapping is intentionally centralised here (one source of truth) and is easy
to extend when a new variant appears: add a modality to :data:`_MODALITIES` or a
rule to :data:`_PROCESSING_RULES`.
"""

from __future__ import annotations

# Modality substrings = the acquisition contrast. MIR-derived BOLD series
# (``optcomMIRDenoised_bold`` = the denoised signal, ``MIRNoise_bold`` = the noise
# that MIR removed) are both **BOLD**; they differ by *processing*, not contrast,
# so matching ``bold`` on the ``_bold`` suffix is exactly right.
_MODALITIES: tuple[str, ...] = ("cbf", "vaso", "bold")

# Ordered (first-match-wins) processing rules — ``(needle_lowercase, value)`` —
# most specific first (``optcomMIRDenoised`` before ``optcom``; a ``clean_`` prefix
# wins over the underlying reconstruction). The values are distinct within any one
# (dataset, task, contrast) group for the current data.
_PROCESSING_RULES: tuple[tuple[str, str], ...] = (
    ("clean", "clean"),
    ("optcommirdenoised", "optcomMIRdenoised"),
    ("optcom", "optcom"),
    ("mirnoise", "MIRnoise"),
    ("bpf", "bpf"),
    ("fcurn", "fcurN"),
    ("furn", "furN"),
    ("mni152", "MNI152"),
)

# Uppercased display labels for the modality contrasts.
CONTRAST_LABELS: dict[str, str] = {
    "bold": "BOLD",
    "vaso": "VASO",
    "cbf": "CBF",
}

_KNOWN_TASKS: frozenset[str] = frozenset({"co2", "rest"})


def detect_contrast(token: str) -> str | None:
    """Imaging contrast (modality) for a variant token, or ``None`` if unknown.

    ``Contrast`` is the acquisition modality (BOLD / VASO / CBF). The MIR-derived
    BOLD series — ``optcomMIRDenoised_bold`` (denoised signal) and ``MIRNoise_bold``
    (the removed noise) — are both **BOLD**, distinguished by their *processing*
    (``optcomMIRdenoised`` vs ``MIRnoise``), not by contrast.
    """
    low = token.lower()
    for modality in _MODALITIES:
        if modality in low:
            return modality
    return None


def normalize_processing(token: str) -> str:
    """Processing-pipeline label for a variant token (``raw`` if unrecognised)."""
    low = token.lower()
    for needle, value in _PROCESSING_RULES:
        if needle in low:
            return value
    return "raw"


def split_variant(token: str) -> tuple[str | None, str]:
    """Split a variant token into ``(contrast, processing)``."""
    return detect_contrast(token), normalize_processing(token)


def peel_task(variant: str) -> tuple[str | None, str]:
    """Peel a leading ``co2_`` / ``rest_`` task off a label variant.

    Returns ``(task, remaining_token)``. Only a genuine task prefix is peeled, so
    a ``kw`` token (``clean_kwCBF4D``) keeps its full name with ``task=None``.
    """
    head, _sep, rest = variant.partition("_")
    if head in _KNOWN_TASKS and rest:
        return head, rest
    return None, variant


def parse_variant(variant: str) -> tuple[str | None, str | None, str]:
    """Full ``(task, contrast, processing)`` for a label/filename variant token."""
    task, token = peel_task(variant)
    contrast, processing = split_variant(token)
    return task, contrast, processing
