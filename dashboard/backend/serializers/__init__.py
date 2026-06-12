"""Serializers: turn ``PipelineResult`` / ``MultiscaleResult`` into JSON plot specs.

One module per pipeline section. A central registry (``PLOT_KINDS``) maps a plot
``kind`` string to a ``(result) -> dict`` function so routes stay declarative and
the frontend can discover available kinds.
"""

from __future__ import annotations

from collections.abc import Callable

from . import descriptive

# kind -> serializer(result) -> JSON-safe dict
PLOT_KINDS: dict[str, Callable] = {
    "heatmap": descriptive.heatmap_spec,
}
