"""Convenience re-exports for interactive notebook sessions.

This module is **deliberately wildcard** — its whole purpose is to let a
notebook write one line at the top::

    from multifunbrain.notebook import *

and get the full toolkit (numpy, pandas, networkx, scipy clustering,
sklearn metrics, plotly/matplotlib, nilearn, plus every public symbol of
``multifunbrain.analysis``, ``multifunbrain.visualization.plotlib``, and
the pipeline helpers) without a 20-line import preamble.

This is **not an anti-pattern in this file**: it is the explicit
ergonomic contract for interactive use. Do **not** replace these
``from X import *`` lines with explicit imports — instead curate the
``__all__`` lists in the source modules so the wildcards land an
intentional, audited set of names. The Ruff config silences F401/F403
specifically here via ``[tool.ruff.lint.per-file-ignores]``.

Library code, CLI code, scripts, and tests must use explicit imports
from canonical homes; this convenience namespace is for notebooks only.
"""

from __future__ import annotations

import json
import logging
import os
import pickle as pk
import re
import xml.etree.ElementTree as ET
from contextlib import suppress
from pathlib import Path

import emd
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import gridspec
from nilearn import datasets, image, plotting
from nilearn.datasets import load_mni152_template
from scipy.cluster.hierarchy import (
    dendrogram,
    fcluster,
    linkage,
    set_link_color_palette,
)
from scipy.linalg import eigh
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)
from sklearn.metrics.cluster import contingency_matrix

from multifunbrain.analysis import *  # noqa: F401,F403
from multifunbrain.pipeline import (  # noqa: F401
    PipelineConfig,
    load_results,
    run_pipeline,
    run_pipeline_batch,
    run_pipeline_directory,
)
from multifunbrain.visualization.plotlib import *  # noqa: F401,F403
