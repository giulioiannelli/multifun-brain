"""Convenience re-exports for interactive notebook sessions.

This submodule collects imports that are repeatedly used across the project's
Jupyter notebooks so that each notebook can stay tidy. Extend or trim the list
below to fit your workflow.

Typical usage::

    from multifunbrain.notebook import *
    # os, Path, json, logging, np, ...
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
