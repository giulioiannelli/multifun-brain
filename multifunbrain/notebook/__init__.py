"""Convenience re-exports for interactive notebook sessions.

This submodule collects imports that are repeatedly used across the project's
Jupyter notebooks so that each notebook can stay tidy. Extend or trim the list
below to fit your workflow.

Typical usage::

    from multifunbrain.notebook import *
    # os, Path, json, logging, np, ...
"""

from __future__ import annotations

import emd
import json
import logging
import os
import re

import networkx as nx
import numpy as np
import pandas as pd
import pickle as pk
import seaborn as sns


import plotly.graph_objects as go
import xml.etree.ElementTree as ET

from contextlib import suppress
from nilearn import datasets, plotting, image
from pathlib import Path
from matplotlib import gridspec

from nilearn.datasets import load_mni152_template
from scipy.linalg import eigh
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, \
    set_link_color_palette
from sklearn.metrics.cluster import contingency_matrix

from multifunbrain.analysis import *
from multifunbrain.visualization.plotlib import *
from multifunbrain.pipeline import (
    PipelineConfig as PipelineConfig,
    load_results as load_results,
    run_pipeline as run_pipeline,
    run_pipeline_batch as run_pipeline_batch,
    run_pipeline_directory as run_pipeline_directory,
)