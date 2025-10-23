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

import xml.etree.ElementTree as ET

from contextlib import suppress
from nilearn import datasets, plotting, image
from pathlib import Path

from nilearn.datasets import load_mni152_template
from scipy.linalg import eigh
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from multifunbrain.analysis import *
from multifunbrain.visualization.plotlib import *
