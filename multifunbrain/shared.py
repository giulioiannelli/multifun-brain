import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.linalg import expm
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import squareform
from typing import List, Tuple