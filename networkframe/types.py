"""Special types for the module."""

from typing import Union

import networkx as nx
import numpy as np
import pandas as pd

Index = Union[pd.Index, pd.MultiIndex, np.ndarray, list]

NetworkXGraph = Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
