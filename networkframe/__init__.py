"""Top-level package for networkframe."""

from importlib.metadata import version

from .networkframe import LocIndexer, NetworkFrame, NodeGroupBy
from .algorithms import aggregate_over_graph

__all__ = ["NetworkFrame", "NodeGroupBy", "LocIndexer", "aggregate_over_graph"]

__version__ = version("networkframe")
