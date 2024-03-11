"""Top-level package for networkframe."""

from importlib.metadata import version

from .networkframe import LocIndexer, NetworkFrame, NodeGroupBy

__all__ = ["NetworkFrame", "NodeGroupBy", "LocIndexer"]

__version__ = version("networkframe")
