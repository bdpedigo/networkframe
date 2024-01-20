"""Top-level package for networkframe."""

import pkg_resources

from .networkframe import LocIndexer, NetworkFrame, NodeGroupBy

__all__ = ["NetworkFrame", "NodeGroupBy", "LocIndexer"]

__version__ = pkg_resources.get_distribution("networkframe").version
