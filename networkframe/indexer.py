import pandas as pd

from .networkframe import NetworkFrame


class LocIndexer:
    """A class for indexing a NetworkFrame using .loc."""

    def __init__(self, frame):
        """Indexer for NetworkFrame."""
        self._frame = frame

    def __getitem__(self, args):
        """Return a NetworkFrame with the given labels."""
        if isinstance(args, tuple):
            if len(args) != 2:
                raise ValueError("Must provide at most two indexes.")
            else:
                row_index, col_index = args
        else:
            raise NotImplementedError("Currently only accepts dual indexing.")

        if isinstance(row_index, int):
            row_index = [row_index]
        if isinstance(col_index, int):
            col_index = [col_index]

        if isinstance(row_index, slice):
            row_index = self._frame.nodes.index[row_index]
        if isinstance(col_index, slice):
            col_index = self._frame.nodes.index[col_index]

        row_index = pd.Index(row_index)
        col_index = pd.Index(col_index)

        source_nodes = self._frame.nodes.loc[row_index]
        target_nodes = self._frame.nodes.loc[col_index]

        edges = self._frame.edges.query(
            "source in @source_nodes.index and target in @target_nodes.index"
        )

        if row_index.equals(col_index):
            nodes = source_nodes
            return NetworkFrame(
                nodes,
                edges,
                directed=self._frame.directed,
            )
        else:
            nodes = pd.concat([source_nodes, target_nodes], copy=False, sort=False)
            nodes = nodes.loc[~nodes.index.duplicated(keep="first")]
            return NetworkFrame(
                nodes,
                edges,
                directed=self._frame.directed,
                sources=row_index,
                targets=col_index,
            )
