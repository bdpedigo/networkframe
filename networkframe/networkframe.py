"""Classes for representing networks and metadata."""


import copy
from collections.abc import Iterator
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from beartype import beartype
from scipy.sparse import csr_array

from .groupby import NodeGroupBy
from .indexer import LocIndexer

AxisType = Union[
    Literal[0], Literal[1], Literal["index"], Literal["columns"], Literal["both"]
]

EdgeAxisType = Union[Literal["source"], Literal["target"], Literal["both"]]

ColumnsType = Union[list, str]

NetworkFrameReturn = Union["NetworkFrame", None]


class NetworkFrame:
    """Represent a network as a pair of dataframes, one for nodes and one for edges.

    Parameters
    ----------
    nodes : pd.DataFrame
        Table of node attributes, the node IDs are assumed to be the index
    edges : pd.DataFrame
        Table of edges, with source and target columns which correspond with the node
        Ids in the nodes dataframe.
    directed : bool, optional
        Whether the network should be treated as directed, by default True
    sources : pd.Index, optional
        Specification of source nodes if representing a subgraph, by default None
    targets : pd.Index, optional
        Specification of target nodes if representing a subgraph, by default None
    """

    @beartype
    def __init__(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        directed: bool = True,
        sources: Optional[pd.Index] = None,
        targets: Optional[pd.Index] = None,
    ):
        # TODO checks ensuring that nodes and edges are valid.

        if not nodes.index.is_unique:
            raise ValueError("Node IDs must be unique.")

        referenced_node_ids = np.union1d(
            edges["source"].unique(), edges["target"].unique()
        )
        if not np.all(np.isin(referenced_node_ids, nodes.index)):
            raise ValueError("All nodes referenced in edges must be in nodes.")

        # should probably assume things like "source" and "target" columns
        # and that these elements are in the nodes dataframe
        # TODO are multigraphs allowed?
        # TODO assert that sources and targets and node index are all unique?
        self.nodes = nodes
        self.edges = edges
        if sources is None and targets is None:
            self.induced = True
            self._sources = None
            self._targets = None
        else:
            self.induced = False
            self._sources = sources
            self._targets = targets
        # TODO some checks on repeated edges if not directed
        self.directed = directed

    def copy(self) -> "NetworkFrame":
        """Return a copy of the NetworkFrame."""
        return copy.deepcopy(self)

    @property
    def sources(self):
        """Return the source node IDs of the network."""
        if self.induced:
            return self.nodes.index
        else:
            return self.nodes.index.intersection(self._sources, sort=False)
            # all_sources = self.edges["source"].unique()
            # # TODO verify that this retains the order
            # return self.nodes.index.intersection(all_sources, sort=False)

    @property
    def targets(self):
        """Return the target node IDs of the network."""
        if self.induced:
            return self.nodes.index
        else:
            return self.nodes.index.intersection(self._targets, sort=False)
            # all_targets = self.edges["target"].unique()
            # # TODO verify that this retains the order
            # return self.nodes.index.intersection(all_targets, sort=False)

    @property
    def source_nodes(self):
        """Return the source nodes of the network and their metadata."""
        return self.nodes.loc[self.sources]

    @property
    def target_nodes(self):
        """Return the target nodes of the network and their metadata."""
        return self.nodes.loc[self.targets]

    def __repr__(self) -> str:
        """Return a string representation of the NetworkFrame."""
        out = f"NetworkFrame(nodes={self.nodes.shape}, edges={self.edges.shape}, "
        out += f"induced={self.induced}, directed={self.directed})"
        return out

    def reindex_nodes(self, index: pd.Index) -> "NetworkFrame":
        """Reindex the nodes dataframe, also removes edges as necessary."""
        nodes = self.nodes.reindex(index=index, axis=0)
        edges = self.edges.query("(source in @nodes.index) & (target in @nodes.index)")
        return NetworkFrame(nodes, edges, directed=self.directed)

    def remove_nodes(
        self, nodes: Union[pd.DataFrame, pd.Index, list, np.ndarray], inplace=False
    ) -> Optional["NetworkFrame"]:
        """Remove nodes from the network and remove edges that are no longer valid."""
        if isinstance(nodes, pd.DataFrame):
            nodes = nodes.index
        nodes = self.nodes.drop(index=nodes)
        # get the edges that are connected to the nodes that are left after the query
        edges = self.edges.query("(source in @nodes.index) & (target in @nodes.index)")
        if inplace:
            self.nodes = nodes
            self.edges = edges
            return None
        else:
            return NetworkFrame(nodes, edges, directed=self.directed)

    def remove_edges(
        self, remove_edges: pd.DataFrame, inplace=False
    ) -> Optional["NetworkFrame"]:
        """Remove edges from the network."""
        # TODO handle inplace better

        remove_edges_index = pd.MultiIndex.from_frame(
            remove_edges[["source", "target"]]
        )
        current_index = pd.MultiIndex.from_frame(self.edges[["source", "target"]])
        new_index = current_index.difference(remove_edges_index)

        # TODO i think this destroys the old index?
        edges = self.edges.set_index(["source", "target"]).loc[new_index].reset_index()
        if inplace:
            self.edges = edges
            return None
        else:
            return NetworkFrame(self.nodes, edges, directed=self.directed)

    def add_nodes(
        self, new_nodes: pd.DataFrame, inplace=False
    ) -> Optional["NetworkFrame"]:
        """Add nodes to the network."""
        nodes = pd.concat([self.nodes, new_nodes], copy=False, sort=False, axis=0)
        if inplace:
            self.nodes = nodes
            return None
        else:
            return NetworkFrame(nodes, self.edges, directed=self.directed)

    def add_edges(
        self, new_edges: pd.DataFrame, inplace=False
    ) -> Optional["NetworkFrame"]:
        """Add edges to the network."""
        edges = pd.concat([self.edges, new_edges], copy=False, sort=False, axis=0)
        if inplace:
            self.edges = edges
            return None
        else:
            return NetworkFrame(self.nodes, edges, directed=self.directed)

    def query_nodes(self, query: str, inplace=False) -> Optional["NetworkFrame"]:
        """Query the nodes dataframe and remove edges that are no longer valid."""
        nodes = self.nodes.query(query)
        # get the edges that are connected to the nodes that are left after the query
        edges = self.edges.query("(source in @nodes.index) & (target in @nodes.index)")
        if inplace:
            self.nodes = nodes
            self.edges = edges
            return None
        else:
            return NetworkFrame(nodes, edges, directed=self.directed)

    def query_edges(self, query: str, inplace=False) -> Optional["NetworkFrame"]:
        """Query the edges dataframe."""
        edges = self.edges.query(query)
        if inplace:
            self.edges = edges
            return None
        else:
            return NetworkFrame(self.nodes, edges, directed=self.directed)

    def remove_unused_nodes(self, inplace=False) -> Optional["NetworkFrame"]:
        """Remove nodes that are not connected to any edges."""
        index = self.nodes.index
        new_index = index.intersection(
            self.edges.source.append(self.edges.target).unique()
        )
        nodes = self.nodes.loc[new_index]
        if inplace:
            self.nodes = nodes
            return None
        else:
            return NetworkFrame(nodes, self.edges, directed=self.directed)

    def apply_node_features(
        self, columns: ColumnsType, axis: EdgeAxisType = "both", inplace=False
    ) -> Optional["NetworkFrame"]:
        """Apply node features to the edges dataframe."""
        if not inplace:
            edges = self.edges.copy()
        else:
            edges = self.edges
        if isinstance(columns, str):
            columns = [columns]
        if axis in ["source", "both"]:
            for col in columns:
                edges[f"source_{col}"] = self.edges["source"].map(self.nodes[col])
        if axis in ["target", "both"]:
            for col in columns:
                edges[f"target_{col}"] = self.edges["target"].map(self.nodes[col])
        if inplace:
            self.edges = edges
            return None
        else:
            return NetworkFrame(self.nodes, edges, directed=self.directed)

    def to_adjacency(self, weight_col: str = "weight", aggfunc="sum") -> pd.DataFrame:
        """Return the adjacency matrix of the network."""
        # TODO: wondering if the sparse method of doing this would actually be faster
        # here too...
        adj_df = self.edges.pivot_table(
            index="source",
            columns="target",
            values=weight_col,
            fill_value=0,
            aggfunc=aggfunc,
            sort=False,
        )
        adj_df = adj_df.reindex(
            index=self.sources,
            columns=self.targets,
            fill_value=0,
        )
        adj_df.index = adj_df.index.set_names("source")
        adj_df.columns = adj_df.columns.set_names("target")
        return adj_df

    def to_networkx(self):
        """Return a networkx graph of the network."""
        import networkx as nx

        if self.directed:
            create_using = nx.MultiDiGraph
        else:
            create_using = nx.MultiGraph

        g = nx.from_pandas_edgelist(
            self.edges,
            source="source",
            target="target",
            edge_attr=True,
            create_using=create_using,
        )
        nx.set_node_attributes(g, self.nodes.to_dict(orient="index"))
        return g

    def to_sparse_adjacency(
        self, weight_col: Optional[str] = None, aggfunc="sum", verify_integrity=True
    ) -> csr_array:
        """Return the adjacency matrix of the network as a sparse array."""
        edges = self.edges
        # TODO only necessary since there might be duplicate edges
        # might be more efficient to have a attributed checking this, e.g. set whether
        # this is a multigraph or not
        if weight_col is not None:
            effective_edges = edges.groupby(["source", "target"])[weight_col].agg(
                aggfunc
            )
        else:
            effective_edges = edges.groupby(["source", "target"]).size()

        data = effective_edges.values
        source_indices = effective_edges.index.get_level_values("source")
        target_indices = effective_edges.index.get_level_values("target")

        if verify_integrity:
            if (
                not np.isin(source_indices, self.nodes.index).all()
                and not np.isin(target_indices, self.nodes.index).all()
            ):
                raise ValueError(
                    "Not all source and target indices are in the nodes index."
                )

        source_indices = pd.Categorical(source_indices, categories=self.nodes.index)
        target_indices = pd.Categorical(target_indices, categories=self.nodes.index)

        adj = csr_array(
            (data, (source_indices.codes, target_indices.codes)),
            shape=(len(self.sources), len(self.targets)),
        )
        return adj

    def _get_component_indices(self) -> tuple[int, np.ndarray]:
        """Helper function for connected_components."""
        from scipy.sparse.csgraph import connected_components

        adjacency = self.to_sparse_adjacency()
        n_components, labels = connected_components(adjacency, directed=self.directed)
        return n_components, labels

    def largest_connected_component(
        self, inplace: bool = False, verbose: bool = False
    ) -> Optional["NetworkFrame"]:
        """Return the largest connected component of the network."""
        _, labels = self._get_component_indices()
        label_counts = pd.Series(labels).value_counts()
        biggest_label = label_counts.idxmax()
        mask = labels == biggest_label

        if verbose:
            n_removed = len(self.nodes) - mask.sum()
            print(f"Nodes removed when taking largest connected component: {n_removed}")

        nodes = self.nodes.iloc[mask]
        edges = self.edges.query("(source in @nodes.index) & (target in @nodes.index)")

        if inplace:
            self.nodes = nodes
            self.edges = edges
            return None
        else:
            return NetworkFrame(nodes, edges, directed=self.directed)

    def connected_components(self) -> Iterator["NetworkFrame"]:
        """Return the connected components of the network."""
        n_components, labels = self._get_component_indices()
        index = self.nodes.index

        for i in range(n_components):
            this_index = index[labels == i]
            yield self.loc[this_index, this_index]

    def n_connected_components(self) -> int:
        """Return the number of connected components of the network."""
        n_components, _ = self._get_component_indices()
        return n_components

    def is_fully_connected(self) -> bool:
        """Return whether the network is fully connected."""
        return self.n_connected_components() == 1

    def label_nodes_by_component(
        self, inplace: bool = False, name: str = "component"
    ) -> Optional["NetworkFrame"]:
        """Add a column labeling nodes by which connected component they are in."""
        _, labels = self._get_component_indices()

        if inplace:
            self.nodes[name] = labels
            self.nodes[name] = self.nodes[name].astype("Int64")
            return None
        else:
            nodes = self.nodes.copy()
            nodes[name] = labels
            nodes[name] = nodes[name].astype("Int64")
            return NetworkFrame(nodes, self.edges, directed=self.directed)

    def groupby_nodes(self, by=None, axis="both", **kwargs) -> "NodeGroupBy":
        """Group the frame by node data for the source or target (or both).

        Parameters
        ----------
        by : _type_, optional
            _description_, by default None
        axis : str, optional
            _description_, by default 'both'

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if axis == 0:
            source_nodes_groupby = self.source_nodes.groupby(by=by, **kwargs)
        elif axis == 1:
            target_nodes_groupby = self.target_nodes.groupby(by=by, **kwargs)
        elif axis == "both":
            source_nodes_groupby = self.source_nodes.groupby(by=by, **kwargs)
            target_nodes_groupby = self.target_nodes.groupby(by=by, **kwargs)
        else:
            raise ValueError("Axis must be 0 or 1 or 'both'")

        return NodeGroupBy(self, source_nodes_groupby, target_nodes_groupby)

    @property
    def loc(self) -> "LocIndexer":
        """Return a LocIndexer for the frame."""
        return LocIndexer(self)

    def __eq__(self, other: object) -> bool:
        """
        Check if two NetworkFrames are equal.

        Note that this considers both node/edge names and features. It does not consider
        the order of the nodes/edges. It does not consider the indexing of the edges.
        This may change in a future release.
        """

        if not isinstance(other, NetworkFrame):
            return False

        nodes1 = self.nodes
        nodes2 = other.nodes
        edges1 = self.edges
        edges2 = other.edges
        if not nodes1.sort_index().equals(nodes2.sort_index()):
            return False
        if (
            not edges1.sort_values(["source", "target"])
            .reset_index(drop=True)
            .equals(edges2.sort_values(["source", "target"]).reset_index(drop=True))
        ):
            return False
        return True

    def __ne__(self, other: object) -> bool:
        """
        Check if two NetworkFrames are not equal.

        Note that this considers both node/edge names and features. It does not consider
        the order of the nodes/edges. It does not consider the indexing of the edges.
        This may change in a future release.
        """
        return not self.__eq__(other)

    def to_dict(self, orient: str = "dict") -> dict:
        """Return a dictionary representation of the NetworkFrame."""
        return {
            "nodes": self.nodes.to_dict(orient=orient),
            "edges": self.edges.to_dict(orient=orient),
            "directed": self.directed,
        }

    def to_json(self, orient: str = "dict") -> str:
        """Return a JSON representation of the NetworkFrame."""
        import json

        return json.dumps(self.to_dict(orient=orient))

    @classmethod
    def from_dict(cls, d: dict, orient="columns", index_dtype=int) -> "NetworkFrame":
        """Return a NetworkFrame from a dictionary representation."""
        edges = pd.DataFrame.from_dict(d["edges"], orient=orient)
        nodes = pd.DataFrame.from_dict(d["nodes"], orient=orient)
        nodes.index = nodes.index.astype(index_dtype)
        return cls(
            nodes,
            edges,
            directed=d["directed"],
        )
