"""Classes for representing networks and metadata."""


import copy
from collections.abc import Iterator
from typing import Any, Callable, Literal, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from beartype import beartype
from scipy.sparse import csr_array

from .groupby import NodeGroupBy

AxisType = Union[
    Literal[0], Literal[1], Literal["index"], Literal["columns"], Literal["both"]
]

EdgeAxisType = Union[Literal["source"], Literal["target"], Literal["both"]]

ColumnsType = Union[list, str]

NetworkFrameReturn = Union["NetworkFrame", None]

Index = Union[pd.Index, pd.MultiIndex, np.ndarray, list]


class NetworkFrame:
    """Represent a network as a pair of DataFrames, one for nodes and one for edges."""

    @beartype
    def __init__(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        directed: bool = True,
        sources: Optional[pd.Index] = None,
        targets: Optional[pd.Index] = None,
    ):
        """
        Parameters
        ----------
        nodes
            Table of node attributes. The node IDs must be in the index.
        edges
            Table of edges, with source and target columns which correspond with the node
            IDs in`nodes.index`.
        directed
            Whether the network should be treated as directed.
        sources
            Specification of source nodes if representing a subgraph.
        targets
            Specification of target nodes if representing a subgraph.
        """

        # TODO checks ensuring that nodes and edges are valid.

        if not nodes.index.is_unique:
            raise ValueError("Node IDs must be unique.")

        referenced_node_ids = np.union1d(
            edges["source"].unique(), edges["target"].unique()
        )
        if not np.all(np.isin(referenced_node_ids, nodes.index)):
            raise ValueError(
                "All nodes referenced in `edges` must be in `nodes` index."
            )

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
        """Return a copy of the NetworkFrame.

        Returns
        -------
        :
            A copy of the NetworkFrame.
        """
        return copy.deepcopy(self)

    @property
    def sources(self) -> pd.Index:
        # """Source node IDs of the network."""
        if self.induced:
            return self.nodes.index
        else:
            return self.nodes.index.intersection(self._sources, sort=False)
            # all_sources = self.edges["source"].unique()
            # # TODO verify that this retains the order
            # return self.nodes.index.intersection(all_sources, sort=False)

    @property
    def targets(self) -> pd.Index:
        # """Return the target node IDs of the network."""
        if self.induced:
            return self.nodes.index
        else:
            return self.nodes.index.intersection(self._targets, sort=False)
            # all_targets = self.edges["target"].unique()
            # # TODO verify that this retains the order
            # return self.nodes.index.intersection(all_targets, sort=False)

    @property
    def source_nodes(self) -> pd.DataFrame:
        # """Return the source nodes of the network and their metadata."""
        return self.nodes.loc[self.sources]

    @property
    def target_nodes(self) -> pd.DataFrame:
        # """Return the target nodes of the network and their metadata."""
        return self.nodes.loc[self.targets]

    def __repr__(self) -> str:
        """Return a string representation of the NetworkFrame.

        Returns
        -------
        :
            A string representation of the NetworkFrame.
        """
        out = f"NetworkFrame(nodes={self.nodes.shape}, edges={self.edges.shape})"
        return out

    def __len__(self) -> int:
        """Return the number of nodes in the network.

        Returns
        -------
        :
            The number of nodes in the network.
        """
        return len(self.nodes)

    def reindex_nodes(self, index: pd.Index) -> "NetworkFrame":
        """Reindex `.nodes`, also removes edges as necessary.

        See [pandas.DataFrame.reindex][] for more information on reindexing.

        Parameters
        ----------
        index
            The new index to use.

        Returns
        -------
        :
            A new NetworkFrame with the reindexed nodes.
        """
        nodes = self.nodes.reindex(index=index, axis=0)
        edges = self.edges.query("(source in @nodes.index) & (target in @nodes.index)")
        return NetworkFrame(nodes, edges, directed=self.directed)

    def remove_nodes(
        self, nodes: Union[pd.DataFrame, Index], inplace=False
    ) -> Optional["NetworkFrame"]:
        """Remove nodes from the network and remove edges that are no longer valid.

        Parameters
        ----------
        nodes
            The index of nodes to remove. If a `pd.DataFrame` is passed, its index is
            used; otherwise the object is interpreted as an index-like.

        Returns
        -------
        :
            A new NetworkFrame with the nodes removed. If `inplace=True`, returns `None`.
        """
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
        # """Remove edges from the network."""
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
        # """Add nodes to the network."""
        nodes = pd.concat([self.nodes, new_nodes], copy=False, sort=False, axis=0)
        if inplace:
            self.nodes = nodes
            return None
        else:
            return NetworkFrame(nodes, self.edges, directed=self.directed)

    def add_edges(
        self, new_edges: pd.DataFrame, inplace=False
    ) -> Optional["NetworkFrame"]:
        # """Add edges to the network."""
        edges = pd.concat([self.edges, new_edges], copy=False, sort=False, axis=0)
        if inplace:
            self.edges = edges
            return None
        else:
            return NetworkFrame(self.nodes, edges, directed=self.directed)

    def query_nodes(
        self,
        expr: str,
        inplace: bool = False,
        local_dict: Optional[dict] = None,
        global_dict: Optional[dict] = None,
    ) -> Optional["NetworkFrame"]:
        """
        Select a subnetwork via a query the `.nodes` DataFrame.

        Parameters
        ----------
        expr
            The query to use on `.nodes`. See [pandas.DataFrame.query][] for more
            information.
        inplace
            Whether to modify the `NetworkFrame` rather than returning a new one.
        local_dict
            A dictionary of local variables. Useful for using the `@` expressions in
            [pandas.DataFrame.query][]. It may be useful to pass `local_dict=locals()` to
            accomplish this.
        global_dict
            A dictionary of global variables. Useful for using the `@` expressions in
            [pandas.DataFrame.query][]. It may be useful to pass `global_dict=globals()`
            to accomplish this.

        Returns
        -------
        :
            A new NetworkFrame for the subnetwork. If `inplace=True`, returns `None`.

        Examples
        --------
        >>> from networkframe import NetworkFrame
        >>> import pandas as pd
        >>> nodes = pd.DataFrame(
        ...     {
        ...         "name": ["A", "B", "C", "D", "E"],
        ...         "color": ["red", "blue", "blue", "red", "blue"],
        ...     },
        ...     index=[0, 1, 2, 3, 4],
        ... )
        >>> edges = pd.DataFrame(
        ...     {
        ...         "source": [0, 1, 2, 3, 4],
        ...         "target": [1, 2, 3, 4, 0],
        ...         "weight": [1, 2, 3, 4, 5],
        ...     }
        ... )
        >>> nf = NetworkFrame(nodes, edges)
        >>> sub_nf = nf.query_nodes("color == 'red'") # select subnetwork of red nodes
        >>> sub_nf
        NetworkFrame(nodes=(2, 2), edges=(2, 3))
        """
        nodes = self.nodes.query(expr, local_dict=local_dict, global_dict=global_dict)
        # get the edges that are connected to the nodes that are left after the query
        edges = self.edges.query("(source in @nodes.index) & (target in @nodes.index)")
        if inplace:
            self.nodes = nodes
            self.edges = edges
            return None
        else:
            return NetworkFrame(nodes, edges, directed=self.directed)

    def query_edges(
        self,
        expr: str,
        inplace: bool = False,
        local_dict: Optional[dict] = None,
        global_dict: Optional[dict] = None,
    ) -> Optional["NetworkFrame"]:
        """
        Select a subnetwork via a query the `.edges` DataFrame.

        Parameters
        ----------
        expr
            The query to use on `.edges`. See [pandas.DataFrame.query][] for more
            information.
        inplace
            Whether to modify the `NetworkFrame` rather than returning a new one.
        local_dict
            A dictionary of local variables. Useful for using the `@` expressions in
            [pandas.DataFrame.query][]. It may be useful to pass `local_dict=locals()` to
            accomplish this.
        global_dict
            A dictionary of global variables. Useful for using the `@` expressions in
            [pandas.DataFrame.query][]. It may be useful to pass `global_dict=globals()`
            to accomplish this.

        Returns
        -------
        :
            A new NetworkFrame for the subnetwork. If `inplace=True`, returns `None`.

        Examples
        --------
        >>> from networkframe import NetworkFrame
        >>> import pandas as pd
        >>> nodes = pd.DataFrame(
        ...     {
        ...         "name": ["A", "B", "C", "D", "E"],
        ...         "color": ["red", "blue", "blue", "red", "blue"],
        ...     },
        ...     index=[0, 1, 2, 3, 4],
        ... )
        >>> edges = pd.DataFrame(
        ...     {
        ...         "source": [0, 1, 2, 3, 4],
        ...         "target": [1, 2, 3, 4, 0],
        ...         "weight": [1, 2, 3, 4, 5],
        ...     }
        ... )
        >>> nf = NetworkFrame(nodes, edges)
        >>> sub_nf = nf.query_edges("weight > 2") # select subnetwork of edges with weight > 2
        >>> sub_nf
        NetworkFrame(nodes=(5, 2), edges=(3, 3))

        """
        edges = self.edges.query(expr, local_dict=local_dict, global_dict=global_dict)
        if inplace:
            self.edges = edges
            return None
        else:
            return NetworkFrame(self.nodes, edges, directed=self.directed)

    def remove_unused_nodes(self, inplace: bool = False) -> Optional["NetworkFrame"]:
        """
        Remove nodes that are not connected to any edges.

        Parameters
        ----------
        inplace
            Whether to modify the `NetworkFrame` rather than returning a new one.

        Returns
        -------
        :
            A new NetworkFrame with the unused nodes removed. If `inplace=True`, returns
            `None`.

        Examples
        --------
        >>> from networkframe import NetworkFrame
        >>> import pandas as pd
        >>> nodes = pd.DataFrame(
        ...     {
        ...         "name": ["A", "B", "C", "D", "E"],
        ...         "color": ["red", "blue", "blue", "red", "blue"],
        ...     },
        ...     index=[0, 1, 2, 3, 4],
        ... )
        >>> edges = pd.DataFrame(
        ...     {
        ...         "source": [0, 1, 2, 2, 3],
        ...         "target": [1, 2, 3, 1, 0],
        ...         "weight": [1, 2, 3, 4, 5],
        ...     }
        ... )
        >>> nf = NetworkFrame(nodes, edges)
        >>> sub_nf = nf.remove_unused_nodes()
        >>> sub_nf
        NetworkFrame(nodes=(4, 2), edges=(5, 3))
        """

        index = self.nodes.index
        source_index = index.intersection(self.edges.source)
        target_index = index.intersection(self.edges.target)
        new_index = source_index.union(target_index)
        nodes = self.nodes.loc[new_index]
        if inplace:
            self.nodes = nodes
            return None
        else:
            return NetworkFrame(nodes, self.edges, directed=self.directed)

    def apply_node_features(
        self, columns: ColumnsType, axis: EdgeAxisType = "both", inplace: bool = False
    ) -> Optional["NetworkFrame"]:
        """
        Apply features from `.nodes` to `.edges`.

        This will create new columns in `.edges` with the node features of the source
        node as `"source_{column}"` and the node features of the target node as
        `"target_{column}"`. This can be useful for performing operations on edges based
        on the nodes they connect.

        Parameters
        ----------
        columns
            The columns in `.nodes` to apply to the edges.
        axis
            Whether to apply the features to the source nodes (`source`), target nodes
            (`target`), or both (`both`).
        inplace
            Whether to modify the `NetworkFrame` rather than returning a new one.

        Returns
        -------
        :
            A new NetworkFrame with the node features applied to the edges. If
            `inplace=True`, returns `None`.

        """
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

    def to_adjacency(
        self, weight_col: str = "weight", aggfunc: Union[str, Callable] = "sum"
    ) -> pd.DataFrame:
        """
        Return the [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix)
        of the network as a [pandas.DataFrame][].

        Source nodes are the rows and target nodes are the columns. The adjacency matrix
        will be returned in the same order as the nodes in `.nodes`.

        Parameters
        ----------
        weight_col
            The column in `.edges` to use as the weight for the adjacency matrix.
            In the current implementation this must be set; future releases will support
            `None` to deal with unweighted networks.
        aggfunc
            The function to use to aggregate multiple edges between the same source and
            target nodes. See [pandas.DataFrame.pivot_table][] for more information.

        Returns
        -------
        :
            The adjacency matrix of the network as a [pd.DataFrame][].
        """

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

    def to_networkx(
        self,
        create_using: Optional[
            Union[nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph]
        ] = None,
    ) -> Union[nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph]:
        """Return a NetworkX graph representation of the network.

        This NetworkX graph also includes all node and edge attributes.

        Parameters
        ----------
        create_using
            The [NetworkX graph class](https://networkx.org/documentation/stable/reference/classes/index.html)
            to use to create the graph. If `None`, defaults to `nx.MultiDiGraph` if the
            network is directed and `nx.MultiGraph` if the network is undirected.

        Returns
        -------
        :
            A NetworkX representation of the network.
        """

        if create_using is None:
            # default to multigraph
            if self.directed:
                create_using = nx.MultiDiGraph
            else:
                create_using = nx.MultiGraph

        g = nx.from_pandas_edgelist(
            self.edges,
            source="source",
            target="target",
            edge_attr=True if len(self.edges.columns) > 2 else None,
            create_using=create_using,
        )

        # add any missing nodes to g
        index = self.nodes.index
        missing_nodes = index.difference(g.nodes)
        g.add_nodes_from(missing_nodes)

        nx.set_node_attributes(g, self.nodes.to_dict(orient="index"))

        return g

    def to_sparse_adjacency(
        self, weight_col: Optional[str] = None, aggfunc="sum", verify_integrity=True
    ) -> csr_array:
        """
        Return the [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix)
        of the network as a [scipy.sparse.csr_array][].

        Source nodes are the rows and target nodes are the columns. The adjacency matrix
        will be returned in the same order as the nodes in `.nodes`.

        Parameters
        ----------
        weight_col
            The column in `.edges` to use as the weight for the adjacency matrix. If
            `None`, then the number of edges between the source and target nodes will be
            used.
        aggfunc
            The function to use to aggregate multiple edges between the same source and
            target nodes. See [pandas.DataFrame.pivot_table][] for more information.
        verify_integrity
            Whether to verify that all source and target nodes are in the nodes index.
            Adds some overhead but prevents a difficult to decipher error later on if
            the node table is missing entries.

        Returns
        -------
        :
            The adjacency matrix of the network as a [scipy.sparse.csr_array][].

        Notes
        -----
        This method currently only supports [scipy.sparse.csr_array][] output, but in
        the future may support other sparse array formats.
        """
        edges = self.edges
        # TODO only necessary since there might be duplicate edges
        # might be more efficient to have a attributed checking this, e.g. set whether
        # this is a multigraph or not
        # TODO doing this reset_index is kinda hacky, but just getting around the case
        # where source/target are used as both column names and index level names
        if weight_col is not None:
            effective_edges = (
                edges.reset_index(drop=True)
                .groupby(["source", "target"])[weight_col]
                .agg(aggfunc)
            )
        else:
            effective_edges = (
                edges.reset_index(drop=True).groupby(["source", "target"]).size()
            )

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
        """
        Find the largest [connected component](https://en.wikipedia.org/wiki/Component_(graph_theory))
        of the network.

        Parameters
        ----------
        inplace
            Whether to modify the `NetworkFrame` to select only the largest connected
            component, rather than returning a new one.
        verbose
            Whether to print the number of nodes removed when taking the largest connected
            component.

        Returns
        -------
        :
            A new NetworkFrame with only the largest connected component. If
            `inplace=True`, returns `None`.
        """
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
        """
        Iterate over the [connected components](https://en.wikipedia.org/wiki/Component_(graph_theory))
        of the network.

        Yields
        ------
        :
            A new NetworkFrame for each connected component.
        """
        n_components, labels = self._get_component_indices()
        index = self.nodes.index

        for i in range(n_components):
            this_index = index[labels == i]
            yield self.loc[this_index, this_index]

    def n_connected_components(self) -> int:
        """
        Return the number of [connected components](https://en.wikipedia.org/wiki/Component_(graph_theory))
        of the network.

        Returns
        -------
        :
            The number of connected components.
        """
        n_components, _ = self._get_component_indices()
        return n_components

    def is_fully_connected(self) -> bool:
        """
        Return whether the network is fully connected.

        Returns
        -------
        :
            Whether the network is fully connected.
        """
        return self.n_connected_components() == 1

    def label_nodes_by_component(
        self, name: str = "component", inplace: bool = False
    ) -> Optional["NetworkFrame"]:
        """
        Add a column to `.nodes` labeling which connected component they are in.

        Parameters
        ----------
        name
            The name of the column to add to `.nodes`.
        inplace
            Whether to modify the `NetworkFrame` rather than returning a new one.

        Returns
        -------
        :
            A new NetworkFrame with the component labels added to `.nodes`. If
            `inplace=True`, returns `None`.
        """
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

    def select_component_from_node(
        self, node_id: Any, directed=True, inplace=False
    ) -> Optional["NetworkFrame"]:
        """
        Select the connected component containing the given node.

        This function avoids computing the entire connected component structure of the
        graph, instead using a shortest path algorithm to find the connected component
        of the node of interest. As such, it may be faster than using
        `connected_components`.

        Parameters
        ----------
        node_id
            The node ID to use to select the connected component.
        directed
            Whether to consider the network as directed for computing the reachable
            nodes.
        inplace
            Whether to modify the `NetworkFrame` rather than returning a new one.

        Returns
        -------
        :
            A new NetworkFrame with only the connected component containing the given
            node. If `inplace=True`, returns `None`.
        """
        from scipy.sparse.csgraph import shortest_path

        sparse_adjacency = self.to_sparse_adjacency()
        node_iloc = self.nodes.index.get_loc(node_id)

        dists = shortest_path(sparse_adjacency, directed=directed, indices=node_iloc)
        mask = ~np.isinf(dists)
        out = self.loc[mask, mask]
        if inplace:
            self.nodes = out.nodes
            self.edges = out.edges
        else:
            return out

    def groupby_nodes(
        self, by: Union[Any, list], axis: EdgeAxisType = "both", **kwargs
    ) -> "NodeGroupBy":
        """Group the frame by node data for the source or target (or both).

        See [pandas.DataFrame.groupby][] for more information.

        Parameters
        ----------
        by
            Column name or list of column names to group by.
        axis
            Whether to group by the source nodes (`source` or `0`), target nodes
            (`target` or `0`), or both (`both`).
        kwargs
            Additional keyword arguments to pass to [pandas.DataFrame.groupby][].

        Returns
        -------
        :
            A `NodeGroupBy` object representing the specified groups.

        Examples
        --------
        >>> from networkframe import NetworkFrame
        >>> import pandas as pd
        >>> nodes = pd.DataFrame(
        ...     {
        ...         "name": ["A", "B", "C", "D", "E"],
        ...         "color": ["red", "blue", "blue", "red", "blue"],
        ...     },
        ...     index=[0, 1, 2, 3, 4],
        ... )
        >>> edges = pd.DataFrame(
        ...     {
        ...         "source": [0, 1, 2, 3, 4],
        ...         "target": [1, 2, 3, 4, 0],
        ...         "weight": [1, 2, 3, 4, 5],
        ...     }
        ... )
        >>> nf = NetworkFrame(nodes, edges)
        >>> for color, subgraph in nf.groupby_nodes("color", axis="both"):
        ...     print(color)
        ...     print(subgraph.edges)
        ('blue', 'blue')
            source  target  weight
        1       1       2       2
        3       2       1       4
        ('blue', 'red')
            source  target  weight
        2       2       3       3
        ('red', 'blue')
            source  target  weight
        0       0       1       1
        ('red', 'red')
            source  target  weight
        4       3       0       5
        """
        if axis == 0 or axis == "source":
            source_nodes_groupby = self.source_nodes.groupby(by=by, **kwargs)
        elif axis == 1 or axis == "target":
            target_nodes_groupby = self.target_nodes.groupby(by=by, **kwargs)
        elif axis == "both":
            source_nodes_groupby = self.source_nodes.groupby(by=by, **kwargs)
            target_nodes_groupby = self.target_nodes.groupby(by=by, **kwargs)
        else:
            raise ValueError("Axis must be 0 or 1 or 'both'")

        return NodeGroupBy(self, source_nodes_groupby, target_nodes_groupby)

    @property
    def loc(self) -> "LocIndexer":
        """Access a subgraph by node ID(s).

        `.loc[]` is primarily label based, but in the future a boolean array may be
        supported. Currently, `.loc` only supports selecting both rows and columns, i.e.
        `nf.loc[row_indexer, column_indexer]`.


        Examples
        --------
        >>> from networkframe import NetworkFrame
        >>> import pandas as pd
        >>> nodes = pd.DataFrame(
        ...     {
        ...         "name": ["A", "B", "C", "D", "E"],
        ...         "color": ["red", "blue", "blue", "red", "blue"],
        ...     },
        ...     index=[0, 1, 2, 3, 4],
        ... )
        >>> edges = pd.DataFrame(
        ...     {
        ...         "source": [0, 1, 2, 3, 4],
        ...         "target": [1, 2, 3, 4, 0],
        ...         "weight": [1, 2, 3, 4, 5],
        ...     }
        ... )
        >>> nf = NetworkFrame(nodes, edges)
        >>> sub_nf = nf.loc[[1, 2], [2, 3]]
        >>> sub_nf
        NetworkFrame(nodes=(3, 1), edges=(2, 3))
        >>> sub_nf.to_adjacency()
        target  2  3
        source
        1       2  0
        2       0  3
        """
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

    # TODO make this docstring use automatic cross-reference to pandas docs
    def to_dict(self, orient: str = "dict") -> dict:
        """Return a dictionary representation of the NetworkFrame.

        Parameters
        ----------
        orient
            The format of the dictionary according to [pandas.DataFrame.to_dict].

        Returns
        -------
        :
            A dictionary representation of the NetworkFrame.
        """
        return {
            "nodes": self.nodes.to_dict(orient=orient),
            "edges": self.edges.to_dict(orient=orient),
            "directed": self.directed,
        }

    def to_json(self, orient: str = "dict") -> str:
        """
        Return a JSON (string) representation of the NetworkFrame.

        Parameters
        ----------
        orient
            The format of the dictionary according to [pandas.DataFrame.to_dict][].

        Returns
        -------
        :
            A JSON (string) representation of the NetworkFrame.
        """
        import json

        return json.dumps(self.to_dict(orient=orient))

    @classmethod
    def from_dict(cls, d: dict, orient="columns", index_dtype=int) -> "NetworkFrame":
        """
        Return a NetworkFrame from a dictionary representation.

        The dictionary representation should have keys "nodes", "edges", and "directed".
        The values of "nodes" and "edges" should be dictionaries with the same format as
        the output of [pandas.DataFrame.to_dict][] according to `orient`.

        Parameters
        ----------
        d
            The dictionary representation of the NetworkFrame.
        orient
            The format of the dictionary according to [pandas.DataFrame.to_dict][].
        index_dtype
            The data type of the index of the nodes DataFrame.

        Returns
        -------
        :
            A NetworkFrame.
        """
        edges = pd.DataFrame.from_dict(d["edges"], orient=orient)
        nodes = pd.DataFrame.from_dict(d["nodes"], orient=orient)
        nodes.index = nodes.index.astype(index_dtype)
        return cls(
            nodes,
            edges,
            directed=d["directed"],
        )


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
