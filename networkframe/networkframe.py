"""Classes for representing networks and metadata."""

import copy
from collections.abc import Iterator
from typing import Any, Callable, Literal, Optional, Self, Union

import networkx as nx
import numpy as np
import pandas as pd
from beartype import beartype
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components, dijkstra, shortest_path
from scipy.sparse.linalg import eigsh
from tqdm.autonotebook import tqdm

from .groupby import NodeGroupBy

AxisType = Union[
    Literal[0], Literal[1], Literal["index"], Literal["columns"], Literal["both"]
]

EdgeAxisType = Union[Literal["source"], Literal["target"], Literal["both"]]

ColumnsType = Union[list, str]

NetworkFrameReturn = Union["NetworkFrame", None]

Index = Union[pd.Index, pd.MultiIndex, np.ndarray, list]

ConnectionType = Union[Literal["weak"], Literal["strong"]]


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
        validate: bool = True,
        induced: bool = True,
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
        validate
            Whether to check that the nodes and edges are valid. This can be turned off
            to speed performance but risks causing errors later on.
        induced
            Whether the network is induced, i.e. whether the nodes and edges are
            specified as a subgraph of a larger network. Currently non-functional,
            subject to some API changes in the future.
        """

        # TODO more checks ensuring that nodes and edges are valid.
        if validate:
            if not nodes.index.is_unique:
                raise ValueError("Node IDs must be unique.")
            referenced_node_ids = np.union1d(
                edges["source"].unique(), edges["target"].unique()
            )
            # TODO fix this very very slow check
            if not np.all(np.isin(referenced_node_ids, nodes.index)):
                raise ValueError(
                    "All nodes referenced in `edges` must be in `nodes` index."
                )

        # should probably assume things li
        # ke "source" and "target" columns
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

    def _return(self, inplace: bool = False, **kwargs):
        if inplace:
            for k, v in kwargs.items():
                setattr(self, k, v)
            return None
        else:
            out = copy.copy(self)
            for k, v in kwargs.items():
                setattr(out, k, v)
            return out

    def _old_return(
        self, nodes: pd.DataFrame, edges: pd.DataFrame, inplace: bool
    ) -> Optional[Self]:
        """Return a view/shallow copy of the NetworkFrame.

        This is used internally to return a view of the NetworkFrame rather than
        a copy.

        Parameters
        ----------
        nodes
            The nodes DataFrame.
        edges
            The edges DataFrame.

        Returns
        -------
        :
            A view of the NetworkFrame.
        """
        if inplace:
            self.nodes = nodes
            self.edges = edges
            return None
        else:
            # kwargs = self.get_public_attributes()
            # kwargs["nodes"] = nodes
            # kwargs["edges"] = edges
            # kwargs["validate"] = False
            # return self.__class__(**kwargs)
            out = copy.copy(self)
            out.nodes = nodes
            out.edges = edges
            return out

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

    def deepcopy(self):
        return copy.deepcopy(self)

    def reindex_nodes(self, index: pd.Index) -> Self:
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
        nodes = self.nodes.reindex(index=index)
        edges = self.edges.query("(source in @nodes.index) & (target in @nodes.index)")
        out: NetworkFrame = self._return(nodes=nodes, edges=edges, inplace=False)  # type: ignore
        return out  # type: ignore

    def remove_nodes(
        self, nodes: Union[pd.DataFrame, Index], inplace=False
    ) -> Optional[Self]:
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
        return self._return(nodes=nodes, edges=edges, inplace=inplace)

    def remove_edges(self, remove_edges: pd.DataFrame, inplace=False) -> Optional[Self]:
        # """Remove edges from the network."""
        # TODO handle inplace better

        remove_edges_index = pd.MultiIndex.from_frame(
            remove_edges[["source", "target"]]
        )
        current_index = pd.MultiIndex.from_frame(self.edges[["source", "target"]])
        new_index = current_index.difference(remove_edges_index)

        # TODO i think this destroys the old index?
        edges = self.edges.set_index(["source", "target"]).loc[new_index].reset_index()

        return self._return(edges=edges, inplace=inplace)

    def add_nodes(self, new_nodes: pd.DataFrame, inplace=False) -> Optional[Self]:
        # """Add nodes to the network."""
        nodes = pd.concat([self.nodes, new_nodes], copy=False, sort=False, axis=0)

        return self._return(nodes=nodes, inplace=inplace)

    def add_edges(self, new_edges: pd.DataFrame, inplace=False) -> Optional[Self]:
        # """Add edges to the network."""
        edges = pd.concat([self.edges, new_edges], copy=False, sort=False, axis=0)

        return self._return(edges=edges, inplace=inplace)

    def query_nodes(
        self,
        expr: str,
        inplace: bool = False,
        local_dict: Optional[dict] = None,
        global_dict: Optional[dict] = None,
        **kwargs,
    ) -> Optional[Self]:
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
        **kwargs
            Additional keyword arguments to pass to [pandas.DataFrame.query][].

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
        nodes = self.nodes.query(
            expr, local_dict=local_dict, global_dict=global_dict, **kwargs
        )
        # get the edges that are connected to the nodes that are left after the query
        edges = self.edges.query(
            "(source in @nodes.index) & (target in @nodes.index)", **kwargs
        )

        return self._return(nodes=nodes, edges=edges, inplace=inplace)

    def get_public_attributes(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def query_edges(
        self,
        expr: str,
        inplace: bool = False,
        local_dict: Optional[dict] = None,
        global_dict: Optional[dict] = None,
        **kwargs,
    ) -> Optional[Self]:
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
        **kwargs
            Additional keyword arguments to pass to [pandas.DataFrame.query][].

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
        edges = self.edges.query(
            expr, local_dict=local_dict, global_dict=global_dict, **kwargs
        )

        return self._return(edges=edges, inplace=inplace)

    def remove_unused_nodes(self, inplace: bool = False) -> Optional[Self]:
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

        return self._return(nodes=nodes, inplace=inplace)

    def apply_node_features(
        self, columns: ColumnsType, axis: EdgeAxisType = "both", inplace: bool = False
    ) -> Optional[Self]:
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
        return self._return(edges=edges, inplace=inplace)

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
        self,
        weight_col: Optional[str] = None,
        aggfunc="sum",
        verify_integrity=True,
        format="csr",
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

        if format == "lil":
            from scipy.sparse import lil_array

            adj = lil_array(adj)
        return adj
    
    def to_torch_geometric(self):
        pass

    def to_torch_geometric(self, directed=True, weight_col=None):
        import torch
        from torch_geometric.data import Data

        nodes = self.nodes
        edges = self.edges
        if isinstance(edges, list):
            edges = np.array(edges)

        edgelist = edges[["source", "target"]].values
        remapped_sources = nodes.index.get_indexer_for(edgelist[:, 0])
        remapped_targets = nodes.index.get_indexer_for(edgelist[:, 1])
        remapped_edges = np.stack([remapped_sources, remapped_targets], axis=1)
        remapped_nodes = nodes.reset_index(drop=True).fillna(0.0)

        if directed:
            edge_index = torch.tensor(remapped_edges.T, dtype=torch.long)
        else:
            edge_index = torch.tensor(
                np.concatenate([remapped_edges.T, remapped_edges[:, ::-1].T], axis=1),
                dtype=torch.long,
            )

        if weight_col is not None:
            if directed:
                edge_attr = torch.tensor(
                    edges[weight_col].values, dtype=torch.float
                ).unsqueeze(1)
            else:
                edge_attr = torch.tensor(
                    np.concatenate(
                        [edges[weight_col].values, edges[weight_col].values]
                    ),
                    dtype=torch.float,
                ).unsqueeze(1)
        else:
            edge_attr = None

        x = torch.tensor(remapped_nodes.values, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data

    def _get_component_indices(
        self, directed: bool = True, connection: ConnectionType = "weak"
    ) -> tuple[int, np.ndarray]:
        """Helper function for connected_components."""

        adjacency = self.to_sparse_adjacency()
        n_components, labels = connected_components(
            adjacency, directed=directed, connection=connection
        )
        return n_components, labels

    def largest_connected_component(
        self,
        directed: bool = True,
        connection: ConnectionType = "weak",
        inplace: bool = False,
        verbose: bool = False,
    ) -> Optional[Self]:
        """
        Find the largest [connected component](https://en.wikipedia.org/wiki/Component_(graph_theory))
        of the network.

        Parameters
        ----------
        directed
            If True (default), then operate on a directed graph: only move from point
            `i` to point `j` along edges from `i` to `j`. If False, then compute
            components on an undirected graph: the algorithm will tread edges from
            `i` to `j` and from `j` to `i` the same.
        connection
            ['weak'|'strong']. For directed graphs, the type of connection to use.
            Nodes `i` and `j` are strongly connected if a path exists both from `i` to
            `j` and from `j` to `i`. A directed graph is weakly connected if replacing
            all of its directed edges with undirected edges produces a connected
            (undirected) graph. If directed == False, this keyword is not referenced.
        inplace
            Whether to modify the `NetworkFrame` to select only the largest connected
            component, rather than returning a new one.
        verbose
            Whether to print the number of nodes removed when taking the largest
            connected component.

        Returns
        -------
        :
            A new NetworkFrame with only the largest connected component. If
            `inplace=True`, returns `None`.
        """
        _, labels = self._get_component_indices(
            directed=directed, connection=connection
        )
        label_counts = pd.Series(labels).value_counts()
        biggest_label = label_counts.idxmax()
        mask = labels == biggest_label

        if verbose:
            n_removed = len(self.nodes) - mask.sum()
            print(f"Nodes removed when taking largest connected component: {n_removed}")

        nodes = self.nodes.iloc[mask]
        edges = self.edges.query("(source in @nodes.index) & (target in @nodes.index)")

        return self._return(nodes=nodes, edges=edges, inplace=inplace)

    def connected_components(
        self, directed: bool = True, connection: ConnectionType = "weak"
    ) -> Iterator["NetworkFrame"]:
        """
        Iterate over the [connected components](https://en.wikipedia.org/wiki/Component_(graph_theory))
        of the network.


        Parameters
        ----------
        directed
            If True (default), then operate on a directed graph: only move from point
            `i` to point `j` along edges from `i` to `j`. If False, then compute
            components on an undirected graph: the algorithm will tread edges from
            `i` to `j` and from `j` to `i` the same.
        connection
            ['weak'|'strong']. For directed graphs, the type of connection to use.
            Nodes `i` and `j` are strongly connected if a path exists both from `i` to
            `j` and from `j` to `i`. A directed graph is weakly connected if replacing
            all of its directed edges with undirected edges produces a connected
            (undirected) graph. If directed == False, this keyword is not referenced.

        Yields
        ------
        :
            A new NetworkFrame for each connected component.
        """
        n_components, labels = self._get_component_indices(
            directed=directed, connection=connection
        )
        index = self.nodes.index

        for i in range(n_components):
            this_index = index[labels == i]
            yield self.loc[this_index, this_index]

    def n_connected_components(
        self, directed: bool = True, connection: ConnectionType = "weak"
    ) -> int:
        """
        Return the number of [connected components](https://en.wikipedia.org/wiki/Component_(graph_theory))
        of the network.

        Parameters
        ----------
        directed
            If True (default), then operate on a directed graph: only move from point
            `i` to point `j` along edges from `i` to `j`. If False, then compute
            components on an undirected graph: the algorithm will tread edges from
            `i` to `j` and from `j` to `i` the same.
        connection
            ['weak'|'strong']. For directed graphs, the type of connection to use.
            Nodes `i` and `j` are strongly connected if a path exists both from `i` to
            `j` and from `j` to `i`. A directed graph is weakly connected if replacing
            all of its directed edges with undirected edges produces a connected
            (undirected) graph. If directed == False, this keyword is not referenced.

        Returns
        -------
        :
            The number of connected components.
        """
        n_components, _ = self._get_component_indices(
            directed=directed, connection=connection
        )
        return n_components

    def is_fully_connected(
        self, directed: bool = True, connection: ConnectionType = "weak"
    ) -> bool:
        """
        Return whether the network is fully connected.

        Parameters
        ----------
        directed
            If True (default), then operate on a directed graph: only move from point
            `i` to point `j` along edges from `i` to `j`. If False, then compute
            components on an undirected graph: the algorithm will tread edges from
            `i` to `j` and from `j` to `i` the same.
        connection
            ['weak'|'strong']. For directed graphs, the type of connection to use.
            Nodes `i` and `j` are strongly connected if a path exists both from `i` to
            `j` and from `j` to `i`. A directed graph is weakly connected if replacing
            all of its directed edges with undirected edges produces a connected
            (undirected) graph. If directed == False, this keyword is not referenced.

        Returns
        -------
        :
            Whether the network is fully connected.
        """
        return (
            self.n_connected_components(directed=directed, connection=connection) == 1
        )

    def label_nodes_by_component(
        self,
        name: str = "component",
        inplace: bool = False,
        directed: bool = True,
        connection: ConnectionType = "weak",
    ) -> Optional[Self]:
        """
        Add a column to `.nodes` labeling which connected component they are in.


        Parameters
        ----------
        directed
            If True (default), then operate on a directed graph: only move from point
            `i` to point `j` along edges from `i` to `j`. If False, then compute
            components on an undirected graph: the algorithm will tread edges from
            `i` to `j` and from `j` to `i` the same.
        connection
            ['weak'|'strong']. For directed graphs, the type of connection to use.
            Nodes `i` and `j` are strongly connected if a path exists both from `i` to
            `j` and from `j` to `i`. A directed graph is weakly connected if replacing
            all of its directed edges with undirected edges produces a connected
            (undirected) graph. If directed == False, this keyword is not referenced.
        name
            The name of the column to add to `.nodes`.
        inplace
            Whether to modify the `NetworkFrame` rather than returning a new one. Copies
            the nodes dataframe if `inplace=False`.

        Returns
        -------
        :
            A new NetworkFrame with the component labels added to `.nodes`. If
            `inplace=True`, returns `None`.
        """
        _, labels = self._get_component_indices(
            directed=directed, connection=connection
        )

        if inplace:
            nodes = self.nodes
        else:
            nodes = self.nodes.copy()

        nodes[name] = labels
        nodes[name] = nodes[name].astype(int)

        return self._return(nodes=nodes, inplace=inplace)

    def component_labels(
        self, directed: bool = True, connection: str = "weak"
    ) -> pd.Series:
        """Return the indices of the connected components.

        Parameters
        ----------
        directed
            If True (default), then operate on a directed graph: only move from point
            `i` to point `j` along edges from `i` to `j`. If False, then compute
            components on an undirected graph: the algorithm will tread edges from
            `i` to `j` and from `j` to `i` the same.
        connection
            ['weak'|'strong']. For directed graphs, the type of connection to use.
            Nodes `i` and `j` are strongly connected if a path exists both from `i` to
            `j` and from `j` to `i`. A directed graph is weakly connected if replacing
            all of its directed edges with undirected edges produces a connected
            (undirected) graph. If directed == False, this keyword is not referenced.

        Returns
        -------
        :
            A series of the same length as the number of nodes, where each element
            corresponds to the connected component of the node at that index.
        """
        _, labels = self._get_component_indices(
            directed=directed, connection=connection
        )
        labels = pd.Series(labels, index=self.nodes.index)
        return labels

    def select_component_from_node(
        self, node_id: Any, directed=True, inplace=False
    ) -> Optional[Self]:
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

        sparse_adjacency = self.to_sparse_adjacency()
        node_iloc = self.nodes.index.get_loc(node_id)

        dists = shortest_path(sparse_adjacency, directed=directed, indices=node_iloc)
        mask = ~np.isinf(dists)
        subindex = self.nodes.index[mask]
        return self.query_nodes(
            "index in @subindex", inplace=inplace, local_dict=locals()
        )

    def groupby_nodes(
        self, by: Union[Any, list], axis: EdgeAxisType = "both", induced=False, **kwargs
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
        induced
            Whether to only yield groups over induced subgraphs, as opposed to all
            subgraphs.
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

        return NodeGroupBy(
            self, source_nodes_groupby, target_nodes_groupby, induced=induced, by=by
        )

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

        index1 = edges1.index.sort_values()
        index2 = edges2.index.sort_values()
        if not index1.equals(index2):
            return False

        # sort the edges the same way (note the index1 twice)
        edges1 = edges1.loc[index1]
        edges2 = edges2.loc[index1]
        if not edges1.equals(edges2):
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
        """Return a dictionary representation of the NetworkFrame.

        Parameters
        ----------
        orient
            The format of the dictionary according to [pandas.DataFrame.to_dict][].

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

    def node_agreement(self, other: Self) -> float:
        """
        Return the fraction of nodes in self that are also in other.

        Parameters
        ----------
        other
            The other NetworkFrame to compare to.

        Returns
        -------
        :
            The fraction of nodes that are shared between the two NetworkFrames.
        """
        return self.nodes.index.isin(other.nodes.index).mean()

    def k_hop_neighborhood(
        self,
        node_id: Union[int, str],
        k: int,
        directed: bool = False,
    ):
        """
        Return the k-hop neighborhood of a node.

        Parameters
        ----------
        node_id
            The node ID to use to select the k-hop neighborhood.
        k
            The number of hops to consider.
        directed
            Whether to consider the network as directed for computing the reachable
            nodes.

        Returns
        -------
        :
            A new NetworkFrame with only the k-hop neighborhood of the given node.
        """
        if k < 0:
            raise ValueError("k must be non-negative.")

        sparse_adjacency = self.to_sparse_adjacency()
        iloc = self.nodes.index.get_loc(node_id)

        dists = dijkstra(
            sparse_adjacency, directed=directed, indices=iloc, limit=k, unweighted=True
        )
        mask = ~np.isinf(dists)
        select_indices = self.nodes.index[mask]
        select_indices
        return self.query_nodes("index in @select_indices", local_dict=locals())

    def k_hop_mask(self, k: int, directed: bool = False, verbose=False):
        """
        Return the k-hop neighborhood of a node as a boolean mask.

        Parameters
        ----------
        node_id
            The node ID to use to select the k-hop neighborhood.
        k
            The number of hops to consider.
        directed
            Whether to consider the network as directed for computing the reachable
            nodes.

        Returns
        -------
        :
            A boolean mask of the nodes in the k-hop neighborhood of the given node.
        """
        if k < 0:
            raise ValueError("k must be non-negative.")

        sparse_adjacency = self.to_sparse_adjacency()

        # TODO add a check for interaction of directed and whether the graph has any
        # bi-directional edges
        dists = dijkstra(sparse_adjacency, directed=directed, limit=k, unweighted=True)
        mask = ~np.isinf(dists)
        return mask

    def k_hop_decomposition(
        self, k: int, directed: bool = False, verbose=False
    ) -> pd.Series:
        """
        Return the k-hop decomposition of the network.

        This function returns a series of NetworkFrames, each representing the k-hop
        neighborhood of a node.

        Parameters
        ----------
        k
            The number of hops to consider.
        """
        mask = self.k_hop_mask(k, directed=directed, verbose=verbose)

        out = {}
        for iloc in tqdm(range(len(self.nodes)), disable=not verbose):
            select_indices = self.nodes.index[mask[iloc]]
            sub_nf = self.query_nodes("index in @select_indices", local_dict=locals())
            out[self.nodes.index[iloc]] = sub_nf
        return pd.Series(out)

    def k_hop_aggregation(
        self,
        k: int,
        aggregations: Union[str, list] = "mean",
        directed: bool = False,
        drop_self_in_neighborhood: bool = True,
        drop_non_numeric: bool = True,
        verbose: int = False,
        engine: Literal["auto", "scipy", "pandas"] = "auto",
    ) -> pd.DataFrame:
        """
        Compute an aggregation over numeric features in the k-hop neighborhood of each
        node.

        Parameters
        ----------
        k
            The number of hops to consider, must be positive.
        aggregations
            The aggregation(s) to compute over the neighborhood. If using
            `engine='pandas'`, can be any aggregation that can be passed to
            [pandas.DataFrame.agg][]. If using `engine='scipy'`, currently can only be
            'mean', 'sum', or 'std'.
        directed
            Whether to consider the network as directed for computing the reachable
            nodes. Note that if `directed=False` but there are bidirectional edges with
            different weights, the results may not be as expected.
        drop_self_in_neighborhood
            Whether to drop the node itself from the neighborhood when aggregating.
        drop_non_numeric
            Whether to drop non-numeric columns when aggregating. Aggregating
            non-numeric columns can lead to errors, depending on the aggregation.
        verbose
            Whether to print progress.
        engine
            The engine to use for computing the aggregation. If 'auto', will use
            'pandas' if the aggregations are not all strings or all values accepted by
            the 'scipy' engine. Otherwise, will use 'scipy'. Note that the SciPy engine
            is likely to be faster for large graphs when the k-hop neighborhood of each
            node is much smaller than the total number of nodes.

        Returns
        -------
        :
            A DataFrame with the aggregated features for each node in the k-hop
            neighborhood. The index will be the same as nodes of the network, and the
            columns will be the aggregated features (with f'_neighbor_{agg}' appended
            to each column name).
        """

        if isinstance(aggregations, str):
            aggregations = [aggregations]

        if engine == "auto":
            # TODO might also want to do a check on sparsity of the graph here
            if not all([isinstance(x, str) for x in aggregations]) or not all(
                [x in ["mean", "sum", "std"] for x in aggregations]
            ):
                engine = "pandas"
            else:
                engine = "scipy"

        nodes = self.nodes
        if drop_non_numeric:
            nodes = nodes.select_dtypes(include=[np.number])

        mask = self.k_hop_mask(k, directed=directed, verbose=verbose)

        if verbose > 0:
            print("Aggregating over neighborhoods")
        if engine == "pandas":
            rows = []
            for iloc in tqdm(range(len(self.nodes)), disable=not verbose):
                # the selection here is pretty quick;
                node = self.nodes.index[iloc]
                select_nodes = self.nodes.loc[mask[iloc]]
                if drop_self_in_neighborhood:
                    select_nodes = select_nodes.drop(index=node)

                # the aggregation takes most of the time
                agg_neighbor_features = select_nodes.agg(aggregations)

                if isinstance(agg_neighbor_features, pd.Series):
                    agg_neighbor_features.index = agg_neighbor_features.index.map(
                        lambda x: f"{x}_neighbor_{aggregations[0]}"
                    )
                elif isinstance(agg_neighbor_features, pd.DataFrame):
                    agg_neighbor_features = agg_neighbor_features.unstack()
                    agg_neighbor_features.index = agg_neighbor_features.index.map(
                        lambda x: f"{x[0]}_neighbor_{x[1]}"
                    )
                agg_neighbor_features.name = node
                rows.append(agg_neighbor_features)
            neighborhood_features = pd.concat(rows, axis=1).T
        elif engine == "scipy":
            if not all([x in ["mean", "sum", "std"] for x in aggregations]):
                raise ValueError(
                    "Currently only 'mean', 'sum', and 'std' are allowed in "
                    "`aggregations` "
                    "when using the 'scipy' engine."
                )

            if drop_self_in_neighborhood:
                mask[np.diag_indices_from(mask)] = False

            # this is an adjacency matrix for whether nodes are in neighborhood
            mask = csr_array(mask)

            feature_mat = nodes.fillna(0).values

            neighborhood_sum_mat = mask @ feature_mat

            if "mean" in aggregations:
                # this sums the number of notna values in the neighborhood for each
                # feature
                divisor_matrix = mask @ nodes.notna().astype(int)
                divisor_matrix[divisor_matrix == 0] = 1

                neighborhood_mean_matrix = neighborhood_sum_mat / divisor_matrix
                neighborhood_mean_matrix = pd.DataFrame(
                    neighborhood_mean_matrix, index=nodes.index, columns=nodes.columns
                )
                neighborhood_mean_matrix.rename(
                    columns=lambda x: f"{x}_neighbor_mean", inplace=True
                )

            if "sum" in aggregations:
                neighborhood_sum_matrix = pd.DataFrame(
                    neighborhood_sum_mat, index=nodes.index, columns=nodes.columns
                )
                neighborhood_sum_matrix.rename(
                    columns=lambda x: f"{x}_neighbor_sum", inplace=True
                )

            if "std" in aggregations:
                # REF: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
                # using "Computing shifted data" method

                # supposedly, this subtraction helps with numerical stability
                # I think it makes the large values closer to correct, but the small
                # values worse (at least with many 0s)
                # could play with details here
                const = feature_mat.mean(axis=0)
                inner_term = feature_mat - const[None, :]

                # this is to deal with NaNs (which were previously set to 0)
                inner_term[nodes.isna().values] = 0

                # sum of squares of the shifted data
                first_term = mask @ (inner_term**2)
                # squared sum of the shifted data, divided by the number of non-NaNs
                second_term = (mask @ inner_term) ** 2 / divisor_matrix

                # this is a node by feature matrix of the variances for each feature
                # in that node's neighborhood
                new_divisor_matrix = divisor_matrix - 1
                new_divisor_matrix[new_divisor_matrix == 0] = 1
                variances = (first_term - second_term) / new_divisor_matrix
                variances[variances < 0] = 0

                neighborhood_std_matrix = np.sqrt(variances)
                neighborhood_std_matrix = pd.DataFrame(
                    neighborhood_std_matrix, index=nodes.index, columns=nodes.columns
                )
                neighborhood_std_matrix.rename(
                    columns=lambda x: f"{x}_neighbor_std", inplace=True
                )

            neighborhood_feature_dfs = []
            if "mean" in aggregations:
                neighborhood_feature_dfs.append(neighborhood_mean_matrix)
            if "sum" in aggregations:
                neighborhood_feature_dfs.append(neighborhood_sum_matrix)
            if "std" in aggregations:
                neighborhood_feature_dfs.append(neighborhood_std_matrix)

            neighborhood_features = pd.concat(neighborhood_feature_dfs, axis=1)

        neighborhood_features.index.name = self.nodes.index.name
        return neighborhood_features

    def condense(
        self,
        by: Union[Any, list],
        func: Union[
            Callable, Literal["mean", "sum", "max", "min", "any", "size"]
        ] = "size",
        weight_name="weight",
        columns=None,
    ) -> "NetworkFrame":
        """Apply a function, and create a new NetworkFrame such that the nodes of the
        new frame are the groups and the edges are the result of the function.

        The API and implementation of this function is rather fragile and subject to
        change.
        """

        edges = self.groupby_nodes(by).apply_edges(func, columns=columns)
        edges.name = weight_name
        edges = edges.reset_index()
        edges = edges.rename(
            columns={f"source_{by}": "source", f"target_{by}": "target"}
        )
        nodes_index = pd.Index(self.nodes[by].unique())
        nodes_index.name = by
        nodes = pd.DataFrame(index=nodes_index)
        return self.__class__(nodes, edges, directed=self.directed)

    def sort_spectral(self, weight_col="weight", inplace=False) -> "NetworkFrame":
        adjacency = self.to_sparse_adjacency(weight_col=weight_col)
        adjacency = adjacency + adjacency.T
        adjacency = adjacency.astype(float)

        _, u = eigsh(adjacency, k=1, which="LM", return_eigenvectors=True)

        nodes = self.nodes.iloc[np.argsort(u[:, 0])]

        return self._return(nodes=nodes, inplace=inplace)


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
            return self._frame._return(nodes=nodes, edges=edges, inplace=False)
            # return NetworkFrame(
            #     nodes,
            #     edges,
            #     directed=self._frame.directed,
            #     validate=False,
            # )
        else:
            nodes = pd.concat([source_nodes, target_nodes], copy=False, sort=False)
            nodes = nodes.loc[~nodes.index.duplicated(keep="first")]
            return self._frame._return(nodes=nodes, edges=edges, inplace=False)
            # return NetworkFrame(
            #     nodes,
            #     edges,
            #     directed=self._frame.directed,
            #     sources=row_index,
            #     targets=col_index,
            #     validate=False,
            # )
