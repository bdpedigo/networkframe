from typing import Callable, Literal, Union


class NodeGroupBy:
    """A class for grouping a `NetworkFrame` by a set of labels."""

    def __init__(
        self, frame, source_groupby, target_groupby, by, induced: bool = False
    ):
        """Groupby on nodes.

        Parameters
        ----------
        frame
            _description_
        source_groupby
            _description_
        target_groupby
            _description_
        induced
            _description_
        """
        self._frame = frame
        self._source_groupby = source_groupby
        self._target_groupby = target_groupby
        self.induced = induced
        self.by = by

        self._axis: Union[str, int]
        if source_groupby is None:
            self._axis = 1
        elif target_groupby is None:
            self._axis = 0
        else:
            self._axis = "both"

        if self.has_source_groups:
            self.source_group_names = list(source_groupby.groups.keys())
        if self.has_target_groups:
            self.target_group_names = list(target_groupby.groups.keys())

    def __len__(self):
        """Return the number of groups."""
        if self._axis == "both":
            if self.induced:
                return len(self._source_groupby)
            else:
                return len(self._source_groupby) * len(self._target_groupby)
        elif self._axis == 0:
            return len(self._source_groupby)
        elif self._axis == 1:
            return len(self._target_groupby)

    @property
    def has_source_groups(self):
        """Whether the frame has row groups."""
        return self._source_groupby is not None

    @property
    def has_target_groups(self):
        """Whether the frame has column groups."""
        return self._target_groupby is not None

    def __iter__(self):
        """Iterate over the groups."""
        if self._axis == "both":
            for source_group, source_objects in self._source_groupby:
                for target_group, target_objects in self._target_groupby:
                    if self.induced and source_group != target_group:
                        continue
                    else:
                        yield (
                            (source_group, target_group),
                            self._frame.loc[source_objects.index, target_objects.index],
                        )
        elif self._axis == 0:
            for source_group, source_objects in self._source_groupby:
                yield source_group, self._frame.loc[source_objects.index]
        elif self._axis == 1:
            for target_group, target_objects in self._target_groupby:
                yield target_group, self._frame.loc[:, target_objects.index]

    # def apply_nodes(self, func):
    #     nodes = self._frame.nodes

        # if isinstance(func, str):
        #     if func == 'n_possible':



    def apply_edges(self, func, columns=None):
        by = self.by
        if isinstance(by, list):
            raise ValueError(
                "Currently can only apply edges to a single group in `by`."
            )
        if self._axis != "both":
            raise ValueError("Currently can only apply edges when groupby is 'both'.")

        if isinstance(func, str):
            if func == "size":
                func = lambda x: x.shape[0]
            elif func == "mean":
                func = lambda x: x.mean()
            elif func == "sum":
                func = lambda x: x.sum()
            elif func == "max":
                func = lambda x: x.max()
            elif func == "min":
                func = lambda x: x.min()
            elif func == "any":
                func = lambda x: x.any()

        edges = self._frame.apply_node_features(by, inplace=False).edges

        edge_by = [f"source_{by}", f"target_{by}"]
        if columns is not None:
            out = edges.groupby(edge_by)[columns].apply(func)
        else:
            out = edges.groupby(edge_by).apply(func)
        return out

    def size_edges(self):
        return self.apply_edges("size")

    # def apply(self, func, to="frame"):
    #     """Apply a function to each group."""
    #     if self._axis == "both":
    #         answer = pd.DataFrame(
    #             index=self.source_group_names, columns=self.target_group_names
    #         )
    #     else:
    #         if self._axis == 0:
    #             answer = pd.Series(index=self.source_group_names)
    #         else:
    #             answer = pd.Series(index=self.target_group_names)
    #     for group, frame in tqdm(self, total=len(self)):
    #         if to == "frame":
    #             value = func(frame)
    #         elif to == "nodes":
    #             value = func(frame.nodes)
    #         elif to == "edges":
    #             value = func(frame.edges)
    #         answer.at[group] = value
    #     return answer

    @property
    def source_groups(self):
        """Return the row groups."""
        if self._axis == "both" or self._axis == 0:
            return self._source_groupby.groups
        else:
            raise ValueError("No source groups, groupby was on targets only")

    @property
    def target_groups(self):
        """Return the column groups."""
        if self._axis == "both" or self._axis == 1:
            return self._target_groupby.groups
        else:
            raise ValueError("No target groups, groupby was on sources only")
