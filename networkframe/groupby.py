class NodeGroupBy:
    """A class for grouping a NetworkFrame by a set of labels."""

    def __init__(self, frame, source_groupby, target_groupby):
        """Groupby on nodes.

        Parameters
        ----------
        frame : _type_
            _description_
        source_groupby : _type_
            _description_
        target_groupby : _type_
            _description_
        """
        self._frame = frame
        self._source_groupby = source_groupby
        self._target_groupby = target_groupby

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

    # def apply(self, func, *args, data=False, **kwargs):
    #     """Apply a function to each group."""
    #     if self._axis == 'both':
    #         answer = pd.DataFrame(
    #             index=self.source_group_names, columns=self.target_group_names
    #         )

    #     else:
    #         if self._axis == 0:
    #             answer = pd.Series(index=self.source_group_names)
    #         else:
    #             answer = pd.Series(index=self.target_group_names)
    #     for group, frame in self:
    #         if data:
    #             value = func(frame.data, *args, **kwargs)
    #         else:
    #             value = func(frame, *args, **kwargs)
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
