#!/usr/bin/env python
"""Tests for `networkframe` package."""

import pandas as pd
import pytest

from networkframe import NetworkFrame

# use pytest fixtures to create a NetworkFrame
# https://docs.pytest.org/en/latest/fixture.html


@pytest.fixture
def simple_networkframe():
    nodes = pd.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
            "color": ["red", "blue", "blue", "red"],
            "size": [1, 2, 3, 4],
        }
    )
    nodes.set_index("name", inplace=True)

    edges = pd.DataFrame(
        {
            "source": ["A", "A", "B", "C"],
            "target": ["B", "C", "C", "D"],
            "weight": [1, 2, 3, 4],
        }
    )

    return NetworkFrame(nodes, edges)


def test_initialize_networkframe(simple_networkframe):
    """Test the NetworkFrame class."""
    assert isinstance(simple_networkframe, NetworkFrame)


def test_indexing_error(simple_networkframe):
    nodes = simple_networkframe.nodes.reset_index()
    edges = simple_networkframe.edges
    with pytest.raises(ValueError):
        NetworkFrame(nodes, edges)


def test_nonunique_index(simple_networkframe):
    nodes = simple_networkframe.nodes.reset_index()
    nodes.loc[0, "name"] = "B"
    edges = simple_networkframe.edges
    with pytest.raises(ValueError):
        NetworkFrame(nodes.set_index("name"), edges)


def test_len(simple_networkframe):
    assert len(simple_networkframe) == 4


def test_query_nodes(simple_networkframe):
    assert len(simple_networkframe.query_nodes("color == 'red'")) == 2


def test_query_edges(simple_networkframe):
    query_networkframe = simple_networkframe.query_edges("weight > 2")
    assert len(query_networkframe.edges) == 2
