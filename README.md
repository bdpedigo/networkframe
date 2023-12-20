# `networkframe`

[![pypi](https://img.shields.io/pypi/v/networkframe.svg)](https://pypi.org/project/networkframe/)
[![python](https://img.shields.io/pypi/pyversions/networkframe.svg)](https://pypi.org/project/networkframe/)
[![Build status](https://github.com/bdpedigo/networkframe/actions/workflows/report.yml/badge.svg)](https://github.com/bdpedigo/networkframe/actions/workflows/report.yml)
[![Downloads](https://static.pepy.tech/badge/networkframe)](https://pepy.tech/project/networkframe)

Lightweight representations of networks using Pandas DataFrames.

- Documentation: <https://bdpedigo.github.io/networkframe>
- GitHub: <https://github.com/bdpedigo/networkframe>
- PyPI: <https://pypi.org/project/networkframe/>
- Free software: MIT

`networkframe` uses Pandas DataFrames to represent networks in a lightweight way.
A `NetworkFrame` object is simply a table representing nodes and a table representing
edges, and a variety of methods to make querying and manipulating that data easy.

**Warning**: `networkframe` is still in early development, so there may be bugs and missing features. Please report any issues you find!

## Examples

Creating a `NetworkFrame` from scratch:

```python
import pandas as pd

from networkframe import NetworkFrame

nodes = pd.DataFrame(
    {
        "name": ["A", "B", "C", "D", "E"],
        "color": ["red", "blue", "blue", "red", "blue"],
    },
    index=[0, 1, 2, 3, 4],
)
edges = pd.DataFrame(
    {
        "source": [0, 1, 2, 2, 3],
        "target": [1, 2, 3, 1, 0],
        "weight": [1, 2, 3, 4, 5],
    }
)

nf = NetworkFrame(nodes, edges)
print(nf)
```

```text.python.console
NetworkFrame(nodes=(5, 2), edges=(5, 3))
```

Selecting a subgraph by node color

```python
red_nodes = nf.query_nodes("color == 'red'")
print(red_nodes.nodes)
```

```text.python.console
  name color
0    A   red
3    D   red
```

Selecting a subgraph by edge weight

```python
strong_nf = nf.query_edges("weight > 2")
print(strong_nf.edges)
```

```text.python.console
   source  target  weight
2       2       3       3
3       2       1       4
4       3       0       5
```

Iterating over subgraphs by node color

```python
for color, subgraph in nf.groupby_nodes("color", axis="both"):
    print(color)
    print(subgraph.edges)
```

```text.python.console
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
```

Applying node information to edges

```python
nf.apply_node_features("color", inplace=True)
print(nf.edges)
```

```text.python.console
   source  target  weight source_color target_color
0       0       1       1          red         blue
1       1       2       2         blue         blue
2       2       3       3         blue          red
3       2       1       4         blue         blue
4       3       0       5          red          red
```

## Is `networkframe` right for you?

**Pros:**

- Lightweight: `NetworkFrame` objects are just two DataFrames, so they're easy to manipulate and integrate with other tools.
- Interoperable: can output to `NetworkX`, `numpy` and `scipy` sparse matrices, and other formats (coming soon).
- Flexible: can represent directed, undirected, and multigraphs.
- Familiar: if you're familiar with `Pandas` `DataFrames`, that is. As much as possible, `networkframe` uses the same syntax as `Pandas`, but also just gives you access to the underlying tables.
- Extensible: it's easy to use `NetworkFrame` as a base graph - for instance, you could make a `SpatialNetworkFrame` that adds spatial information to the nodes and edges.

**Cons:**

- No guarantees: since `networkframe` gives you access to the underlying `DataFrames`, it doesn't do much validation of the data. This is by design, to keep it lightweight and flexible, but it means you can also mess up a `NetworkFrame` if you aren't careful (for instance, you could delete the index used to map edges to nodes).
- Not optimized for graph computations: since `networkframe` is storing data as simple node and edge tables, it's not optimized for doing actual computations on those graphs (e.g. like searching for shortest paths). A typical workflow would be to use `networkframe` to load and manipulate your data, then convert to a more graph-oriented format like `scipy` sparse matrices or `NetworkX` for computations.

**Room for improvement:**

- Early development: there are likely bugs and missing features. Please report any issues you find!
- More interoperability: `networkframe` can currently output to `NetworkX`, `numpy` and `scipy` sparse matrices, and other formats (coming soon). It would be nice to be able to read in from these formats as well.
- Graph-type handling: `networkframe` has mainly been tested on directed graphs, less so for undirected and multigraphs.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [bdpedigo/cookiecutter-pypackage](https://github.com/bdpedigo/cookiecutter-pypackage) project template (which builds on several previous versions).
