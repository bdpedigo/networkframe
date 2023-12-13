# networkframe

[![pypi](https://img.shields.io/pypi/v/networkframe.svg)](https://pypi.org/project/networkframe/)
[![python](https://img.shields.io/pypi/pyversions/networkframe.svg)](https://pypi.org/project/networkframe/)
[![Build Status](https://github.com/bdpedigo/networkframe/actions/workflows/dev.yml/badge.svg)](https://github.com/bdpedigo/networkframe/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/bdpedigo/networkframe/branch/main/graphs/badge.svg)](https://codecov.io/github/bdpedigo/networkframe)

Lightweight representations of networks using Pandas DataFrames.

- Documentation: <https://bdpedigo.github.io/networkframe>
- GitHub: <https://github.com/bdpedigo/networkframe>
- PyPI: <https://pypi.org/project/networkframe/>
- Free software: MIT

`networkframe` uses Pandas DataFrames to represent networks in a lightweight way.
A `NetworkFrame` object is simply a table representing nodes and a table representing
edges, and a variety of methods to make querying and manipulating that data easy.

**Pros:**

- Lightweight: `NetworkFrame` objects are just two DataFrames, so they're easy to manipulate and integrate with other tools.
- Interoperable: can output to `NetworkX`, `numpy` and `scipy` sparse matrices, and other formats (coming soon).
- Flexible: can represent directed, undirected, and multigraphs.
- Familiar: if you're familiar with `Pandas` `DataFrames`, that is. As much as possible, `networkframe` uses the same syntax as `Pandas`, but also just gives you access to the underlying tables.
- Extensible: it's easy to use `NetworkFrame` as a base graph - for instance, you could make a `SpatialNetworkFrame` that adds spatial information to the nodes and edges.

**Cons:**

- No guarantees: since `networkframe` gives you access to the underlying `DataFrames`, it doesn't do much validation of the data.
- Not optimized for graph computations: since `networkframe` is storing data as simple node and edge tables, it's not optimized for doing actual computations on those graphs (e.g. like searching for shortest paths). A typical workflow would be to use `networkframe` to load and manipulate your data, then convert to a more graph-oriented format like `scipy` sparse matrices or `NetworkX` for computations.

**Room for improvement:**

- Early development: there are likely bugs and missing features. Please report any issues you find!
- More interoperability: `networkframe` can currently output to `NetworkX`, `numpy` and `scipy` sparse matrices, and other formats (coming soon). It would be nice to be able to read in from these formats as well.
- Graph-type handling: `networkframe` has mainly been tested on directed graphs, less so for undirected and multigraphs.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [bdpedigo/cookiecutter-pypackage](https://github.com/bdpedigo/cookiecutter-pypackage) project template (which builds on several previous versions).
