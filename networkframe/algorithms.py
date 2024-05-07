from typing import Literal, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from tqdm.autonotebook import tqdm


def aggregate_over_graph(
    mask: np.ndarray,
    nodes: pd.DataFrame,
    aggregations: Union[str, list] = "mean",
    drop_self_in_neighborhood: bool = True,
    verbose: int = False,
    engine: Literal["auto", "scipy", "pandas"] = "auto",
) -> pd.DataFrame:
    if verbose > 0:
        print("Aggregating over neighborhoods")

    if engine == "auto":
        # TODO might also want to do a check on sparsity of the graph here
        if not all([isinstance(x, str) for x in aggregations]) or not all(
            [x in ["mean", "sum", "std"] for x in aggregations]
        ):
            engine = "pandas"
        else:
            engine = "scipy"

    if engine == "pandas":
        rows = []
        for iloc in tqdm(range(len(nodes)), disable=not verbose):
            # the selection here is pretty quick;
            node = nodes.index[iloc]
            select_nodes = nodes.loc[mask[iloc]]
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
    else: 
        raise ValueError(f"Unknown engine {engine}")

    return neighborhood_features
