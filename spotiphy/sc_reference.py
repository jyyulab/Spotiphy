import anndata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy import stats
import time
from typing import List, Dict, Tuple, Union
from typing import List, Dict, Union


def initialization(
    adata_sc: anndata.AnnData,
    adata_st: anndata.AnnData,
    min_genes: int = 200,
    min_cells: int = 200,
    min_std: float = 20,
    normalize_st: np.ndarray = None,
    filtering: bool = True,
    verbose: int = 0,
) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """
    Filter single cell and spatial data, normalize to CPM, and optionally filter genes/cells.

    Args:
        adata_sc: Single cell data (AnnData).
        adata_st: Spatial data (AnnData).
        min_genes: Minimum genes per cell for filtering.
        min_cells: Minimum cells per gene for filtering.
        min_std: Minimum std for gene filtering after CPM normalization.
        normalize_st: If None, spatial data is normalized to one million. If ndarray, represents cell counts per spot.
        filtering: Whether to filter genes/cells in adata_sc.
        verbose: Verbosity level.

    Returns:
        Tuple of filtered and normalized (adata_sc, adata_st).
    """
    time_start = time.time()
    X = adata_sc.X if isinstance(adata_sc.X, np.ndarray) else adata_sc.X.toarray()
    Y = adata_st.X if isinstance(adata_st.X, np.ndarray) else adata_st.X.toarray()
    if verbose == 1:
        print(
            f"Convert expression matrix to array: {np.round(time.time()-time_start, 2)}s"
        )
        time_start = time.time()

    # CPM normalization
    X = X * 1e6 / X.sum(axis=1, keepdims=True)
    if normalize_st is None:
        Y = Y * 1e6 / Y.sum(axis=1, keepdims=True)
    else:
        if not isinstance(normalize_st, np.ndarray) or normalize_st.shape != (
            len(adata_st),
        ):
            raise ValueError("normalize_st must be a numpy array with shape (n_spots,)")
        Y = Y * 1e6 / Y.sum(axis=1, keepdims=True) * normalize_st[:, np.newaxis]
    if verbose == 1:
        print(f"Normalization: {np.round(time.time()-time_start, 2)}s")
        time_start = time.time()
    adata_st.X = Y

    # Initial filtering
    if filtering:
        cell_f1 = np.sum(X > 0, axis=1) > min_genes
        gene_f1 = np.sum(X > 0, axis=0) > min_cells
        gene_f2 = np.std(X, axis=0) > min_std
        adata_sc.X = X
        adata_sc = adata_sc[cell_f1]
        adata_sc = adata_sc[:, np.logical_and(gene_f1, gene_f2)]
        if verbose == 1:
            print(f"Filtering: {np.round(time.time()-time_start, 2)}s")
            time_start = time.time()
    else:
        adata_sc.X = X

    # Find the intersection of genes
    common_genes = list(set(adata_st.var_names).intersection(set(adata_sc.var_names)))
    adata_sc = adata_sc[:, common_genes]
    adata_st = adata_st[:, common_genes]
    if verbose == 1:
        print(f"Find common genes: {np.round(time.time()-time_start, 2)}s")
    return adata_sc, adata_st


def marker_selection(
    adata_sc: anndata.AnnData,
    key_type: str,
    threshold_cover: float = 0.6,
    threshold_p: float = 0.1,
    threshold_fold: float = 1.5,
    n_select: int = 40,
    verbose: int = 0,
    return_dict: bool = False,
    q: float = 0.0,
) -> Union[List[str], Dict[str, List[str]]]:
    """
    Find marker genes based on pairwise ratio test.

    Args:
        adata_sc: scRNA data (AnnData).
        key_type: Key for cell type in adata_sc.obs.
        threshold_cover: Min proportion of non-zero reads in cell type.
        threshold_p: Max p-value for marker gene.
        threshold_fold: Min fold change for marker gene.
        n_select: Number of marker genes per cell type.
        verbose: Verbosity level.
        return_dict: If True, return dict of marker genes by cell type.
        q: Quantile for fold-change selection.

    Returns:
        List or dict of marker genes.
    """
    X = adata_sc.X if isinstance(adata_sc.X, np.ndarray) else adata_sc.X.toarray()

    # Derive mean and std matrix
    type_list = sorted(list(adata_sc.obs[key_type].unique()))  # list of the cell type.
    n_gene, n_type = adata_sc.shape[1], len(type_list)
    expression_mu = np.zeros(
        (n_type, n_gene)
    )  # Mean expression of each gene in each cell type.
    expression_sd = np.zeros(
        (n_type, n_gene)
    )  # Standard deviation of expression of each gene in each cell type.
    n_cell_by_type = np.zeros(n_type)
    data_type = []  # The expression data categorized by cell types.
    for i in range(n_type):
        data_type.append(X[adata_sc.obs[key_type] == type_list[i]])
        expression_mu[i] = np.mean(data_type[i], axis=0)
        expression_sd[i] = np.std(data_type[i], axis=0)
        n_cell_by_type[i] = len(data_type[i])
    del X
    expression_sd = expression_sd + 1e-10
    expression_mu = expression_mu + 1e-10

    # t test
    fold_change = np.zeros((n_type, n_gene))
    p_value = np.zeros((n_type, n_gene))
    type_index_max = np.argmax(
        expression_mu, axis=0
    )  # Cell type index with the maximum mean expression of each gene
    for i in range(n_gene):
        mu0 = expression_mu[type_index_max[i], i]
        sd0 = expression_sd[type_index_max[i], i]
        n0 = n_cell_by_type[type_index_max[i]]
        A = sd0**2 / n0 + expression_sd[:, i] ** 2 / n_cell_by_type
        B = (sd0**2 / n0) ** 2 / (n0 - 1) + (
            expression_sd[:, i] ** 2 / n_cell_by_type
        ) ** 2 / (n_cell_by_type - 1)
        t_stat = (mu0 - expression_mu[:, i]) / np.sqrt(A)
        fold_change[:, i] = mu0 / expression_mu[:, i]
        df = A**2 / B
        p_value[:, i] = stats.t.sf(abs(t_stat), df)

    # determine the marker genes
    p_value_sort = np.sort(p_value, axis=0)
    fold_change_sort = np.sort(fold_change, axis=0)
    gene_name = np.array(adata_sc.var_names)
    marker_gene = dict() if return_dict else []
    for i in range(n_type):
        # fraction of non-zero reads in current datatype
        cover_fraction = (
            np.sum(data_type[i][:, type_index_max == i] > 0, axis=0) / n_cell_by_type[i]
        )
        p_value_temp = p_value_sort[-2, type_index_max == i]  # second-largest p-value
        # fold_change_temp = fold_change_sort[1, type_index_max == i]  # second-smallest fold change
        fold_change_temp = fold_change_sort[
            max(1, int(np.round(q * (n_type - 1)))), type_index_max == i
        ]
        selected = np.logical_and(
            cover_fraction >= threshold_cover, p_value_temp < threshold_p
        )
        selected = np.logical_and(fold_change_temp >= threshold_fold, selected)
        gene_name_temp = gene_name[type_index_max == i][selected]

        fold_change_temp = fold_change_temp[selected]
        selected_gene_idx = np.argsort(fold_change_temp)[::-1][:n_select]
        selected_gene = gene_name_temp[selected_gene_idx]

        if return_dict:
            marker_gene[type_list[i]] = list(selected_gene)
        else:
            marker_gene.extend(list(selected_gene))
        if verbose == 1:
            print(f"{type_list[i]}: {len(list(selected_gene)):d}")
        elif len(list(selected_gene)) < 5:
            print(
                f"Warning: Only {len(list(selected_gene)):d} genes are selected for {type_list[i]}."
            )
    return marker_gene


def construct_sc_ref(adata_sc: anndata.AnnData, key_type: str) -> np.ndarray:
    """
    Construct the scRNA reference from scRNA data.

    Args:
        adata_sc: scRNA data (AnnData).
        key_type: Key for cell type in adata_sc.obs.

    Returns:
        scRNA reference (np.ndarray, shape n_type x n_gene).
    """
    type_list = sorted(list(adata_sc.obs[key_type].unique()))
    n_gene, n_type = adata_sc.shape[1], len(type_list)
    sc_ref = np.zeros((n_type, n_gene))
    X = np.array(adata_sc.X)
    for i, cell_type in tqdm(enumerate(type_list)):
        sc_X_temp = np.sum(X[adata_sc.obs[key_type] == cell_type], axis=0)
        sc_ref[i] = sc_X_temp / np.sum(sc_X_temp)
    return sc_ref


def plot_sc_ref(
    sc_ref: np.ndarray,
    type_list: List[str],
    fig_size: Tuple[int, int] = (10, 4),
    dpi: int = 300,
) -> None:
    """
    Plot the heatmap of the single cell reference.

    Args:
        sc_ref: scRNA reference (np.ndarray, n_type x n_gene).
        type_list: List of cell types.
        fig_size: Figure size.
        dpi: Dots per inch.
    """
    fig_size_adjust = (fig_size[0], fig_size[1] * sc_ref.shape[0] / 20)
    plt.figure(figsize=fig_size_adjust, dpi=dpi)
    sc_ref_df = pd.DataFrame(sc_ref, index=type_list)
    sns.heatmap(sc_ref_df, robust=True)
    plt.show()


def plot_heatmap(
    adata_sc: anndata.AnnData,
    key_type: str,
    fig_size: Tuple[int, int] = (10, 4),
    dpi: int = 300,
    save: bool = False,
    out_dir: str = "",
) -> None:
    """
    Plot the heatmap of the mean expression.

    Args:
        adata_sc: scRNA data (AnnData).
        key_type: Key for cell type in adata_sc.obs.
        fig_size: Figure size.
        dpi: Dots per inch.
        save: Whether to save the heatmap.
        out_dir: Output directory.
    """
    X = adata_sc.X if isinstance(adata_sc.X, np.ndarray) else adata_sc.X.toarray()
    type_list = sorted(list(adata_sc.obs[key_type].unique()))  # list of the cell type.
    n_gene, n_type = adata_sc.shape[1], len(type_list)
    expression_mu = np.zeros(
        (n_type, n_gene)
    )  # Mean expression of each gene in each cell type.

    for i in range(n_type):
        data_temp = X[adata_sc.obs[key_type] == type_list[i]]
        expression_mu[i] = np.mean(data_temp, axis=0)
    del X

    expression_mu = expression_mu / np.max(expression_mu, axis=0)
    fig_size_adjust = (fig_size[0], fig_size[1] * n_type / 20)
    plt.figure(figsize=fig_size_adjust, dpi=dpi)
    sc_ref_df = pd.DataFrame(expression_mu, index=type_list)
    sns.heatmap(sc_ref_df, robust=True)
    if save:
        plt.savefig(out_dir + "heatmap.jpg", bbox_inches="tight")
    plt.show()
