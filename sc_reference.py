import anndata
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def initialization(adata_sc: anndata.AnnData, adata_st: anndata.AnnData, min_genes: int = 200, min_cells: int = 200,
                   min_std: float = 80, normalize_st=False):
    """
    Filter single cell data and spatial data, and normalize the data to count per million (CPM).
    Args:
        adata_sc: Single cell data.
        adata_st: Spatial data.
        min_genes: Minimum number of genes expressed required for a cell to pass filtering.
        min_cells: Minimum number of cells expressed required for a gene to pass filtering.
        min_std: Minimum std of counts required for a gene to pass filtering after CPM normalization.
        normalize_st: If False, spatial data is also normalized to one million. Otherwise, normalized_st should be
                      np.ndarray, representing the number of cells in each spot, and the expression of each spot is
                      normalized to n_cell million.
    """
    if type(adata_sc.X) is not np.ndarray:
        adata_sc.X = adata_sc.X.toarray()
    if type(adata_st.X) is not np.ndarray:
        adata_st.X = adata_st.X.toarray()

    # CPM normalization
    adata_sc.X = adata_sc.X * 1e6 / adata_sc.X.sum(axis=1, keepdims=True)
    if not normalize_st:
        adata_st.X = adata_st.X * 1e6 / adata_st.X.sum(axis=1, keepdims=True)
    else:
        assert np.shape(normalize_st) == (len(adata_st),)
        adata_st.X = adata_st.X * 1e6 / adata_st.X.sum(axis=1, keepdims=True) * normalize_st[:, np.newaxis]
    # Initial filtering
    # One can also achieve the same result by using sc.pp.filter_cells(adata_sc1, min_genes=min_genes) and
    # sc.pp.filter_genes(adata_sc1, min_cells=min_cells). However, the code below is faster based on our test.
    gene_sum = np.sum(adata_sc.X > 0, axis=1)
    adata_sc = adata_sc[gene_sum > min_genes]
    cell_sum = np.sum(adata_sc.X > 0, axis=0)
    adata_sc = adata_sc[:, cell_sum > min_cells]
    adata_sc = adata_sc[:, np.std(adata_sc.X, axis=0) > min_std]

    # Find the intersection of genes
    common_genes = list(set(adata_st.var_names).intersection(set(adata_sc.var_names)))
    adata_sc = adata_sc[:, common_genes]
    adata_st = adata_st[:, common_genes]
    return adata_sc, adata_st


def marker_selection(adata_sc: anndata.AnnData, type_key: str, R: float = 2, threshold_cover=0.6, threshold_z=0,
                     n_select=40, verbose=0, return_dict=False):
    """
    Find marker genes based on pairwise ratio test.
    Args:
        adata_sc: scRNA data (Anndata).
        type_key: The key that is used to extract cell type information from adata_sc.obs.
        R: Ratio in hypothesis test
        threshold_cover: Minimum proportion of non-zero reads of a marker gene in assigned cell type.
        threshold_z: Minimum z-score in ratio tests for a gene to be marker gene.
        n_select: Number of marker genes selected for each cell type.
        verbose: 0: silent. 1: print the number of marker genes of each cell type.
        return_dict: If true, return a dictionary of marker genes, where the keys are the name of the cell types.
    Returns:
        marker_gene: List of the marker genes or a dictionary of marker genes, where the keys are the name of the cell
                     types.
    """
    if type(adata_sc.X) is np.ndarray:
        X = adata_sc.X
    else:
        X = adata_sc.X.toarray()

    # Derive mean and std matrix
    type_list = sorted(list(adata_sc.obs[type_key].unique()))  # list of the cell type.
    n_gene, n_type = adata_sc.shape[1], len(type_list)
    expression_mu = np.zeros((n_type, n_gene))  # Mean expression of each gene in each cell type.
    expression_sd = np.zeros((n_type, n_gene))  # Standard deviation of expression of each gene in each cell type.
    n_cell_by_type = np.zeros(n_type)
    data_type = []  # The expression data categorized by cell types.
    for i in range(n_type):
        data_type.append(X[adata_sc.obs[type_key] == type_list[i]])
        expression_mu[i] = np.mean(data_type[i], axis=0)
        expression_sd[i] = np.std(data_type[i], axis=0)
        n_cell_by_type[i] = len(data_type[i])
    del X

    # Ratio test
    z_score = np.zeros((n_type, n_gene))
    type_index_max = np.argmax(expression_mu, axis=0)  # Cell type index with the maximum mean expression of each gene
    for i in range(n_gene):
        mu0 = expression_mu[type_index_max[i], i]
        sd0 = expression_sd[type_index_max[i], i]
        n0 = n_cell_by_type[type_index_max[i]]
        z_score[:, i] = (mu0 - R * expression_mu[:, i]) / \
                        np.sqrt(sd0 ** 2 / n0 + R ** 2 * expression_sd[:, i] ** 2 / n_cell_by_type)

    # determine the marker genes
    z_score_sort = np.sort(z_score, axis=0)
    gene_name = np.array(adata_sc.var_names)
    marker_gene = dict() if return_dict else []
    for i in range(n_type):
        # fraction of non-zero reads in current datatype
        cover_fraction = np.sum(data_type[i][:, type_index_max == i] > 0, axis=0) / n_cell_by_type[i]
        gene_name_temp = gene_name[type_index_max == i][cover_fraction > threshold_cover]
        z_score_temp = z_score_sort[2, type_index_max == i][cover_fraction > threshold_cover]  # second smallest z-score
        selected_gene_idx = np.argsort(z_score_temp)[-n_select:]
        selected_gene = gene_name_temp[selected_gene_idx]
        if return_dict:
            marker_gene[type_list[i]] = list(selected_gene)
        else:
            marker_gene.extend(list(selected_gene))
        if verbose == 1:
            print(type_list[i] + ': {:d}'.format(len(list(selected_gene))))
    return marker_gene


def construct_sc_ref(adata_sc: anndata.AnnData, type_key: str):
    """
    Construct the scRNA reference from scRNA data.
    Args:
        adata_sc: scRNA data.
        type_key: The key that is used to extract cell type information from adata_sc.obs.
    Returns:
        sc_ref: scRNA reference. Numpy assay with dimension n_type*n_gene
    """
    type_list = sorted(list(adata_sc.obs[type_key].unique()))
    n_gene, n_type = adata_sc.shape[1], len(type_list)
    sc_ref = np.zeros((n_type, n_gene))
    for i, cell_type in enumerate(type_list):
        sc_X_temp = np.sum(adata_sc.X[adata_sc.obs[type_key]==cell_type], axis=0)
        sc_ref[i] = sc_X_temp/np.sum(sc_X_temp)
    return sc_ref


def plot_sc_ref(sc_ref, type_list, fig_size=(10, 4), dpi=300):
    """
    Plot the heatmap of the single cell reference
    Args:
        sc_ref: scRNA reference. np.ndarray n_type*n_gene.
        type_list: List of the cell types.
        fig_size: Size of the figure.
        dpi: Dots per inch (DPI) of the figure.
    """
    plt.figure(figsize=fig_size, dpi=dpi)
    sc_ref_df = pd.DataFrame(sc_ref, index=type_list)
    sns.heatmap(sc_ref_df, robust=True)
    plt.show()
