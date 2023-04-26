import anndata
import numpy as np
import scanpy as sc


def sc_initialization(adata_sc: anndata.AnnData, min_genes: int = 200, min_cells: int = 200, min_var: float = 1):
    """
    Initial filtering of the single cell data and normalize the data to count per million (CPM).
    Args:
        adata_sc: Single cell data.
        min_genes: Minimum number of counts required for a cell to pass filtering.
        min_cells: Minimum number of counts required for a gene to pass filtering.
        min_var: Minimum variance of counts required for a gene to pass filtering.
    """
    adata_sc = adata_sc
    sc.pp.filter_cells(adata_sc, min_genes=min_genes)
    sc.pp.filter_genes(adata_sc, min_cells=min_cells)
    if not isinstance(adata_sc.X, np.ndarray):
        adata_sc.X = adata_sc.X.toarray()
    var = np.var(adata_sc.X, axis=0)
    adata_sc = adata_sc[:, var > min_var]
    adata_sc.X = adata_sc.X/adata_sc.X.sum(axis=1, keepdims=True)
    return adata_sc


def marker_selection(adata_sc: anndata.AnnData, type_key: str, R: float = 3, threshold_cover=0.6, threshold_z=2,
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
        marker_gene: list of the marker genes
    """
    if not isinstance(adata_sc.X, np.ndarray):
        adata_sc.X = adata_sc.X.toarray()
    X = adata_sc.X.copy()

    # derive mean and std matrix




    cell_type_unlist = []
    cluster_label = []
    for cluster in type_list:
        if type(cluster) is list:
            cell_type_unlist += cluster
            cluster_label.append(' + '.join(cluster))
        else:
            cell_type_unlist.append(cluster)
            cluster_label.append(cluster)
    print('unlist cell types: ' + str(cell_type_unlist))
    cell_type = list(adata_sc.obs[key].unique())
    rest = list(set(cell_type) - set(cell_type_unlist))
    n_interested = len(type_list) + 1 if rest else len(type_list)
    n_gene = adata_sc.shape[1]
    cell_type_mu = np.zeros((n_interested, n_gene))
    cell_type_sd = np.zeros((n_interested, n_gene))
    n_type_record = np.zeros(n_interested)
    data_type = []
    for i in range(len(type_list)):
        type_temp = type_list[i]
        if type(type_temp) is list:
            data_type.append(X[np.isin(adata_sc.obs[key], type_temp)])
        else:
            data_type.append(X[adata_sc.obs[key] == type_temp])
        n_type_record[i] = len(data_type[i])
        cell_type_mu[i] = np.mean(data_type[i], axis=0)
        cell_type_sd[i] = np.std(data_type[i], axis=0)

    if rest:  # "rest" is one interested cell type
        data_type.append(X[np.isin(adata_sc.obs[key], rest)])
        n_type_record[n_interested - 1] = len(data_type[n_interested - 1])
        cell_type_mu[n_interested - 1] = np.mean(data_type[n_interested - 1], axis=0)
        cell_type_sd[n_interested - 1] = np.std(data_type[n_interested - 1], axis=0)
    del X

    # ratio test
    z_score = np.zeros((n_interested, n_gene))
    type_idx = np.argmax(cell_type_mu, axis=0)
    for i in range(n_gene):
        mu0 = cell_type_mu[type_idx[i], i]
        sd0 = cell_type_sd[type_idx[i], i]
        n0 = n_type_record[type_idx[i]]
        z_score[:, i] = (mu0 - R * cell_type_mu[:, i]) / \
                        np.sqrt(sd0 ** 2 / n0 + R ** 2 * cell_type_sd[:, i] ** 2 / n_type_record)

    # determine the marker genes
    z_score_sort = np.sort(z_score, axis=0)
    gene_name = np.array(adata_sc.var_names)
    if return_dict:
        marker_gene = dict()
    else:
        marker_gene = []
    if include_rest:
        n_interested_correct = n_interested
    else:
        n_interested_correct = len(type_list)
    for i in range(n_interested_correct):
        # fraction of non-zero reads in current datatype
        fraction = np.sum(data_type[i][:, type_idx == i] > 1e-2, axis=0) / n_type_record[i]
        gene_name_temp = gene_name[type_idx == i][fraction > threshold_cover]
        z_score_temp = z_score_sort[2, type_idx == i][fraction > threshold_cover]  # second smallest z-score
        n_significant = np.sum(z_score_temp > threshold_z)  # number of genes that pass the ratio test
        n_select_temp = max(1, min(n_select, n_significant))  # number of marker gene we select
        selected_gene_idx = np.argsort(z_score_temp)[-n_select_temp:]
        selected_gene = gene_name_temp[selected_gene_idx]
        if return_dict:
            marker_gene[cluster_label[i]] = list(selected_gene)
        else:
            marker_gene.extend(list(selected_gene))
        if verbose:
            print(type_list[i] if i < len(type_list) else 'rest', len(list(selected_gene)))
    return marker_gene
