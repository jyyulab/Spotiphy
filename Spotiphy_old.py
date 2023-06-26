import anndata
import numpy as np
import scanpy as sc
import anndata as ad
from scipy.stats import ttest_ind

import torch
import matplotlib.pyplot as plt
import pyro
from tqdm import tqdm

import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO


def marker_selection(adata_sc: ad.AnnData, type_list: list, key: str, R: float=3, threshold_cover=0.6, threshold_z=2,
                     n_select=40, include_rest=False, verbose=0, return_dict=False):
    """
    Find marker genes based on ratio test.
    Args:
        adata_sc: scRNA data (Anndata).
        type_list: List of interested cell types in adata_sc.
        key: The key that is used to extract cell type information from adata_sc.obs.
        R: Ratio in hypothesis test
        threshold_cover: Minimum proportion of non-zero reads of a marker gene in assigned cell type.
        threshold_z: Minimum z-score in ratio tests for a gene to be marker gene.
        n_select: Number of marker genes selected for each cell type.
        include_rest: Whether the marker genes of the "rest" cell type are included.
        verbose: Print the number of marker genes of each cell type if verbose=1.
        return_dict: If true, return a dictionary of marker genes.
    Returns:
        marker_gene: list of the marker genes
    """

    # initial filtering
    # sc.pp.filter_cells(adata_sc, min_genes=200)  # filter 1
    # sc.pp.filter_genes(adata_sc, min_cells=200)  # filter 2
    if not type(adata_sc.X) is np.ndarray:
        X = adata_sc.X.toarray()
    else:
        X = adata_sc.X
    # var = np.var(X, axis=0)
    # adata_sc = adata_sc[:, var>0.025]  # filter 3
    # X = X[:, var>0.025]

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
    n_interested = len(type_list)+1 if rest else len(type_list)
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
            data_type.append(X[adata_sc.obs[key]==type_temp])
        n_type_record[i] = len(data_type[i])
        cell_type_mu[i] = np.mean(data_type[i], axis=0)
        cell_type_sd[i] = np.std(data_type[i], axis=0)

    if rest:  # "rest" is one interested cell type
        data_type.append(X[np.isin(adata_sc.obs[key], rest)])
        n_type_record[n_interested-1] = len(data_type[n_interested-1])
        cell_type_mu[n_interested-1] = np.mean(data_type[n_interested-1], axis=0)
        cell_type_sd[n_interested-1] = np.std(data_type[n_interested-1], axis=0)
    del X

    # ratio test
    z_score = np.zeros((n_interested, n_gene))
    type_idx = np.argmax(cell_type_mu, axis=0)
    for i in range(n_gene):
        mu0 = cell_type_mu[type_idx[i], i]
        sd0 = cell_type_sd[type_idx[i], i]
        n0 = n_type_record[type_idx[i]]
        z_score[:, i] = (mu0 - R*cell_type_mu[:, i])/ \
                        np.sqrt(sd0**2/n0 + R**2*cell_type_sd[:, i]**2/n_type_record)

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
        fraction = np.sum(data_type[i][:, type_idx==i]>1e-2, axis=0)/n_type_record[i]
        gene_name_temp = gene_name[type_idx==i][fraction>threshold_cover]
        z_score_temp = z_score_sort[2, type_idx==i][fraction>threshold_cover]  # second smallest z-score
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


def marker_ratio(adata_sc: ad.AnnData, type_list: list, key: str, R: float=3, threshold_cover=0.6, threshold_z=2, n_select=40, include_rest=False,
                 verbose=0, return_dict=False):
    """
    Find marker genes based on ratio test.
    Args:
        adata_sc: scRNA data (Anndata).
        type_list: List of interested cell types in adata_sc.
        key: The key that is used to extract cell type information from adata_sc.obs.
        R: Ratio in hypothesis test
        threshold_cover: Minimum proportion of non-zero reads of a marker gene in assigned cell type.
        threshold_z: Minimum z-score in ratio tests for a gene to be marker gene.
        n_select: Number of marker genes selected for each cell type.
        include_rest: Whether the marker genes of the "rest" cell type are included.
        verbose: Print the number of marker genes of each cell type if verbose=1.
        return_dict: If true, return a dictionary of marker genes.
    Returns:
        marker_gene: list of the marker genes
    """

    # initial filtering
    # sc.pp.filter_cells(adata_sc, min_genes=200)  # filter 1
    # sc.pp.filter_genes(adata_sc, min_cells=200)  # filter 2
    if not type(adata_sc.X) is np.ndarray:
        X = adata_sc.X.toarray()
    else:
        X = adata_sc.X
    # var = np.var(X, axis=0)
    # adata_sc = adata_sc[:, var>0.025]  # filter 3
    # X = X[:, var>0.025]

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
    n_interested = len(type_list)+1 if rest else len(type_list)
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
            data_type.append(X[adata_sc.obs[key]==type_temp])
        n_type_record[i] = len(data_type[i])
        cell_type_mu[i] = np.mean(data_type[i], axis=0)
        cell_type_sd[i] = np.std(data_type[i], axis=0)

    if rest:  # "rest" is one interested cell type
        data_type.append(X[np.isin(adata_sc.obs[key], rest)])
        n_type_record[n_interested-1] = len(data_type[n_interested-1])
        cell_type_mu[n_interested-1] = np.mean(data_type[n_interested-1], axis=0)
        cell_type_sd[n_interested-1] = np.std(data_type[n_interested-1], axis=0)
    del X

    # ratio test
    z_score = np.zeros((n_interested, n_gene))
    type_idx = np.argmax(cell_type_mu, axis=0)
    for i in range(n_gene):
        mu0 = cell_type_mu[type_idx[i], i]
        sd0 = cell_type_sd[type_idx[i], i]
        n0 = n_type_record[type_idx[i]]
        z_score[:, i] = (mu0 - R*cell_type_mu[:, i])/ \
                        np.sqrt(sd0**2/n0 + R**2*cell_type_sd[:, i]**2/n_type_record)

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
        fraction = np.sum(data_type[i][:, type_idx==i]>1e-2, axis=0)/n_type_record[i]
        gene_name_temp = gene_name[type_idx==i][fraction>threshold_cover]
        z_score_temp = z_score_sort[2, type_idx==i][fraction>threshold_cover]  # second smallest z-score
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


def learn_ref(adata_sc, type_list, key='Allen.subclass_label'):
    """
    Extract the scRNA reference from scRNA data.
    Args:
        adata_sc: scRNA data (Anndata).
        type_list: List of useful cell types in adata_sc. Cell types that should be combined are put in the same sublist.
                   E.g.:[['L2/3 IT', 'L6 CT', 'L5/6 NP', 'L6 IT', 'L5 IT', 'L5 ET', 'L6b'], 'SMC', ['Sst', 'Lamp5',
                        'Pvalb', 'Vip''Sncg'],'Oligo', 'Peri', 'OPC', 'VLMC', 'Endo', 'CR', 'Prog/IP']
        key: The key that is used to extract cell type information from adata_sc.obs.
    Returns:
        sc_ref: scRNA reference. Numpy assay with dimension n_type*n_gene
    """
    n_type = len(type_list)
    n_gene = adata_sc.shape[1]
    sc_ref = np.zeros((n_type, n_gene))
    for i, cell_type in enumerate(type_list):
        if type(cell_type) is list:
            idx = adata_sc.obs[key].isin(cell_type).to_numpy()
        else:
            idx = adata_sc.obs[key].to_numpy() == cell_type
        sc_X_temp = np.sum(adata_sc.X[idx], axis=0)
        sc_ref[i] = sc_X_temp/np.sum(sc_X_temp)
    return sc_ref


def deconvolute(X, s, sc_ref, device='cpu', plot=False, adata_params=None, n_steps=8000, option=2,
                adam_params=None):
    if adam_params is None:
        adam_params = {"lr": 0.003, "betas": (0.95, 0.999)}
    sc_ref = torch.tensor(sc_ref, device=device)
    X = X/np.sum(X, axis=1)[:, np.newaxis]
    for i in range(len(X)):
        X[i] = to_cell_count(X[i], s)
    X = torch.tensor(X, device=device, dtype=torch.int32)

    if option == 0:
        def model(sc_ref, s, X=None):
            n_spot = len(X)
            n_type = len(sc_ref)
            with pyro.plate("spot", n_spot):
                p = pyro.sample("Proportion", dist.Dirichlet(torch.full((n_type,), 1., device=device, dtype=torch.float64)))
                mu = torch.matmul(p, sc_ref)
                pyro.sample("Spatial RNA", dist.Multinomial(total_count=int(s), probs=mu), obs=X)
        def guide(sc_ref, s, X=None):
            n_spot = len(X)
            n_type = len(sc_ref)
            with pyro.plate("spot", n_spot):
                alpha = pyro.param('alpha', lambda: torch.full((n_spot,n_type), 2., device=device, dtype=torch.float64), constraint=constraints.positive)
                pyro.sample("Proportion", dist.Dirichlet(alpha))
    elif option == 1:
        def model(sc_ref, s, X):
            n_spot = len(X)
            n_type, n_gene = sc_ref.shape
            with pyro.plate("gene", n_gene, dim=-2):
                r = pyro.sample("Batch effect", dist.Gamma(torch.tensor(5., device=device), torch.tensor(5., device=device)))
            with pyro.plate("spot", n_spot, dim=-1):
                p = pyro.sample("Proportion", dist.Dirichlet(torch.full((n_type,), 1., device=device, dtype=torch.float64)))
                mu = torch.matmul(p, sc_ref)
                mu = mu * r.squeeze(1)
                mu = mu/mu.sum(dim=-1).unsqueeze(-1)
                pyro.sample("Spatial RNA", dist.Multinomial(total_count=int(s), probs=mu), obs=X)
        def guide(sc_ref, s, X):
            n_spot = len(X)
            n_type, n_gene = sc_ref.shape
            with pyro.plate("gene", n_gene, dim=-2):
                shape = pyro.param('shape', lambda: torch.full((n_gene,), 5., device=device, dtype=torch.float64), constraint=constraints.positive)
                rate = pyro.param('rate', lambda: torch.full((n_gene,), 5., device=device, dtype=torch.float64), constraint=constraints.positive)
                r = pyro.sample("Batch effect", dist.Gamma(shape.unsqueeze(-1), rate.unsqueeze(-1)))
            with pyro.plate("spot", n_spot):
                alpha = pyro.param('alpha', lambda: torch.full((n_spot,n_type), 2., device=device, dtype=torch.float64), constraint=constraints.positive)
                p = pyro.sample("Proportion", dist.Dirichlet(alpha))
    else:
        def model(sc_ref, s, X):
            n_spot = len(X)
            n_type, n_gene = sc_ref.shape
            r = pyro.sample("Batch effect", dist.Dirichlet(torch.full((n_gene,), 2., device=device, dtype=torch.float64)))
            with pyro.plate("spot", n_spot, dim=-1):
                p = pyro.sample("Proportion", dist.Dirichlet(torch.full((n_type,), 3., device=device, dtype=torch.float64)))
                mu = torch.matmul(p, sc_ref)
                mu = mu * r
                mu = mu/mu.sum(dim=-1).unsqueeze(-1)
                pyro.sample("Spatial RNA", dist.Multinomial(total_count=int(s), probs=mu), obs=X)
        def guide(sc_ref, s, X):
            n_spot = len(X)
            n_type, n_gene = sc_ref.shape
            beta = pyro.param('beta', lambda: torch.full((n_gene,), 2., device=device, dtype=torch.float64), constraint=constraints.positive)
            r = pyro.sample("Batch effect", dist.Dirichlet(beta))
            # print(shape.shape, r.shape)
            with pyro.plate("spot", n_spot):
                alpha = pyro.param('alpha', lambda: torch.full((n_spot,n_type), 2., device=device, dtype=torch.float64), constraint=constraints.positive)
                p = pyro.sample("Proportion", dist.Dirichlet(alpha))

    pyro.clear_param_store()
    if adata_params is None:
        adam_params = {"lr": 0.003, "betas": (0.95, 0.999)}
    optimizer = Adam(adam_params)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    losses = []
    for step in tqdm(range(n_steps)):
        loss = svi.step(sc_ref, s, X)
        losses.append(loss)
    if plot:
        plt.plot(losses)
    return pyro.get_param_store()


def deconvolute_single_bayesprism(X_s, ref, init=None, prior=None, conf=1):
    """
    Deconvolution method similar to BayesPrism. Only deconvolute a single spot or one aggregated spot.
    Args:
        X_s: RNA sequence of spot s.
        ref: Reference matrix of scRNA.
        init: Initialized estimation of mu.
    Returns:
        U: Estimated expression matrix. n_type*n_gene
        mu: Cell type proportion.
    """
    T = len(ref)
    if init is not None:
        mu = init.copy()
    elif prior is not None:
        mu = prior.copy()
    else:
        mu = np.ones(T)/T
    mu_old = np.ones(T)
    U = ref.copy()
    while np.sum(np.abs(mu-mu_old))>1e-6:
        mu_old = mu.copy()
        U_temp = np.reshape(mu, (-1, 1)) * ref
        U_temp_sum = np.sum(U_temp, axis=0)
        U_temp_sum[U_temp_sum<1e-5] = 1e-5
        U_temp = U_temp / U_temp_sum
        U = U_temp * X_s
        mu = np.sum(U, axis=1)
        if np.sum(mu) == 0:
            return U, np.ones(T)/T
        mu = mu/np.sum(mu)
        if prior is not None:
            mu = mu + prior * conf
            mu = mu/np.sum(mu)
    return U, mu


def deconvolute_single_bayesprism_prior(X_s, ref, init=None):
    """
    Deconvolution method similar to BayesPrism. Only deconvolute a single spot or one aggregated spot.
    Args:
        X_s: RNA sequence of spot s.
        ref: Reference matrix of scRNA.
        init: Initialized estimation of mu.
    Returns:
        U: Estimated expression matrix. n_type*n_gene
        mu: Cell type proportion.
    """
    T = len(ref)
    if init is not None:
        mu = init.copy()
    else:
        mu = np.ones(T)/T
    mu_old = np.ones(T)
    U = ref.copy()
    while np.sum(np.abs(mu-mu_old))>1e-5:
        mu_old = mu.copy()
        U_temp = np.reshape(mu, (-1, 1)) * ref
        U_temp_sum = np.sum(U_temp, axis=0)
        U_temp_sum[U_temp_sum<1e-5] = 1e-5
        U_temp = U_temp / U_temp_sum
        U = U_temp * X_s
        mu = np.sum(U, axis=1) + 1
        mu = mu/np.sum(mu)
    return U, mu


def deconvolute_bayesprism_1(data_st, sc_ref, init=None, verbose=0, key=None):
    """
    Deconvolution method similar to BayesPrism. Deconvolute all the spot.
    No special technique is used.
    Args:
        data_st: Spatial data(Anndata) of spatial expression matrix.
        ref: Reference matrix of scRNA.
        init: Initialized estimation of mu.
        verbose: 1:print program progress
    Returns:
        P: Proportion matrix. n_spot*n_type.
        U: Expression tensor. n_spot*n_type*n_gene.
    """
    n_type, n_gene = np.shape(sc_ref)
    if type(data_st) is ad.AnnData:
        if type(data_st.X) is np.ndarray:
            X0 = data_st.X
        else:
            X0 = data_st.X.toarray()
    else:
        X0 = data_st
    P = np.ones((len(data_st), n_type)) / n_type
    U = np.zeros((len(data_st), n_type, n_gene))

    for i in range(len(data_st)):
        if (i+1)%250 == 0 and verbose==1:
            print('>'*((i+1)//250)+str(i+1))

        X = X0[i]
        if not np.sum(X) == 0:
            # if key and type(data_st) is ad.AnnData:
            #     # normalized to n_s*10**6. However, this normalization does not affect the result mu.
            #     X = X/np.sum(X) * max(data_st.obs[key][i], 1)
            if init is not None:
                U[i], P[i] = deconvolute_single_bayesprism(X, sc_ref, init=init[i])
            else:
                U[i], P[i] = deconvolute_single_bayesprism(X, sc_ref)
    return P, U


def annotate_proportion(adata_st, P, type_list, suffix='_0'):
    assert(np.shape(P)[1] == len(type_list))
    for i, type0 in enumerate(type_list):
        adata_st.obs[type0+suffix] = P[:, i]


def deconvolute_bayesprism_2(adata_st, sc_ref, k=1, n=20, init=None, verbose=0, key='cell_count'):
    """
    Deconvolution method similar to BayesPrism. Deconvolute all the spot.
    We randomly select n spots. Then do deconvolution twice: once when we aggregate n+1 spots, once when we aggregate n spots.
    The difference between these two deconvoluton result will be the deconvolution of the current spot.
    Repeat the procedure above k times.
    Args:
        adata_st: Spatial data (Anndata).
        sc_ref: Reference matrix of scRNA.
        k: Repeat times.
        n: Number of selected spots in each round.
        init: Initialized estimation of mu.
        key: The key that used to extract cell count from adata_st.obs.
    Returns:
        P: Proportion matrix. n_spot*n_type.
        U: Expression tensor. n_spot*n_type*n_gene.
    """
    n_type, n_gene = np.shape(sc_ref)
    P = np.zeros((len(adata_st), n_type))
    U = np.zeros((len(adata_st), n_type, n_gene))
    if not type(adata_st.X) is np.ndarray:
        X0 = adata_st.X.toarray()
    else:
        X0 = adata_st.X
    for i in range(len(adata_st)):
        if (i+1)%100 == 0 and verbose==1:
            print('>'*((i+1)//100)+str(i+1))
        X = X0[i]
        X = X/np.sum(X) * max(adata_st.obs[key][i], 1)  # normalized to n_s*10**6. However, this normalization will not affect the result mu.

        n_s = max(adata_st.obs[key][i], 1)  # number of cells in current spot
        idx_agg = np.random.choice(np.arange(len(adata_st)), size=n, replace=False)
        X_agg = X0[idx_agg]
        n_agg = 0
        for j, idx in enumerate(idx_agg):
            X_agg[j] = X_agg[j]/np.sum(X_agg[j]) * max(adata_st.obs[key][idx], 1)
            n_agg += max(adata_st.obs[key][idx], 1)
        X_agg = np.sum(X_agg, axis=0)
        U_temp1, mu1 = deconvolute_single_bayesprism(X_agg+X, sc_ref, init=init)
        U_temp0, mu0 = deconvolute_single_bayesprism(X_agg, sc_ref, init=init)
        mu = mu1 * (n_agg+n_s) - mu0*n_agg
        P[i] = mu
        U[i] = U_temp1 - U_temp0
    return P, U


def deconvolute_bayesprism_2_cluster(adata_st, sc_ref, n_repeat=1, n_agg=20, init=None, verbose=0, key_count='cell_count',
                                     key_cluster='leiden', correction=False):
    """
    Deconvolution method similar to BayesPrism. Deconvolute all the spot.
    We randomly select n spots within the same cluster. Then do deconvolution twice: once when we aggregate n+1 spots, once when we aggregate n spots.
    The difference between these two deconvoluton result will be the deconvolution of the current spot.
    Repeat the procedure above n_repeat times.
    Args:
        adata_st: Spatial data (Anndata).
        sc_ref: Reference matrix of scRNA.
        n_repeat: Repeat times.
        n_agg: Number of selected spots in each round.
        init: Initialized estimation of mu.
        key_count: The key that used to extract cell count from adata_st.obs.
        correction: If correction is true, negative cell number will be change to 0.
    Returns:
        key_cluster: The key that used to extract cluster information from adata_st.obs.
        P: Proportion matrix. n_spot*n_type.
        U: Expression tensor. n_spot*n_type*n_gene.
    """
    n_type, n_gene = np.shape(sc_ref)
    P = np.zeros((len(adata_st), n_type))
    U = np.zeros((len(adata_st), n_type, n_gene))
    if not type(adata_st.X) is np.ndarray:
        X = adata_st.X.toarray()
    else:
        X = adata_st.X
    for i in range(len(adata_st)):
        if (i+1)%100 == 0 and verbose==1:
            print('>'*((i+1)//100)+str(i+1))
        X0 = X[i]
        X0 = X0/np.sum(X0) * max(adata_st.obs[key_count][i], 1)  # normalized to n_s*10**6. However, this normalization will not affect the result mu.

        n_s = max(adata_st.obs[key_count][i], 1)  # number of cells in current spot
        cluster = adata_st.obs.iloc[i, 1]
        cluster_idx = np.where(adata_st.obs[key_cluster]==cluster)[0]

        P_list = np.zeros((n_repeat, n_type))
        U_list = np.zeros((n_repeat, n_type, n_gene))
        for k in range(n_repeat):
            if len(cluster_idx) > n_agg:
                idx_agg = np.random.choice(cluster_idx, size=n_agg, replace=False)
            else:
                # idx_agg = np.delete(cluster_idx, i)
                idx_agg = cluster_idx
            X_agg = X[idx_agg]
            n_cell_agg = 0
            for j, idx in enumerate(idx_agg):
                X_agg[j] = X_agg[j]/np.sum(X_agg[j]) * max(adata_st.obs[key_count][idx], 1)
                n_cell_agg += max(adata_st.obs[key_count][idx], 1)
            X_agg = np.sum(X_agg, axis=0)
            U_temp1, mu1 = deconvolute_single_bayesprism(X_agg+X0, sc_ref, init=init)
            U_temp0, mu0 = deconvolute_single_bayesprism(X_agg, sc_ref, init=init)
            # mu = mu1 * (n_cell_agg+n_s) - mu0*n_cell_agg
            mu = to_cell_count(mu1, n_cell_agg+n_s) - to_cell_count(mu0, n_cell_agg)
            P_list[k] = mu
            U_list[k] = U_temp1 - U_temp0
        P[i] = np.mean(P_list, axis=0)
        U[i] = np.mean(U_list, axis=0)
    if correction:
        P[P < 0] = 0
    return P, U


def deconvolute_bayesprism_3(adata_st, sc_ref, n_agg=20, init=None, verbose=0, key_count='cell_count',
                             key_cluster='leiden', correction=False):
    """
    Deconvolution method similar to BayesPrism. Deconvolute all the spot.
    Repeat the procedure above n_repeat times.
    Use the deconvolution result of aggregated spots to update sc_ref.
    Args:
        adata_st: Spatial data (Anndata).
        sc_ref: Reference matrix of scRNA.
        n_agg: Number of selected spots in each round.
        init: Initialized estimation of mu.
        key_count: The key that used to extract cell count from adata_st.obs.
        correction: If correction is true, negative cell number will be change to 0.
    Returns:
        key_cluster: The key that used to extract cluster information from adata_st.obs.
        P: Proportion matrix. n_spot*n_type.
        U: Expression tensor. n_spot*n_type*n_gene.
    """
    n_type, n_gene = np.shape(sc_ref)
    P = np.zeros((len(adata_st), n_type))
    U = np.zeros((len(adata_st), n_type, n_gene))
    if not type(adata_st.X) is np.ndarray:
        X = adata_st.X.toarray()
    else:
        X = adata_st.X
    for i in range(len(adata_st)):
        if (i+1)%100 == 0 and verbose==1:
            print('>'*((i+1)//100)+str(i+1))
        X0 = X[i]
        X0 = X0/np.sum(X0) * max(adata_st.obs[key_count][i], 1)  # normalized to n_s*10**6. However, this normalization will not affect the result mu.

        n_s = max(adata_st.obs[key_count][i], 1)  # number of cells in current spot
        cluster = adata_st.obs.iloc[i, 1]
        cluster_idx = np.where(adata_st.obs[key_cluster]==cluster)[0]

        if len(cluster_idx) > n_agg:
            idx_agg = np.random.choice(cluster_idx, size=n_agg, replace=False)
        else:
            # idx_agg = np.delete(cluster_idx, i)
            idx_agg = cluster_idx
        X_agg = X[idx_agg]
        n_cell_agg = 0
        for j, idx in enumerate(idx_agg):
            X_agg[j] = X_agg[j]/np.sum(X_agg[j]) * max(adata_st.obs[key_count][idx], 1)
            n_cell_agg += max(adata_st.obs[key_count][idx], 1)
        X_agg = np.sum(X_agg, axis=0)
        U_temp, mu_temp = deconvolute_single_bayesprism(X_agg+X0, sc_ref, init=init)
        U_temp = U_temp/ np.sum(U_temp, axis=1)[:, np.newaxis]
        U[i], P[i] = deconvolute_single_bayesprism(X0, U_temp, init=init)
    if correction:
        P[P < 0] = 0
    return P, U


def to_cell_count(mu, n, verbose=0):
    '''
    Convert the cell proportion to the absolute cell number.
    '''
    assert(np.abs(np.sum(mu)-1)<1e-5)
    c0 = mu * n
    c = np.floor(c0)
    p = c0-c
    if np.sum(c) == n:
        return c
    idx = np.argsort(p)[-int(np.round(n-np.sum(c))):]
    sample = np.zeros(len(mu))
    sample[idx] = 1
    if verbose:
        print(np.sum(sample+c))
    return sample + c


def correlation(pred, target):
    assert(np.shape(pred) == np.shape(target))
    cor = np.sum(pred*target, axis=1)/np.sqrt(np.sum(pred**2, axis=1)*np.sum(target**2, axis=1))
    return cor


