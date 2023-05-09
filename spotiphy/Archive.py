import anndata
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
from tqdm import tqdm
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO


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
        n_significant = np.sum(z_score_temp > threshold_z)  # number of genes that pass the ratio test
        n_select_temp = max(1, min(n_select, n_significant))  # number of marker gene we select
        selected_gene_idx = np.argsort(z_score_temp)[-n_select_temp:]
        selected_gene = gene_name_temp[selected_gene_idx]
        if return_dict:
            marker_gene[type_list[i]] = list(selected_gene)
        else:
            marker_gene.extend(list(selected_gene))
        if verbose == 1:
            print(type_list[i] + ': {:d}'.format(len(list(selected_gene))))
    return marker_gene


def deconvolute_old(X, s, sc_ref, device='gpu', plot=False, adata_params=None, n_steps=8000, option=2,
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