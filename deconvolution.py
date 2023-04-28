import matplotlib.pyplot as plt
import numpy as np
import torch
import pyro
from tqdm import tqdm
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO


def deconvolute(X, sc_ref, device='cuda', n_epoch=8000, adam_params=None, plot=False, fig_size=(4.8, 3.6), dpi=200):
    """
    Deconvolution of the proportion of genes contributed by each cell type.

    Args:
        X: Spatial transcriptomics data. n_spot*n_gene.
        sc_ref: Single cell reference. n_type*n_gene.
        device: The device used for the deconvolution.
        plot: Whether to plot the ELBO loss.
        n_epoch: Number of training epochs.
        adam_params: Parameters for the adam optimizer.
        fig_size: Size of the figure.
        dpi: Dots per inch (DPI) of the figure.
    """
    if adam_params is None:
        adam_params = {"lr": 0.003, "betas": (0.95, 0.999)}
    X = torch.tensor(X, device=device, dtype=torch.int32)
    sc_ref = torch.tensor(sc_ref, device=device)
    assert X.shape[1] == sc_ref.shape[1], "Spatial data and SC reference data must have the same number of genes."

    def model(sc_ref, X):
        n_spot = len(X)
        n_type, n_gene = sc_ref.shape
        alpha = pyro.sample("Batch effect", dist.Dirichlet(torch.full((n_gene,), 2., device=device,
                                                                      dtype=torch.float64)))
        with pyro.plate("spot", n_spot, dim=-1):
            q = pyro.sample("Proportion", dist.Dirichlet(torch.full((n_type,), 3., device=device, dtype=torch.float64)))
            rho = torch.matmul(q, sc_ref)
            rho = rho * alpha
            rho = rho/rho.sum(dim=-1).unsqueeze(-1)
            pyro.sample("Spatial RNA", dist.Multinomial(total_count=10**6, probs=rho), obs=X)

    def guide(sc_ref, X):
        n_spot = len(X)
        n_type, n_gene = sc_ref.shape
        pi = pyro.param('pi', lambda: torch.full((n_gene,), 2., device=device, dtype=torch.float64),
                        constraint=constraints.positive)
        alpha = pyro.sample("Batch effect", dist.Dirichlet(pi))
        with pyro.plate("spot", n_spot):
            sigma = pyro.param('sigma', lambda: torch.full((n_spot, n_type), 2., device=device, dtype=torch.float64),
                               constraint=constraints.positive)
            q = pyro.sample("Proportion", dist.Dirichlet(sigma))

    pyro.clear_param_store()
    optimizer = Adam(adam_params)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    loss_history = []
    for _ in tqdm(range(n_epoch)):
        loss = svi.step(sc_ref, X)
        loss_history.append(loss)
    if plot:
        plt.figure(figsize=fig_size, dpi=dpi)
        plt.plot(loss_history, lw=0.5)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()
    return pyro.get_param_store()
