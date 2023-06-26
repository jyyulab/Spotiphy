import anndata
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import pyro
from tqdm import tqdm
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from scipy.spatial.distance import jensenshannon

from . import sc_reference


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

    Returns:
        Parameters in the generative model.
    """
    if adam_params is None:
        adam_params = {"lr": 0.003, "betas": (0.95, 0.999)}
    max_exp = int(np.max(np.sum(X, axis=1)))
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
            rho = rho / rho.sum(dim=-1).unsqueeze(-1)
            pyro.sample("Spatial RNA", dist.Multinomial(total_count=max_exp, probs=rho), obs=X)

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


def proportion_to_count(p, n):
    """
    Convert the cell proportion to the absolute cell number.

    Args:
        p: Cell proportion.
        n: Number of cells.

    Returns:
        Cell count of each cell type.
    """
    assert (np.abs(np.sum(p) - 1) < 1e-5)
    c0 = p * n
    count = np.floor(c0)
    r = c0 - count
    if np.sum(count) == n:
        return count
    idx = np.argsort(r)[-int(np.round(n - np.sum(count))):]
    count[idx] += 1
    return count


def simulation(adata_st: anndata.AnnData, adata_sc: anndata.AnnData, key_type: str, cell_proportion: np.ndarray,
               n_cell=10, batch_effect_sigma=0.1, zero_proportion=0.3, additive_noise_sigma=0.05, save=True,
               out_dir=''):
    """
    Simulation of the spatial transcriptomics data based on a real spatial sample and deconvolution results of Spotiphy.

    Args:
        adata_st: Original spatial transcriptomics data.
        adata_sc: Original single-cell data.
        key_type: The key that is used to extract cell type information from adata_sc.obs.
        cell_proportion: Proportion of each cell type obtained by the deconvolution.
        n_cell: Number of cells in each spot, either a key of adata_st.obs or a positive integer.
        batch_effect_sigma: Sigma of the log-normal distribution when generate batch effect.
        zero_proportion: Proportion of gene expression set to 0. Note that since some gene expression in the original
                         X is already 0, the final proportion of 0 gene read is larger than zero_proportion.
        additive_noise_sigma: Sigma of the log-normal distribution when generate additive noise.
        save: If True, save the generated adata_st as a file.
        out_dir: Output directory.
    Returns:
        Simulated ST Anndata.
    """
    # Construct ground truth
    type_list = sorted(list(adata_sc.obs[key_type].unique()))  # list of the cell type.
    assert len(type_list) == cell_proportion.shape[1]
    assert len(cell_proportion) == len(adata_st)
    n_spot = len(adata_st)
    n_type = len(type_list)
    if isinstance(n_cell, (int, float)):
        assert n_cell >= 1
        n_cell = np.array([int(n_cell)] * n_spot)
    else:
        n_cell = adata_st.obs[n_cell].values.astype(int)
        n_cell[n_cell <= 0] = 1
    cell_count = np.zeros(cell_proportion.shape)
    for i in range(n_spot):
        cell_count[i] = proportion_to_count(cell_proportion[i], n_cell[i])
    cell_count = cell_count.astype(int)
    adata_st.obsm['ground_truth'] = cell_count
    adata_st.obs['cell_count'] = n_cell
    adata_st.uns['type_list'] = type_list

    # Construct expression matrix
    common_genes = list(set(adata_st.var_names).intersection(set(adata_sc.var_names)))
    adata_sc = adata_sc[:, common_genes]
    adata_st = adata_st[:, common_genes]
    n_gene = len(common_genes)
    X = np.zeros(adata_st.shape)
    if type(adata_sc.X) is np.ndarray:
        Y = adata_sc.X
    else:
        Y = adata_sc.X.toarray()
    Y = Y * 1e6 / np.sum(Y, axis=1, keepdims=True)
    type_index = []
    for i in range(n_type):
        type_index.append(np.where(adata_sc.obs[key_type] == type_list[i])[0])
    for i in range(n_spot):
        for j in range(n_type):
            if cell_count[i, j] > 0:
                X_temp = np.array(np.sum(Y[np.random.choice(type_index[j], cell_count[i, j])], axis=0))
                if X_temp.ndim > 1:
                    X[i] += X_temp[0]
                else:
                    X[i] += X_temp

    # Add noise
    # Batch effect
    batch_effect = np.random.lognormal(0, batch_effect_sigma, size=n_gene)
    X = X * batch_effect
    # Zero reads
    zero_read = np.random.binomial(1, 1 - zero_proportion, size=X.shape)
    X = X * zero_read
    # Additive noise
    additive_noise = np.random.lognormal(0, additive_noise_sigma, size=X.shape)
    X = X * additive_noise
    adata_st.X = X
    adata_st.uns['batch_effect'] = batch_effect

    if save:
        if out_dir and not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_dir = out_dir + '/' if out_dir else ''
        adata_st.write(out_dir + "Simulated_ST.h5ad")

    return adata_st


class Evaluation:
    def __init__(self, proportion_truth, proportion_estimated_list, methods, out_dir="", cluster=None, type_list=None):
        """
        Args:
            proportion_truth: Ground truth of the cell proportion.
            proportion_estimated_list: List of estimated proportion by each method.
            methods: List of methods names.
            out_dir: Output directory.
            cluster: Cluster label of each spot.
            type_list: List of cell types
        """
        self.proportion_truth = proportion_truth
        self.proportion_estimated_list = proportion_estimated_list
        self.methods = methods
        self.metric_dict = dict()  # Save the metric values.
        self.n_method = len(methods)
        self.n_type = proportion_truth.shape[1]
        if out_dir:
            self.out_dir = out_dir if out_dir[-1] == '/' else out_dir + '/'
        else:
            self.out_dir = ''
        if self.out_dir and not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        if not os.path.exists(out_dir+'figures'):
            os.mkdir(out_dir+'figures')
        self.cluster = cluster
        self.type_list = type_list
        self.function_map = {'Cosine similarity': self.cosine, 'Absolute error spot': self.absolute_error_spot,
                             'Square error spot': self.square_error_spot, 'JSD': self.JSD,
                             'Correlation': self.correlation_coef,
                             'Fraction of cells correctly mapped': self.correct_fraction,
                             'Absolute error': self.absolute_error}
        self.spot_metric_names = {'Cosine similarity', 'Absolute error spot', 'Square error spot', 'JSD'}
        self.general_metric_names = {'Absolute error', 'Square error'}
        assert len(proportion_estimated_list) == self.n_method

    @staticmethod
    def cosine(proportion_truth: np.ndarray, proportion_estimated: np.ndarray):
        """
        [spot metric]Calculate the cosine similarity between the true proportion and estimated proportion of each spot.

        Args:
            proportion_truth: Ground truth of the cell proportion.
            proportion_estimated: Estimated cell proportion.

        Returns:
            Cosine similarity of each spot.
        """
        assert proportion_truth.shape == proportion_estimated.shape
        cosine_similarity = np.sum(proportion_truth * proportion_estimated, axis=1) / \
                            np.linalg.norm(proportion_estimated, axis=1) / np.linalg.norm(proportion_truth, axis=1)
        return cosine_similarity

    @staticmethod
    def absolute_error_spot(proportion_truth: np.ndarray, proportion_estimated: np.ndarray):
        """[spot metric]"""
        assert proportion_truth.shape == proportion_estimated.shape
        error = np.sum(np.abs(proportion_truth-proportion_estimated), axis=1)
        return error

    @staticmethod
    def square_error_spot(proportion_truth: np.ndarray, proportion_estimated: np.ndarray):
        """[spot metric]"""
        assert proportion_truth.shape == proportion_estimated.shape
        error = np.sum((proportion_truth-proportion_estimated)**2, axis=1)
        return error

    @staticmethod
    def JSD(proportion_truth: np.ndarray, proportion_estimated: np.ndarray):
        """
        [spot metric]Jensen–Shannon divergence.
        """
        assert proportion_truth.shape == proportion_estimated.shape
        return jensenshannon(proportion_truth, proportion_estimated, axis=1)

    @staticmethod
    def correlation_coef(proportion_truth: np.ndarray, proportion_estimated: np.ndarray):
        """
        [cell type metric]Pearson correlation coefficient.
        """
        assert proportion_truth.shape == proportion_estimated.shape
        proportion_truth_centered = proportion_truth - np.mean(proportion_truth, axis=0)
        proportion_estimated_centered = proportion_estimated - np.mean(proportion_estimated, axis=0)
        correlations = np.sum(proportion_truth_centered * proportion_estimated_centered, axis=0) / \
                       np.sqrt(np.sum(proportion_truth_centered**2, axis=0) *
                               np.sum(proportion_estimated_centered**2, axis=0))
        return correlations

    @staticmethod
    def correct_fraction(proportion_truth: np.ndarray, proportion_estimated: np.ndarray):
        """
        [cell type metric]Jensen–Shannon divergence.
        """
        assert proportion_truth.shape == proportion_estimated.shape
        correct_proportion = np.minimum(proportion_truth, proportion_estimated)
        return np.sum(correct_proportion, axis=0)/np.sum(proportion_truth, axis=0)

    @staticmethod
    def absolute_error(proportion_truth: np.ndarray, proportion_estimated: np.ndarray):
        """[general metric]"""
        assert proportion_truth.shape == proportion_estimated.shape
        error = np.abs(proportion_truth-proportion_estimated)
        return error

    def evaluate_metric(self, metric='Cosine similarity'):
        metric_values = []
        func = self.function_map.get(metric)
        for i in range(self.n_method):
            metric_values.append(func(self.proportion_truth, self.proportion_estimated_list[i]))
        self.metric_dict[metric] = metric_values
        return metric_values

    def plot_metric_spot(self, save=False, region=None, metric='Cosine similarity'):
        """
        Plot the box plot of each method based on the metric. Each value in box plot represents a spot.
        Box number equals to the number of methods.
        """
        assert metric in self.spot_metric_names
        plt.figure(dpi=300)
        if metric not in self.metric_dict.keys():
            self.evaluate_metric(metric=metric)
        sns.set_palette("Dark2")
        n_spot = len(self.proportion_truth)
        region_name = ''
        if region is None:
            metric_values = np.concatenate(self.metric_dict[metric])
            methods_name = np.repeat(self.methods, n_spot)
        else:
            if isinstance(region, list):
                select = np.array([i in region for i in self.cluster])
                region_name = '_' + '+'.join(region)
            else:
                select = np.array([i == region for i in self.cluster])
                region_name = '_' + region
            if not any(select):
                raise ValueError('Region must exist.')
            metric_values = [x[select] for x in self.metric_dict[metric]]
            metric_values = np.concatenate(metric_values)
            methods_name = np.repeat(self.methods, np.sum(select))

        df = pd.DataFrame({metric:metric_values, 'Method':methods_name})
        ax = sns.boxplot(data=df, y=metric, x='Method', showfliers=False)
        for patch in ax.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .7))
        ax = sns.stripplot(data=df, y=metric, x='Method', ax=ax, jitter=0.2, palette='Dark2', size=2)
        ax.set(xlabel='')
        if save:
            plt.savefig(f'{self.out_dir}figures/{metric}{region_name}.jpg', dpi=500, bbox_inches = 'tight')
        plt.show()

    def plot_metric_type(self, save=False, metric="Correlation"):
        """
        Same to plot_metric_spot, but each value in box plot represents a cell type. Box number equals to the number of methods.
        """
        assert metric not in self.spot_metric_names and metric in self.function_map.keys()
        if metric not in self.metric_dict.keys():
            self.evaluate_metric(metric=metric)
        sns.set_palette("Dark2")
        metric_values = np.concatenate(self.metric_dict[metric])
        methods_name = np.repeat(self.methods, self.n_type)
        df = pd.DataFrame({metric:metric_values, 'Method':methods_name})
        ax = sns.boxplot(data=df, y=metric, x='Method', showfliers=False)
        for patch in ax.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .7))
        ax = sns.stripplot(data=df, y=metric, x='Method', ax=ax, jitter=True, color='black', size=2)
        ax.set(xlabel='')
        if save:
            plt.savefig(f'{self.out_dir}figures/{metric}.jpg', dpi=500, bbox_inches = 'tight')
        plt.show()

    def plot_metric_spot_type(self, save=False, metric='Absolute error'):
        """
        Similar to plot_metric_spot, but the figures are separated for each cell type.
        """
        """
        Plot the box plot of each method based on the metric. Each value in box plot represents a spot.
        Box number equals to the number of methods.
        """
        assert metric in self.general_metric_names
        plt.figure(dpi=300)
        if metric not in self.metric_dict.keys():
            self.evaluate_metric(metric=metric)
        sns.set_palette("Dark2")
        n_spot = len(self.proportion_truth)
        metric_values = np.vstack(self.metric_dict[metric])
        metric_values = metric_values.flatten('F')
        methods_name = np.repeat(self.methods, n_spot)
        methods_name = np.tile(methods_name, self.n_type)
        cell_type = np.repeat(self.type_list, n_spot*self.n_method)

        df = pd.DataFrame({'metric': metric_values, 'Method': methods_name, 'Cell type': cell_type})

        for cell_type in self.type_list:
            ax = sns.boxplot(data=df[df['Cell type'] == cell_type], y='metric', x='Method', showfliers=False)
            for patch in ax.patches:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, .7))
            ax = sns.stripplot(data=df[df['Cell type'] == cell_type], y='metric', x='Method', ax=ax, jitter=0.2, palette='Dark2', size=2)
            ax.set(xlabel='')
            if save:
                cell_type = "".join(x for x in cell_type if x.isalnum())
                plt.savefig(f'{self.out_dir}figures/{metric} {cell_type}.jpg', dpi=500, bbox_inches='tight')
            plt.show()
            del ax

    def plot_metric_all(self, save=False, metric="Absolute error", region=None):
        assert metric in self.general_metric_names
        plt.figure(figsize=(self.n_method*self.n_type/4, 5), dpi=300)
        if metric not in self.metric_dict.keys():
            self.evaluate_metric(metric=metric)
        sns.set_palette("Dark2")
        region_name = ''
        if region is None:
            metric_values = np.vstack(self.metric_dict[metric])
            metric_values = metric_values.flatten('F')
            n_spot = len(self.proportion_truth)
        else:
            if isinstance(region, list):
                select = np.array([i in region for i in self.cluster])
                region_name = '_' + '+'.join(region)
            else:
                select = np.array([i == region for i in self.cluster])
                region_name = '_' + region
            if not any(select):
                raise ValueError('Region must exist.')
            metric_values = [x[select] for x in self.metric_dict[metric]]
            metric_values = np.vstack(metric_values)
            metric_values = metric_values.flatten('F')
            n_spot = np.sum(select)

        methods_name = np.repeat(self.methods, n_spot)
        methods_name = np.tile(methods_name, self.n_type)
        cell_type = np.repeat(self.type_list, n_spot*self.n_method)

        df = pd.DataFrame({metric:metric_values, 'Method':methods_name, 'Cell type':cell_type})
        ax = sns.boxplot(data=df, y=metric, hue='Method', x='Cell type', flierprops={"marker": "o"}, dodge=True,
                         linewidth=0.6, fliersize=0.5)
        # sns.violinplot(data=df, y=metric, hue='Method', x='Cell type')
        # ax = sns.catplot(data=df, y=metric, hue='Method', x='Cell type', kind='boxen')
        # for patch in ax.patches:
        #     r, g, b, a = patch.get_facecolor()
        #     patch.set_facecolor((r, g, b, .7))
        # ax = sns.stripplot(data=df, y=metric, x='Method', hue='Cell type', ax=ax, jitter=0.2, palette='Dark2', size=2)
        # ax.set(xlabel='')
        if save:
            plt.savefig(f'{self.out_dir}figures/{metric}{region_name}.tiff', dpi=800, bbox_inches='tight')
        plt.show()


def decomposition(adata_st: anndata.AnnData, adata_sc: anndata.AnnData, key_type: str, cell_proportion: np.ndarray,
                  save=True, out_dir='', threshold=0.1, n_cell=None, spot_location: np.ndarray = None,
                  filtering_gene=False, filename="ST_decomposition.h5ad"):
    """
    Decompose ST.

    Args:
        adata_st: Original spatial transcriptomics data.
        adata_sc: Original single-cell data.
        key_type: The key that is used to extract cell type information from adata_sc.obs.
        cell_proportion: Proportion of each cell type obtained by the deconvolution.
        save: If True, save the generated adata_st as a file.
        out_dir: Output directory.
        threshold: If n_cell is none, discard cell types with proportion less than threshold.
        n_cell: Number of cells in each spot.
        spot_location: Coordinates of the spots.
        filtering_gene: Whether filter the genes in sc_reference.initialization.
        filename: Name of the saved file.
    Returns:
        adata_st_decomposed: Anndata similar to scRNA, but obtained by decomposing ST.
    """
    type_list = sorted(list(adata_sc.obs[key_type].unique()))  # list of the cell type.
    assert len(type_list) == cell_proportion.shape[1]
    assert len(cell_proportion) == len(adata_st)
    if spot_location is not None:
        assert len(spot_location) == len(adata_st)
    n_type = len(type_list)

    cell_proportion_temp = cell_proportion.copy()
    if n_cell is None:
        select = cell_proportion_temp >= threshold
        cell_proportion_temp[cell_proportion_temp < threshold] = 0
        cell_proportion_temp = cell_proportion_temp/(np.sum(cell_proportion_temp, axis=1, keepdims=True)+1e-6)
    else:
        cell_count = proportion_to_count(cell_proportion_temp, n_cell)
        select = cell_count > 0.9
        cell_proportion_temp = cell_count/(np.sum(cell_count, axis=1, keepdims=True)+1e-6)

    adata_sc_temp, adata_st_temp = sc_reference.initialization(adata_sc, adata_st, filtering=filtering_gene)
    sc_ref = sc_reference.construct_sc_ref(adata_sc_temp, key_type=key_type)
    if type(adata_st_temp.X) is not np.ndarray:
        X = adata_st_temp.X.toarray().copy()
    else:
        X = adata_st.X.copy()
    Y = cell_proportion_temp[:, np.newaxis, :] * sc_ref.T
    Y = Y/(np.sum(Y, axis=2, keepdims=True)+1e-10)
    Y = Y * X[:, :, np.newaxis]  # n_spot*n_gene*n_type

    cell_type = []
    ST_decompose = []
    for i in range(n_type):
        ST_decompose += [Y[select[:, i], :, i]]
        cell_type += [type_list[i]]*np.sum(select[:, i])
    ST_decompose = np.vstack(ST_decompose)
    ST_decompose = ST_decompose/(np.sum(ST_decompose, axis=1, keepdims=True)+1e-10)*1e6
    adata_st_decomposed = anndata.AnnData(ST_decompose)
    adata_st_decomposed.var_names = adata_st_temp.var_names
    adata_st_decomposed.obs['cell_type'] = cell_type

    if spot_location is not None:
        location = [spot_location[select[:, i], :] for i in range(n_type)]
        location = np.vstack(location)
        adata_st_decomposed.obs['location_x'] = location[:, 0]
        adata_st_decomposed.obs['location_y'] = location[:, 1]

    if save:
        if out_dir and not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_dir = out_dir + '/' if out_dir else ''
        adata_st_decomposed.write(out_dir + filename)

    return adata_st_decomposed
