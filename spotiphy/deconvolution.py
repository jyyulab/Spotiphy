import anndata
import os
# import matplotlib as mpl
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from . import sc_reference


def deconvolute(X, sc_ref, device='cuda', n_epoch=8000, adam_params=None, batch_prior=2,
                plot=False, fig_size=(4.8, 3.6), dpi=200):
    """
    Deconvolution of the proportion of genes contributed by each cell type.

    Args:
        X: Spatial transcriptomics data. n_spot*n_gene.
        sc_ref: Single cell reference. n_type*n_gene.
        device: The device used for the deconvolution.
        plot: Whether to plot the ELBO loss.
        n_epoch: Number of training epochs.
        adam_params: Parameters for the adam optimizer.
        batch_prior: Parameter of the prior Dirichlet distribution of the batch effect: 2^(Uniform(0, batch_prior))
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
    batch_prior = torch.tensor(batch_prior, device=device, dtype=torch.float64)
    assert X.shape[1] == sc_ref.shape[1], "Spatial data and SC reference data must have the same number of genes."

    def model(sc_ref, X, batch_prior):
        n_spot = len(X)
        n_type, n_gene = sc_ref.shape
        # alpha = pyro.sample("Batch effect", dist.Dirichlet(torch.full((n_gene,), batch_prior, device=device,
        #                                                               dtype=torch.float64)))
        alpha = pyro.sample("Batch effect", dist.Uniform(torch.tensor(0., device=device),
                                                         torch.tensor(1., device=device)).expand([n_gene]).to_event(1))
        alpha_exp = torch.exp2(alpha * batch_prior)
        with pyro.plate("spot", n_spot, dim=-1):
            q = pyro.sample("Proportion", dist.Dirichlet(torch.full((n_type,), 3., device=device, dtype=torch.float64)))
            rho = torch.matmul(q, sc_ref)
            rho = rho * alpha_exp
            # print(torch.min(rho.sum(dim=-1).unsqueeze(-1)))
            rho = rho / rho.sum(dim=-1).unsqueeze(-1)
            if torch.any(torch.isnan(rho)):
                print('rho contains NaN at')
                print(torch.where(torch.isnan(rho)))
            pyro.sample("Spatial RNA", dist.Multinomial(total_count=max_exp, probs=rho), obs=X)

    def guide(sc_ref, X, batch_prior):
        n_spot = len(X)
        n_type, n_gene = sc_ref.shape
        # pi = pyro.param('pi', lambda: torch.full((n_gene,), 10, device=device, dtype=torch.float64),
        #                 constraint=constraints.positive)
        # alpha = pyro.sample("Batch effect", dist.Dirichlet(pi))
        alpha_loc = pyro.param("alpha_loc", torch.zeros(n_gene, device=device, dtype=torch.float64))
        alpha_scale = pyro.param("alpha_scale", torch.ones(n_gene, device=device, dtype=torch.float64),
                                 constraint=constraints.positive)
        base_dist = dist.Normal(alpha_loc, alpha_scale)
        sigmoid_trans = dist.transforms.SigmoidTransform()  # Transforms to [0, 1]

        alpha_dist = pyro.distributions.TransformedDistribution(base_dist, [sigmoid_trans])

        alpha = pyro.sample("Batch effect", alpha_dist.to_event(1))
        with pyro.plate("spot", n_spot):
            sigma = pyro.param('sigma', lambda: torch.full((n_spot, n_type), 3., device=device, dtype=torch.float64),
                               constraint=constraints.positive)
            if torch.any(torch.isnan(sigma)):
                print('sigma contains NaN at')
                print(torch.where(torch.isnan(sigma)))
            # print(torch.min(sigma))
            q = pyro.sample("Proportion", dist.Dirichlet(sigma))

    pyro.clear_param_store()
    optimizer = Adam(adam_params)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    loss_history = []
    for _ in tqdm(range(n_epoch)):
        loss = svi.step(sc_ref, X, batch_prior)
        loss_history.append(loss)
    if plot:
        plt.figure(figsize=fig_size, dpi=dpi)
        plt.plot(loss_history, lw=0.5)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()
    return pyro.get_param_store()


def estimation_proportion(X, adata_sc, sc_ref, type_list, key_type, device='cuda', n_epoch=8000, adam_params=None,
                          batch_prior=2, plot=False, fig_size=(4.8, 3.6), dpi=200):
    """
    Estimate the proportion of each cell type in each spot.

    Args:
        X: Spatial transcriptomics data. n_spot*n_gene.
        adata_sc: scRNA data (Anndata).
        sc_ref: Single cell reference. n_type*n_gene.
        type_list: List of the cell types.
        key_type: Column name of the cell types in adata_sc.
        device: The device used for the deconvolution.
        plot: Whether to plot the ELBO loss.
        n_epoch: Number of training epochs.
        adam_params: Parameters for the adam optimizer.
        batch_prior: Parameter of the prior Dirichlet distribution of the batch effect: 2^(Uniform(0, batch_prior))
        fig_size: Size of the figure.
        dpi: Dots per inch (DPI) of the figure.

    Returns:
        Parameters in the generative model.
    """
    pyro_params = deconvolute(X, sc_ref, device=device, n_epoch=n_epoch, adam_params=adam_params,
                              batch_prior=batch_prior, plot=plot, fig_size=fig_size, dpi=dpi)
    sigma = pyro_params['sigma'].cpu().detach().numpy()
    Y = np.array(adata_sc.X)
    mean_exp = np.array([np.mean(np.sum(Y[adata_sc.obs[key_type]==type_list[i]], axis=1))
                         for i in range(len(type_list))])
    cell_proportion = sigma/mean_exp
    cell_proportion = cell_proportion/np.sum(cell_proportion, axis=1)[:, np.newaxis]
    return cell_proportion


def plot_proportion(img, proportion, spot_location, radius, cmap_name='viridis', alpha=0.4, save_path='proportion.png',
                    vmax=0.98, spot_scale=1.3, show_figure=False):
    """
    Plot the proportion of a cell type.

    Args:
        img: 3 channel img with integer values in [0, 255]
        proportion: Proportion of a cell type.
        spot_location: Location of the spots.
        radius: Radius of the spot
        cmap_name: Name of the camp.
        alpha: Level of transparency of the background img.
        save_path: If not none, save the img to the path.
        vmax: Quantile of the maximum value in the color bar.
        spot_scale: Scale of the spot in the figure.
    """
    def render_to_array(fig):
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

    img = img*alpha + np.ones(img.shape)*255*(1-alpha)
    img = img.astype(np.int32)
    vmin = 0
    vmax = np.quantile(proportion, vmax)
    if vmax - vmin < 0.03:
        vmax = vmin + 0.03
    proportion = (proportion-vmin)/vmax
    cmap = plt.cm.get_cmap(cmap_name)
    for i, p in enumerate(proportion):
        rgb_float = cmap(p)[:3]
        rgb_int = tuple(int(255 * x) for x in rgb_float)
        cv.circle(img, spot_location[i], int(radius*spot_scale), rgb_int, -1)

    fig = plt.figure(figsize=(1, 2.5), dpi=1200)
    cbar_ax = fig.add_axes([0, 0.1, 0.25, 0.85])
    if vmax >= 0.2:
        a = int(vmax*10)/20
    else:
        a = int(vmax*100)/200
    ticks=[0, a, a*2]
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap), cax=cbar_ax, ticks=ticks)
    cb1.outline.set_edgecolor("none")
    for label in cb1.ax.get_yticklabels():
        label.set_size(13)
    cbar_array = render_to_array(fig)
    plt.close(fig)

    c = [5800, 8100]
    img[c[0]:c[0]+cbar_array.shape[0], c[1]:c[1]+cbar_array.shape[1]][np.sum(cbar_array, axis=2) < 252*3] \
        = cbar_array[np.sum(cbar_array, axis=2) < 252*3]

    if save_path is not None:
        print('Saving the image.')
        img1 = img[:, :, [2, 1, 0]]
        cv.imwrite(save_path, img1)
    plt.imshow(img)
    if not show_figure:
        plt.close()


def proportion_to_count(p, n, multiple_spots=False):
    """
    Convert the cell proportion to the absolute cell number.

    Args:
        p: Cell proportion.
        n: Number of cells.
        multiple_spots: If the data is related to multiple spots

    Returns:
        Cell count of each cell type.
    """
    if multiple_spots:
        assert len(p) == len(n)
        count = np.zeros(p.shape)
        for i in range(len(p)):
            count[i] = proportion_to_count(p[i], n[i])
    else:
        assert (np.abs(np.sum(p) - 1) < 1e-5)
        c0 = p * n
        count = np.floor(c0)
        r = c0 - count
        if np.sum(count) == n:
            return count
        idx = np.argsort(r)[-int(np.round(n - np.sum(count))):]
        count[idx] += 1
    return count


def estimation_N(proportion, mean_exp_type, adata_st):
    sum_exp_spot = np.sum(np.array(adata_st.X))



def simulation(adata_st: anndata.AnnData, adata_sc: anndata.AnnData, key_type: str, cell_proportion: np.ndarray,
               n_cell=10, batch_effect_sigma=0.1, zero_proportion=0.3, additive_noise_sigma=0.05, save=True,
               out_dir='', filename="ST_Simulated.h5ad", verbose=0):
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
        filename: Name of the saved file.
        verbose: Whether print the time spend.
    Returns:
        Simulated ST Anndata.
    """
    time_start = time.time()
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
    if verbose:
        print('Prepared the ground truth. Time use {:.2f}'.format(time.time() - time_start))
        time_start = time.time()

    # Construct expression matrix
    common_genes = list(set(adata_st.var_names).intersection(set(adata_sc.var_names)))
    adata_sc = adata_sc[:, common_genes]
    adata_st = adata_st[:, common_genes]
    n_gene = len(common_genes)
    X = np.zeros(adata_st.shape)
    Y = adata_sc.X if type(adata_sc.X) is np.ndarray else adata_sc.X.toarray()
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
    if verbose:
        print('Constructed the ground truth. Time use {:.2f}'.format(time.time() - time_start))
        time_start = time.time()

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
    if verbose:
        print('Added batch effect, zero reads, and additive noise. Time use {:.2f}'.format(time.time() - time_start))
        time_start = time.time()

    if save:
        if out_dir and not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_dir = out_dir + '/' if out_dir else ''
        adata_st.raw = None
        adata_st.write(out_dir + filename)
        if verbose:
            print('Saved the simulated data to file. Time use {:.2f}'.format(time.time() - time_start))

    return adata_st


class Evaluation:
    def __init__(self, proportion_truth, proportion_estimated_list, methods, out_dir="", cluster=None, type_list=None,
                 colors=None, coordinates=None, min_spot_distance=112):
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
        self.metric_dict = dict()  # Saved metric values.
        self.n_method = len(methods)
        self.coordinates = coordinates
        self.min_spot_distance = min_spot_distance
        self.cluster = cluster
        self.type_list = type_list
        self.n_type = proportion_truth.shape[1]

        self.function_map = {'Absolute error': self.absolute_error, 'Square error': self.square_error,
                             'Cosine similarity': self.cosine, 'Correlation': self.correlation,
                             'JSD': self.JSD, 'Fraction of cells correctly mapped': self.correct_fraction}
        self.metric_type_dict = {'Spot': {'Cosine similarity', 'Absolute error', 'Square error', 'JSD', 'Correlation',
                                          'Fraction of cells correctly mapped'},
                                 'Cell type': {'Cosine similarity', 'Absolute error', 'Square error', 'JSD',
                                               'Fraction of cells correctly mapped', 'Correlation'},
                                 'Individual': {'Absolute error', 'Square error'}}
        if colors is None:
            self.colors = ["#3c93c2", "#089099", "#7ccba2", "#fcde9c", "#f0746e", "#dc3977", "#7c1d6f"]
        else:
            self.colors = colors
        self.colors = self.colors[:len(self.methods)]

        if out_dir:
            self.out_dir = out_dir if out_dir[-1] == '/' else out_dir + '/'
        else:
            self.out_dir = ''
        if self.out_dir and not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        if not os.path.exists(out_dir + 'figures'):
            os.mkdir(out_dir + 'figures')
        assert len(proportion_estimated_list) == self.n_method

    @staticmethod
    def absolute_error(proportion_truth: np.ndarray, proportion_estimated: np.ndarray, metric_type='Spot'):
        assert proportion_truth.shape == proportion_estimated.shape
        error = np.abs(proportion_truth - proportion_estimated)
        if metric_type == 'Individual':
            return error
        elif metric_type == 'Spot':
            return np.sum(error, axis=1)
        elif metric_type == 'Cell type':
            return np.sum(error, axis=0)
        else:
            raise ValueError(f"Invalid metric type {metric_type}")

    @staticmethod
    def square_error(proportion_truth: np.ndarray, proportion_estimated: np.ndarray, metric_type='Spot'):
        assert proportion_truth.shape == proportion_estimated.shape
        error = (proportion_truth - proportion_estimated) ** 2
        if metric_type == 'Individual':
            return error
        elif metric_type == 'Spot':
            return np.sum(error, axis=1)
        elif metric_type == 'Cell type':
            return np.sum(error, axis=0)
        else:
            raise ValueError(f"Invalid metric type {metric_type}")

    @staticmethod
    def cosine(proportion_truth: np.ndarray, proportion_estimated: np.ndarray, metric_type='Spot'):
        assert proportion_truth.shape == proportion_estimated.shape
        if metric_type == 'Spot':
            cosine_similarity = np.sum(proportion_truth * proportion_estimated, axis=1) / \
                                np.linalg.norm(proportion_estimated, axis=1) / np.linalg.norm(proportion_truth, axis=1)
            return cosine_similarity
        elif metric_type == 'Cell type':
            cosine_similarity = np.sum(proportion_truth * proportion_estimated, axis=0) / \
                                np.linalg.norm(proportion_estimated, axis=0) / np.linalg.norm(proportion_truth, axis=0)
            return cosine_similarity
        else:
            raise ValueError(f"Invalid metric type {metric_type}")

    @staticmethod
    def correlation(proportion_truth: np.ndarray, proportion_estimated: np.ndarray, metric_type='Spot'):
        assert proportion_truth.shape == proportion_estimated.shape
        if metric_type == 'Cell type':
            proportion_truth_centered = proportion_truth - np.mean(proportion_truth, axis=0)
            proportion_estimated_centered = proportion_estimated - np.mean(proportion_estimated, axis=0)
            correlation_values = np.sum(proportion_truth_centered * proportion_estimated_centered, axis=0) / \
                                 np.sqrt(np.sum(proportion_truth_centered ** 2, axis=0) *
                                         np.sum(proportion_estimated_centered ** 2, axis=0))
            return correlation_values
        elif metric_type == 'Spot':
            proportion_truth_centered = proportion_truth - np.mean(proportion_truth, axis=1, keepdims=True)
            proportion_estimated_centered = proportion_estimated - np.mean(proportion_estimated, axis=1, keepdims=True)
            correlation_values = np.sum(proportion_truth_centered * proportion_estimated_centered, axis=1) / \
                                 np.sqrt(np.sum(proportion_truth_centered ** 2, axis=1) *
                                         np.sum(proportion_estimated_centered ** 2, axis=1))
            return correlation_values
        else:
            raise ValueError(f"Invalid metric type {metric_type}")

    @staticmethod
    def correct_fraction(proportion_truth: np.ndarray, proportion_estimated: np.ndarray, metric_type='Spot'):
        assert proportion_truth.shape == proportion_estimated.shape
        correct_proportion = np.minimum(proportion_truth, proportion_estimated)
        if metric_type == 'Cell type':
            return np.sum(correct_proportion, axis=0) / np.sum(proportion_truth, axis=0)
        elif metric_type == 'Spot':
            return np.sum(correct_proportion, axis=1)
        else:
            raise ValueError(f"Invalid metric type {metric_type}")

    @staticmethod
    def JSD(proportion_truth: np.ndarray, proportion_estimated: np.ndarray, metric_type='Spot'):
        """
        Jensen–Shannon divergence
        Args:
            proportion_truth: Ground truth of the cell proportion.
            proportion_estimated: Estimated proportion.
            metric_type: How the metric is calculated.
        """
        assert proportion_truth.shape == proportion_estimated.shape
        if metric_type == 'Spot':
            return jensenshannon(proportion_truth, proportion_estimated, axis=1)
        elif metric_type == 'Cell type':
            return jensenshannon(proportion_truth, proportion_estimated, axis=0)
        else:
            raise ValueError(f"Invalid metric type {metric_type}")

    def evaluate_metric(self, metric='Cosine similarity', metric_type='Spot', region=None):
        """
        Evaluate the proportions based on the metric.
        Args:
            metric: Name of the metric.
            metric_type: How the metric is calculated. 'Spot': metric is calculated for each spot; 'Cell type',
                         metric is calculated for each cell type; 'Individual': metric is calculated for each individual
                         proportion estimation.
            region: The region that is being evaluated.
        """
        assert metric in self.metric_type_dict[metric_type]
        metric_values = []
        func = self.function_map.get(metric)
        if region is None:
            select = np.array([True]*len(self.proportion_truth))
        elif isinstance(region, list):
            select = np.array([i in region for i in self.cluster])
        else:
            select = np.array([i == region for i in self.cluster])

        if not any(select):
            raise ValueError(f'Region {region} do/does not exist.')
        elif metric_type == 'Cell type' and np.sum(select) == 1:
            raise ValueError(f'Region {region} only contain(s) one spot.')
        for i in range(self.n_method):
            metric_values.append(func(self.proportion_truth[select], self.proportion_estimated_list[i][select],
                                      metric_type))
        self.metric_dict[metric+' '+metric_type] = metric_values
        return metric_values

    def plot_metric(self, save=False, region=None, metric='Cosine similarity', metric_type='Spot', cell_types=None,
                    suffix='', show=True):
        """
        Plot the box plot of each method based on the metric.
        Box number equals to the number of methods.
        Args:
            save: If true, save the figure.
            region: Regions of the tissue.
            metric: Name of the metric.
            metric_type: How the metric is calculated. 'Spot': metric is calculated for each spot; 'Cell type',
                         metric is calculated for each cell type; 'Individual': metric is calculated for each individual
                         proportion estimation.
            cell_types: If metric_type is 'Cell type' and cell_types is not None, then only plot the results
                        corresponding to the cell_types.
            suffix: suffix of the save file.
            show: Whether to show the figure
        """
        assert metric_type == 'Spot' or metric_type == 'Cell type'
        assert metric in self.metric_type_dict[metric_type]
        if metric+' '+metric_type not in self.metric_dict.keys():
            self.evaluate_metric(metric=metric, metric_type=metric_type, region=region)
        region_name = ''
        if region is not None:
            region_name = '_' + '+'.join(region) if isinstance(region, list) else '_' + region

        if metric_type == 'Cell type' and cell_types is not None:
            idx = np.array([np.where(self.type_list == type_)[0][0] for type_ in cell_types])
            metric_values = self.metric_dict[metric+' '+metric_type]
            metric_values = [m[idx] for m in metric_values]
            metric_values = np.concatenate(metric_values)
            methods_name = np.repeat(self.methods, len(idx))
        else:
            metric_values = np.concatenate(self.metric_dict[metric+' '+metric_type])
            methods_name = np.repeat(self.methods, len(self.metric_dict[metric+' '+metric_type][0]))

        df = pd.DataFrame({metric: metric_values, 'Method': methods_name})
        if show:
            plt.figure(dpi=300)
            palette = self.colors if len(self.methods) <= len(self.colors) else 'Dark2'
            ax = sns.boxplot(data=df, y=metric, x='Method', showfliers=False, palette=palette)
            if metric_type == 'Spot':
                ax = sns.stripplot(data=df, y=metric, x='Method', ax=ax, jitter=0.2, palette=palette, size=1)
            else:
                ax = sns.stripplot(data=df, y=metric, x='Method', ax=ax, jitter=True, color='black', size=1.5)
            for patch in ax.patches:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, .8))
            ax.set(xlabel='')
            sns.despine(top=True, right=True)
            plt.gca().yaxis.grid(False)
            plt.gca().xaxis.grid(False)
            plt.gca().spines['left'].set_color('black')
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().tick_params(left=True, axis='y', colors='black')
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
            if save:
                plt.savefig(f'{self.out_dir}figures/{metric}_{metric_type}{region_name}{suffix}.jpg', dpi=500,
                            bbox_inches='tight')
            plt.show()
        return df

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
        cell_type = np.repeat(self.type_list, n_spot * self.n_method)

        # print(np.shape(metric_values), np.shape(methods_name), np.shape(cell_type))
        df = pd.DataFrame({'metric': metric_values, 'Method': methods_name, 'Cell type': cell_type})

        for cell_type in self.type_list:
            ax = sns.boxplot(data=df[df['Cell type'] == cell_type], y='metric', x='Method', showfliers=False)
            for patch in ax.patches:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, .7))
            ax = sns.stripplot(data=df[df['Cell type'] == cell_type], y='metric', x='Method', ax=ax, jitter=0.2,
                               palette='Dark2', size=2)
            ax.set(xlabel='')
            if save:
                cell_type = "".join(x for x in cell_type if x.isalnum())
                plt.savefig(f'{self.out_dir}figures/{metric} {cell_type}.jpg', dpi=500, bbox_inches='tight')
            plt.show()
            del ax

    def plot_metric_all(self, save=False, metric="Absolute error", region=None):
        assert metric in self.general_metric_names
        plt.figure(figsize=(self.n_method * self.n_type / 4, 5), dpi=300)
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
        cell_type = np.repeat(self.type_list, n_spot * self.n_method)

        df = pd.DataFrame({metric: metric_values, 'Method': methods_name, 'Cell type': cell_type})
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
                  filtering_gene=False, filename="ST_decomposition.h5ad", verbose=0, use_original_proportion=False):
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
        verbose: Whether print the time spend.
        use_original_proportion: If the original proportion is used to estimate the iscRNA. Note that even when the
                                 original proportion is used, we still filter some cells in iscRNA.
    Returns:
        adata_st_decomposed: Anndata similar to scRNA, but obtained by decomposing ST.
    """
    time_start = time.time()
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
        cell_proportion_temp = cell_proportion_temp / (np.sum(cell_proportion_temp, axis=1, keepdims=True) + 1e-6)
    else:
        cell_count = np.zeros((len(spot_location), n_type))
        for i in range(len(spot_location)):
            cell_count[i] = proportion_to_count(cell_proportion_temp[i], n_cell[i])
        cell_count = cell_count.astype(np.int32)
        select = []
        for i in range(n_type):
            select_temp = []
            for j in range(len(cell_count)):
                if cell_count[j, i] > 0:
                    select_temp += [j]*cell_count[j, i]
            select += [np.array(select_temp)]
        cell_proportion_temp = cell_count / (np.sum(cell_count, axis=1, keepdims=True) + 1e-6)
    if verbose:
        print('Prepared proportion data. Time use {:.2f}'.format(time.time() - time_start))
        time_start = time.time()

    adata_sc_temp, adata_st_temp = sc_reference.initialization(adata_sc, adata_st, filtering=filtering_gene)
    if verbose:
        print('Initialized scRNA and ST data. Time use {:.2f}'.format(time.time() - time_start))
        time_start = time.time()
    sc_ref = sc_reference.construct_sc_ref(adata_sc_temp, key_type=key_type)
    X = adata_st_temp.X if type(adata_st_temp.X) is np.ndarray else adata_st_temp.X.toarray()
    # X = np.array(adata_st_temp.X)
    if verbose:
        print('Processed scRNA and ST data. Time use {:.2f}'.format(time.time() - time_start))
        time_start = time.time()
    if use_original_proportion:
        Y = cell_proportion[:, np.newaxis, :] * sc_ref.T
    else:
        Y = cell_proportion_temp[:, np.newaxis, :] * sc_ref.T
    Y = Y / (np.sum(Y, axis=2, keepdims=True) + 1e-10)
    Y = Y * X[:, :, np.newaxis]  # n_spot*n_gene*n_type
    if verbose:
        print('Decomposition complete. Time use {:.2f}'.format(time.time() - time_start))
        time_start = time.time()

    cell_type = []
    ST_decompose = []
    spot_name = []
    n_cell_duplicate = []
    for i in range(n_type):
        if n_cell is not None:
            if len(select[i])>0:
                n_cell_duplicate += list(cell_count[select[i], i])
                ST_decompose += [Y[select[i], :, i]]
                cell_type += [type_list[i]] * np.sum(cell_count[:, i])
                spot_name += list(np.array(adata_st.obs_names)[select[i]])
        else:
            ST_decompose += [Y[select[:, i], :, i]]
            cell_type += [type_list[i]] * np.sum(select[:, i])
            spot_name += list(np.array(adata_st.obs_names)[select[:, i]])
    ST_decompose = np.vstack(ST_decompose)
    ST_decompose = ST_decompose / (np.sum(ST_decompose, axis=1, keepdims=True) + 1e-10) * 1e6
    adata_st_decomposed = anndata.AnnData(ST_decompose)
    adata_st_decomposed.uns = adata_st.uns
    adata_st_decomposed.var_names = adata_st_temp.var_names
    adata_st_decomposed.obs['cell_type'] = cell_type
    adata_st_decomposed.obs['spot_name'] = spot_name
    if n_cell is not None:
        adata_st_decomposed.obs['n_cell'] = n_cell_duplicate
        adata_st_decomposed.uns['cell_count'] = cell_count
    if spot_location is not None:
        if n_cell is not None:
            select = np.concatenate(select).astype(np.int32)
            location = spot_location[select]
        else:
            location = [spot_location[select[:, i], :] for i in range(n_type)]
            location = np.vstack(location)
        adata_st_decomposed.obs['location_x'] = location[:, 0]
        adata_st_decomposed.obs['location_y'] = location[:, 1]
    if verbose:
        print('Constructed ST decomposition data file. Time use {:.2f}'.format(time.time() - time_start))
        time_start = time.time()

    if save:
        if out_dir and not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_dir = out_dir + '/' if out_dir else ''
        adata_st_decomposed.write_h5ad(out_dir + filename, compression='gzip')
        if verbose:
            print('Saved file to output folder. Time use {:.2f}'.format(time.time() - time_start))

    return adata_st_decomposed


def assign_type_spot(nucleus_df, n_cell_df, cell_number, type_list):
    """
    Assign the cell type to the cells inside the spot.

    Args:
        nucleus_df: Dataframe of the nucleus. Part of spotiphy.segmentation.Segmentation.
        n_cell_df: Dataframe of the number of cells in each spot. Part of spotiphy.segmentation.Segmentation.
        cell_number: Number of each cell type in each spot.
        type_list: List of the cell types.
    Returns:
        nucleus_df with assigned spot
    """
    assert len(type_list) == cell_number.shape[1]
    assert len(n_cell_df) == len(cell_number)
    cell_number = cell_number.astype(np.int32)
    if 'cell_type' not in nucleus_df.columns:
        nucleus_df['cell_type'] = 'unknown'
    for i in range(len(n_cell_df)):
        if n_cell_df.loc[i, 'cell_count'] > 0:
            index = n_cell_df.loc[i, 'Nucleus indices']
            np.random.shuffle(index)
            j1 = 0
            for j2, n in enumerate(cell_number[i]):
                if n > 0:
                    nucleus_df.loc[index[j1:j1+n], 'cell_type'] = type_list[j2]
                    j1 += n
    return nucleus_df


def assign_type_out(nucleus_df, cell_proportion, spot_centers, type_list, max_distance=100, band_width=100):
    """
    Assign the cell type to the cells outside the spot.

    Args:
        nucleus_df: Dataframe of the nucleus. Part of spotiphy.segmentation.Segmentation.
        spot_centers: Centers of the spots.
        cell_proportion: Proportion of each cell type in each spot.
        type_list: List of the cell types.
        max_distance: If the distance between a nucleus and the closest spot is larger than max_distance, the cell type
                      will not be assigned to this nucleus.
        band_width: Band width of the kernel.
    Returns:
        nucleus_df with assigned spot
    """
    assert len(type_list) == cell_proportion.shape[1]
    assert len(cell_proportion) == len(spot_centers)
    if 'cell_type' not in nucleus_df.columns:
        nucleus_df['cell_type'] = 'unknown'
    nucleus_centers = nucleus_df[['x', 'y']].values
    d = np.sum((nucleus_centers[:, :, np.newaxis] - spot_centers.T)**2, axis=1)
    d_min = np.min(d, axis=1)
    weights = np.exp(-d/band_width**2)
    smooth = np.sum(weights[:, :, np.newaxis] * cell_proportion, axis=1) / np.sum(weights, axis=1, keepdims=True)

    r = np.random.random(len(nucleus_centers))
    for i in range(len(nucleus_centers)):
        if d_min[i] > max_distance**2 or nucleus_df.loc[i, 'in_spot']:
            continue
        j = 0
        t_temp = r[i]
        while t_temp - smooth[i, j] > 0:
            t_temp -= smooth[i, j]
            j += 1
        nucleus_df.loc[i, 'cell_type'] = type_list[j]
    return nucleus_df, smooth


def archive_assign_type_out_gp(nucleus_df, cell_proportion, spot_centers, type_list, max_distance=100, return_gp=False):
    """
    Assign the cell type to the cells outside the spot.

    Args:
        nucleus_df: Dataframe of the nucleus. Part of spotiphy.segmentation.Segmentation.
        spot_centers: Centers of the spots.
        cell_proportion: Proportion of each cell type in each spot.
        type_list: List of the cell types.
        max_distance: If the distance between a nucleus and the closest spot is larger than max_distance, the cell type
                      will not be assigned to this nucleus.
        return_gp: If return the fitted GP models.
    Returns:
        nucleus_df with assigned spot
    """
    assert len(type_list) == cell_proportion.shape[1]
    assert len(cell_proportion) == len(spot_centers)
    if 'cell_type' not in nucleus_df.columns:
        nucleus_df['cell_type'] = 'unknown'
    nucleus_centers = nucleus_df[['x', 'y']].values
    d = np.sum((nucleus_centers[:, :, np.newaxis] - spot_centers.T)**2, axis=1)
    d = np.min(d, axis=1)

    gp_list = []
    interpolation = np.zeros((len(nucleus_centers), len(type_list)))
    mean, std = np.mean(cell_proportion, axis=0), np.std(cell_proportion, axis=0)
    cell_proportion_trans = (cell_proportion-mean)/std
    print('Fitting the GP models.')
    for i in tqdm(range(len(type_list))):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(100.0, (1e-1, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
        gp.fit(spot_centers, cell_proportion_trans[:, i])
        gp_list += [gp]
        interpolation[:, i] = gp_list[i].predict(nucleus_centers, return_std=False)*std[i] + mean[i]
    interpolation[interpolation < 0] = 0
    interpolation += 1e-8
    interpolation = interpolation/np.sum(interpolation, axis=1, keepdims=True)

    print('Assigning the cell types.')
    r = np.random.random(len(nucleus_centers))
    for i in tqdm(range(len(nucleus_centers))):
        if d[i] > max_distance**2 or nucleus_df.loc[i, 'in_spot']:
            continue
        j = 0
        t_temp = r[i]
        while t_temp - interpolation[i, j] > 0:
            t_temp -= interpolation[i, j]
            j += 1
        nucleus_df.loc[i, 'cell_type'] = type_list[j]
    if return_gp:
        return nucleus_df, gp_list
    else:
        return nucleus_df

