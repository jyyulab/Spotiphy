import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import tensorflow as tf
from packaging import version
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Segmentation:
    def __init__(self, Img, spot_center: np.ndarray, out_dir: str = '', prob_thresh: float = 0.2,
                 nms_thresh: float = 0.5, spot_radius: float = 36.5):
        """
        Args:
            Img: Array(_*_*3). Three channel stained image. In default, it should be hematoxylin and eosin (H&E) stained image.
            spot_center: Coordinates of the center of the spots.
            out_dir: Output directory.
            Prob_thresh, nms_thresh: Two thresholds used in Stardist. User should adjust these two threshold based on
                                    the segmentation results.
            spot_radius: Radius of the spots. In 10X Visium, it should be 36.5.
        """
        self.Img = Img
        self.spot_center = np.array(spot_center)
        assert self.spot_center.shape[1] == 2, "spot_center must have exact two columns."
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.spot_radius = spot_radius
        if out_dir and not os.path.exists(out_dir):
            os.mkdir(out_dir)
        self.out_dir = out_dir + '/' if out_dir else ''
        # Segmentation results
        self.label, self.nucleus_boundary, self.nucleus_center, self.probability = None, None, None, None
        self.n_cell_df = None
        self.is_segmented = False  # Whether we have conducted segmentation or not.
        if version.parse(tf.__version__) >= version.parse("2.9.0"):
            tf.keras.Model.predict = change_predict_defaults(tf.keras.Model.predict)
            print(f"Suppress the output of tensorflow prediction for tensorflow version {tf.version.VERSION}>=2.9.0.")

    @staticmethod
    def stardist_2D_versatile_he(Img, prob_thresh: float = 0.2, nms_thresh: float = 0.5, n_tiles=(4, 4, 1),
                                 verbose: bool = True):
        """
        Segmentation function provided by Stardist..
        Args:
            Img: Three channel image.
            nms_thresh: Parameter of non-maximum suppression.
            prob_thresh: The probability threshold that determines the retention of a nucleus.
            n_tiles: Out of memory (OOM) errors can occur if the input image is too large.
                     To avoid this problem, the input image is broken up into (overlapping) tiles that are processed
                     independently and re-assembled. (Copied from stardist document).
                     In default, we break the image into 4*4 tiles.
            verbose: Whether to print segmentation progress.
        Returns:
            label: 2D np.ndarray represents the segmented image. Background pixels has value 0 and nucleus pixels has
                    the positive integer value as the index of the nucleus.
            details: Details[0]: np.ndarray(n_nucleus*2*32). Boundaries of each nucleus.
                     Details[1]: np.ndarray(n_nucleus*2). Center of each nucleus.
                     Details[2]: np.ndarray(n_nucleus). Probability that a segmented nucleus is indeed a nucleus.
        """
        axis_norm = (0, 1, 2)  # normalize channels jointly
        img = normalize(Img, 1, 99.8, axis=axis_norm)
        model = StarDist2D.from_pretrained('2D_versatile_he')
        label, details = model.predict_instances(img, nms_thresh=nms_thresh, prob_thresh=prob_thresh, n_tiles=n_tiles,
                                                 show_tile_progress=verbose)
        device = cuda.get_current_device()
        device.reset()
        return label, details

    @staticmethod
    def n_cell_in_spot(nucleus_center: np.ndarray, spot_center: np.ndarray, spot_radius: float) -> pd.DataFrame:
        """
        Find the number of cells in each spot. If the center of a nucleus is inside the spot, we assume that the cell is
        in the spot.
        Args:
            nucleus_center: np.ndarray(n_nucleus*2). Coordinates of the nucleus centers.
            spot_center: np.ndarray(n_spot*2). Coordinates of the spot centers.
            spot_radius: Radius of the spots.
        Returns:
            n_cell_df: Pandas data frame with two columns.
                       Column 'cell_count' represents the number of cells in a spot.
                       Column 'centers' represents the coordinates of the nucleus centers.
        """
        n_spot = len(spot_center)
        n_cell_df = pd.DataFrame({'cell_count': [0] * n_spot, 'centers': None})
        distance = np.sum((spot_center[:, :, np.newaxis] - nucleus_center.T) ** 2, axis=1)
        for i in range(n_spot):
            nucleus_index = np.where(distance[i] < spot_radius ** 2)[0]
            n_cell_df.iloc[i, 0] = len(nucleus_index)
            n_cell_df.at[i, 'centers'] = nucleus_center[nucleus_index] if len(nucleus_index) else np.array([])
        return n_cell_df

    def segment_nucleus(self, save=True):
        """
        Conduct the segmentation using StarDist pretrained model.
        Args:
            save: Whether to save the segmentation results.
        """
        self.label, details = self.stardist_2D_versatile_he(self.Img, nms_thresh=self.nms_thresh,
                                                            prob_thresh=self.prob_thresh)
        nucleus_boundary, nucleus_center, self.probability = details['coord'], details['points'], details['prob']
        nucleus_boundary = np.transpose(nucleus_boundary, [0, 2, 1])
        self.nucleus_boundary = nucleus_boundary[:, :, [1, 0]]
        self.nucleus_center = nucleus_center[:, [1, 0]]
        self.n_cell_df = self.n_cell_in_spot(self.nucleus_center, self.spot_center, self.spot_radius)
        self.is_segmented = True
        if save:
            self.save_results()

    def save_results(self):
        """
        Save segmentation results.
        """
        assert self.is_segmented, "Please conduct segmentation first."
        np.save(f'{self.out_dir}segmentation_label.npy', self.label)
        np.save(f'{self.out_dir}segmentation_boundary.npy', self.nucleus_boundary)
        np.save(f'{self.out_dir}segmentation_center.npy', self.nucleus_center)
        np.save(f'{self.out_dir}segmentation_probability.npy', self.probability)
        self.n_cell_df.to_csv(f'{self.out_dir}n_cell_df.csv')

    def plot(self, fig_size=(10, 4), dpi=300, crop=None, cmap_segmented='hot'):
        """
        Plot the segmentation results.
        It is recommended to adjust the stardist parameters nms_thresh and prob_thresh based on this plot.
        Args:
            fig_size: Size of the figure.
            dpi: Dots per inch (DPI) of the figure.
            crop: If None, show the full image.
                  Otherwise, crop should be
            cmap_segmented: Color map of the segmented image.
        """
        assert self.is_segmented, "Please conduct segmentation first."
        fig, ax = plt.subplots(1, 2, figsize=fig_size, dpi=dpi)
        if crop is None:
            img = self.Img
            img_segmented = self.label
        else:
            img = self.Img[crop[0]:crop[1], crop[2]:crop[3]]
            img_segmented = self.label[crop[0]:crop[1], crop[2]:crop[3]]
        ax[0].imshow(img)
        ax[0].set_title("Original image")
        ax[1].imshow(img_segmented, cmap=cmap_segmented)
        ax[1].set_title("Segmented image")
        plt.show()


def change_predict_defaults(predict_function):
    """
    Start from tensorflow 2.9.0, the default verbose value in function tensorflow.keras.Model.predict is set to 'auto',
    which takes value 1 for the most of the time. Thus, this function is a decorator to wrap the predict function and
    set the default verbose to 0.
    Args:
        predict_function: tensorflow.keras.Model.predict
    """
    def wrapper(instance, x, verbose=0, **kwargs):
        return predict_function(instance, x, verbose=verbose, **kwargs)
    return wrapper
