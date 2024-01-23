.. Spotiphy documentation master file, created by
   sphinx-quickstart on Tue Dec  5 11:26:42 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: _contributors.rst

Spotiphy!
====================================
|Github| |PyPI| |PyPI_downloads| |Open_in_colab| |Download Data|

.. |Github| image:: https://img.shields.io/badge/View_in_Github-6C51D4?logo=github&logoColor=white
   :target: https://github.com/jyyulab/Spotiphy
.. |PyPI| image:: https://img.shields.io/pypi/v/spotiphy
   :target: https://pypi.org/project/spotiphy/
.. |PyPI_downloads| image:: https://static.pepy.tech/badge/spotiphy
   :target: https://pepy.tech/project/spotiphy
.. |Open_in_colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/jyyulab/Spotiphy/blob/main/tutorials/Spotiphy_tutorial_1.ipynb
.. |Download Data| image:: https://img.shields.io/badge/data_download-Zenodo?logo=Zenodo&labelColor=ffcc6d&color=b28e4c
   :target: https://zenodo.org/records/10520022

Spotiphy is a Python-based pipeline designed to enhance our understanding of biological tissues by integrating sequencing-based spatial transcriptomics data, scRNA-seq data, and high-resolution histological images. Employing a probabilistic model, Bayesian inference, and advanced image processing techniques, Spotiphy primarily executes three key tasks:

- **Deconvolution**: Spotiphy estimates the abundance of each cell type in each capture area of spatial tissue.
- **Decomposition**: Spotiphy decomposes spatial transcriptomics data to the single-cell level.
- **Pseudo single-cell resolution image**: Spotiphy generates a pseudo single-cell resolution image to reconstruct cell neighbors.

With these outputs, Spotiphy facilitates numerous downstream analyses. For more detailed information, please refer to the associated research paper.



.. image:: _static/figures/cover.png
    :align: center

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Basic

   install
   tutorials
   questions

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   scRNA Reference
   Deconvolution and Decomposition
   Segmentation
   Visualization

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: About

   About