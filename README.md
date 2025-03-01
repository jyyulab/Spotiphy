# Spotiphy enables single-cell spatial whole transcriptomics via generative modeling

[![Pypi version](https://img.shields.io/pypi/v/spotiphy)](https://pypi.org/project/spotiphy/)
[![Downloads](https://static.pepy.tech/badge/spotiphy)](https://pepy.tech/project/spotiphy)
[![Github star](https://img.shields.io/github/stars/jyyulab/Spotiphy)](https://github.com/jyyulab/Spotiphy/stargazers)
[![Static Badge](https://img.shields.io/badge/Document-Latest-green)](https://jyyulab.github.io/Spotiphy)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jyyulab/Spotiphy/blob/main/tutorials/Spotiphy_tutorial_1.ipynb)
[![Zenodo](https://img.shields.io/badge/data_download-Zenodo?logo=Zenodo&labelColor=ffcc6d&color=b28e4c)](https://zenodo.org/records/10520022)

![Spotiphy_cover](https://github.com/jyyulab/Spotiphy/blob/9a16882511aef6e0e7db9154f0d9f59a2c567c6f/figures/cover.png)

Spotiphy is a Python-based pipeline designed to enhance our understanding of biological tissues by integrating sequencing-based spatial transcriptomics data, scRNA-seq data, and high-resolution histological images. Employing a probabilistic model, Bayesian inference, and advanced image processing techniques, Spotiphy primarily executes three key tasks:
- **Deconvolution**: Spotiphy estimates the abundance of each cell type in each capture area of spatial tissue.
- **Decomposition**: Spotiphy decomposes spatial transcriptomics data to the single-cell level.
- **Pseudo single-cell resolution image**: Spotiphy generates a pseudo single-cell resolution image to reconstruct cell neighbors.

With these outputs, Spotiphy facilitates numerous downstream analyses. For more detailed information, please refer to the associated research paper.

![Spotiphy_overview](https://github.com/jyyulab/Spotiphy/blob/d62e05cb677ef6177acbda660b029ee0de1e82b3/figures/Spotiphy_overview.png)

## Tutorials and documents

The following tutorial are available:

+ Deconvolution and decomposition of mouse cortex with Spotiphy [[document](https://colab.research.google.com/github/jyyulab/Spotiphy/blob/main/tutorials/Spotiphy_tutorial_1.ipynb)][[Google Colab](https://colab.research.google.com/github/jyyulab/Spotiphy/blob/main/tutorials/Spotiphy_tutorial_1.ipynb)]


For more details, please refer to the [documents](https://jyyulab.github.io/Spotiphy).

## Installation

[//]: # (### Requirements)
[//]: # (+ Linux/UNIX/Windows system)
[//]: # (+ Python >= 3.9)
[//]: # (+ pytorch == 1.7.1)

To install Spotiphy, it is recommended to create a separate conda environment. This approach helps to manage 
dependencies and avoid conflicts with other packages.
```bash
conda create -n Spotiphy-env python=3.9
conda activate Spotiphy-env
```

Spotiphy is built based on [Pytorch](https://pytorch.org/). Although installing Spotiphy automatically includes PyTorch,
it is recommended that users manually install PyTorch [(link)](https://pytorch.org/get-started/locally/) to allow for 
more flexibility, particularly for those who wish to utilize CUDA capabilities.
We offer two methods for installing the Spotiphy package:
+ **Install from GitHub**: This method allows you to install the latest version directly from the source code hosted on 
GitHub.
```bash
pip install git+https://github.com/jyyulab/Spotiphy.git
```
+ **Install from PyPI**: This approach is for installing the Spotiphy package from the Python Package Index 
(PyPI), which is more streamlined for users who prefer standard package installations.
```bash
pip install spotiphy
```

To test the Installation, try to import Spotiphy in Python.
```Python
import spotiphy
```


## Frequently asked questions
Answers to frequently asked questions can be found [here](https://jyyulab.github.io/Spotiphy/questions.html).

Should you have any further questions, feel free to start a [discussion](https://github.com/jyyulab/Spotiphy/discussions) or reach out directly to the package authors:
+ Ziqian Zheng - [zzheng92@wisc.edu](mailto:zzheng92@wisc.edu)
+ Jiyuan Yang - [jiyuan.yang@stjude.org](mailto:jiyuan.yang@stjude.org)


## Cite Spotiphy:

Pending
