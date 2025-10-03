# Spotiphy enables single-cell spatial whole transcriptomics via generative modeling

[![Pypi version](https://img.shields.io/pypi/v/spotiphy)](https://pypi.org/project/spotiphy/)
[![Downloads](https://static.pepy.tech/badge/spotiphy)](https://pepy.tech/project/spotiphy)
[![Github star](https://img.shields.io/github/stars/jyyulab/Spotiphy)](https://github.com/jyyulab/Spotiphy/stargazers)
[![Static Badge](https://img.shields.io/badge/Document-Latest-green)](https://jyyulab.github.io/Spotiphy)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jyyulab/Spotiphy/blob/main/tutorials/Spotiphy_tutorial_1.ipynb)
[![Zenodo](https://img.shields.io/badge/data_download-Zenodo?logo=Zenodo&labelColor=ffcc6d&color=b28e4c)](https://zenodo.org/records/10520022)

![Spotiphy_cover](https://github.com/jyyulab/Spotiphy/blob/5e7209b1b9e5524417c95ebfdc1ee9ee601587b0/figures/Cover%20image_NMETH-A55722_2.jpg)

**Spotiphy** is a Python package for integrating sequencing-based spatial transcriptomics, scRNA-seq data, and high-resolution histological images. Using a probabilistic generative model, Bayesian inference, and advanced image processing, Spotiphy performs three key tasks:

- **Deconvolution** – Estimate the abundance of each cell type in every spatial capture area.  
- **Decomposition** – Resolve bulk spatial transcriptomics data down to the single-cell level.  
- **Pseudo single-cell image reconstruction** – Generate images with pseudo single-cell resolution, enabling reconstruction of cell neighborhoods.  

These outputs enable a wide range of downstream analyses. For further details, see our [Nature Methods publication](https://www.nature.com/articles/s41592-025-02622-5).

![Spotiphy_overview](https://github.com/jyyulab/Spotiphy/blob/d62e05cb677ef6177acbda660b029ee0de1e82b3/figures/Spotiphy_overview.png)

## 📚 Tutorials & Documentation  

Currently available tutorial:  
- **Mouse cortex analysis**: [Documentation](https://colab.research.google.com/github/jyyulab/Spotiphy/blob/main/tutorials/Spotiphy_tutorial_1.ipynb) | [Google Colab](https://colab.research.google.com/github/jyyulab/Spotiphy/blob/main/tutorials/Spotiphy_tutorial_1.ipynb)  

Full documentation is available at [jyyulab.github.io/Spotiphy](https://jyyulab.github.io/Spotiphy).  

## ⚙️ Installation 

[//]: # (### Requirements)
[//]: # (+ Linux/UNIX/Windows system)
[//]: # (+ Python >= 3.9)
[//]: # (+ pytorch == 1.7.1)

We recommend using a dedicated conda environment:  
```bash
conda create -n Spotiphy-env python=3.9
conda activate Spotiphy-env
```

Spotiphy is built based on [Pytorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/), which must be installed manually before use.
```bash
# macOS with Apple Silicon
conda install -c apple tensorflow-deps -y
pip install tensorflow-macos==2.16.2 tensorflow-metal==1.2.0
pip install torch

# Windows
pip install torch  # Or follow https://pytorch.org/get-started/locally/ to install with CUDA support
pip install tensorflow==2.16.2
```

After installing the dependencies, Spotiphy itself can be installed in one of the following ways:
+ **From GitHub**: Installs the latest development version directly from the source code.
```bash
pip install git+https://github.com/jyyulab/Spotiphy.git
```
+ **From PyPI**: Installs the stable release from the Python Package Index (recommended for most users).
```bash
pip install spotiphy==0.3.1
```

To test the Installation, try to import Spotiphy in Python.
```Python
import spotiphy
```


## ❓ FAQ & Support
Frequently asked questions: [Spotiphy FAQ](https://jyyulab.github.io/Spotiphy/questions.html).

For further assistance, start a [GitHub Discussion](https://github.com/jyyulab/Spotiphy/discussions) or contact the authors:
+ Ziqian Zheng - [zzheng92@wisc.edu](mailto:zzheng92@wisc.edu)
+ Jiyuan Yang - [jiyuan.yang@stjude.org](mailto:jiyuan.yang@stjude.org)


## Cite Spotiphy:
```tex
@article{yang2025spotiphy,
  title={Spotiphy enables single-cell spatial whole transcriptomics across an entire section},
  author={Yang, Jiyuan and Zheng, Ziqian and Jiao, Yun and Yu, Kaiwen and Bhatara, Sheetal and Yang, Xu and Natarajan, Sivaraman and Zhang, Jiahui and Pan, Qingfei and Easton, John and others},
  journal={Nature Methods},
  pages={1--13},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```
[Nature Methods article](https://www.nature.com/articles/s41592-025-02622-5)

