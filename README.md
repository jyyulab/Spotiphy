# Spotiphy reveals location-specific cell subtypes through transcriptome profile at single-cell resolution

## Usage and Tutorials
Tutorials of spotiphy can be found in folder [Tutorials](https://github.com/jyyulab/Spotiphy/tree/main/tutorials).

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
+ **Install from PyPI using `pip`**: This approach is for installing the Spotiphy package from the Python Package Index 
(PyPI), which is more streamlined for users who prefer standard package installations.
```bash
pip install spotiphy==0.1.1
```

To test the Installation, try to import Spotiphy in Python.
```Python
import spotiphy
```



## Cite Spotiphy:

If you have questions, please contact the authors of the method:
+ Ziqian Zheng - zzheng92@wic.edu
+ Jiyuan Yang - jiyuan.yang@stjude.org