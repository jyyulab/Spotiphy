Installation
=====================

To install Spotiphy, it is recommended to create a separate conda environment. This approach helps to manage
dependencies and avoid conflicts with other packages.

.. code-block::

    conda create -n Spotiphy-env python=3.9
    conda activate Spotiphy-env


Spotiphy is built based on `Pytorch <https://pytorch.org/>`_. Although installing Spotiphy automatically includes PyTorch,
it is recommended that users manually install PyTorch `link <https://pytorch.org/get-started/locally/>`_ to allow for
more flexibility, particularly for those who wish to utilize CUDA capabilities.

We offer two methods for installing the Spotiphy package:

+ **Install from GitHub**: This method allows you to install the latest version directly from the source code hosted on GitHub.

.. code-block::

    pip install git+https://github.com/jyyulab/Spotiphy.git

+ **Install from PyPI**: This approach is for installing the Spotiphy package from the Python Package Index (PyPI), which is more streamlined for users who prefer standard package installations.

.. code-block::

    pip install spotiphy==0.1.2


To test the Installation, try to import Spotiphy in Python.

.. code-block::

    import spotiphy
    