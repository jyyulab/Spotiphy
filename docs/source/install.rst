Installation
=====================

To install Spotiphy, we recommend creating a separate conda environment. This helps manage
dependencies and avoid conflicts with other packages.

.. code-block::

    conda create -n Spotiphy-env python=3.9
    conda activate Spotiphy-env

Dependencies
---------------------

Spotiphy is built on top of `PyTorch <https://pytorch.org/>`_ and `TensorFlow <https://www.tensorflow.org/>`.  
These must be installed manually before installing Spotiphy.

.. code-block:: bash

    # macOS with Apple Silicon
    conda install -c apple tensorflow-deps -y
    pip install tensorflow-macos==2.16.2 tensorflow-metal==1.2.0
    pip install torch

    # Windows
    pip install torch  # Or follow https://pytorch.org/get-started/locally/ to enable CUDA support
    pip install tensorflow==2.16.2

Spotiphy Installation
---------------------

After installing the dependencies, Spotiphy itself can be installed in one of two ways:

+ **From GitHub**: Installs the latest development version directly from the source code.

  .. code-block:: bash

      pip install git+https://github.com/jyyulab/Spotiphy.git

+ **From PyPI**: Installs the stable release from the Python Package Index (recommended for most users).

  .. code-block:: bash

      pip install spotiphy==0.3.0

Verification
---------------------

To test the installation, try importing Spotiphy in Python:

.. code-block:: python

    import spotiphy
