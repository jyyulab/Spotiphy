from setuptools import setup

setup(
    name='spotiphy',
    version='0.1.0',
    packages=['spotiphy'],
    url='https://github.com/jyyulab/Spotiphy',
    license='Apache-2.0',
    author='Ziqian Zheng',
    author_email='zzheng92@wisc.edu',
    description='An integrated pipeline designed to deconvolute and decompose spatial transcriptomics data, '
                'and produce pseudo single-cell resolution images.',
    python_requires=">=3.8.5",
    install_requires=[
        "pip",
        "torch",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "scanpy",
        "scikit-learn",
        "tqdm",
    ]
)
