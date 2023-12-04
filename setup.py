from setuptools import setup

required_packages = [
    "squidpy==1.2.3",
    "scanpy==1.9.3",
    "scikit-learn==1.2.2",
    "scipy==1.9.1",
    "pandas==2.0.0",
    "numpy==1.22.4",
    "matplotlib",
    "seaborn",
    "tqdm",
    "charset-normalizer==3.1.0",
    "tensorflow==2.12.0",
    "stardist==0.8.3",
    "torch",
    "pyro-ppl==1.8.4",
    "opencv-python>=4.7",
    "jupyter"
]

setup(
    name='spotiphy',
    version='0.1.2',
    packages=['spotiphy'],
    url='https://github.com/jyyulab/Spotiphy',
    license='Apache-2.0',
    author='Ziqian Zheng',
    author_email='zzheng92@wisc.edu',
    description='An integrated pipeline designed to deconvolute and decompose spatial transcriptomics data, '
                'and produce pseudo single-cell resolution images.',
    python_requires=">=3.9",
    install_requires=required_packages
)
