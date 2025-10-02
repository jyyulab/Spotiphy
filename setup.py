from setuptools import setup

required_packages = [
    "squidpy==1.6.1",
    "scanpy==1.10.3",
    "scikit-learn==1.6.1",
    "scipy==1.13.1",
    "pandas==2.3.2",
    "numpy==1.26.4",
    "matplotlib",
    "seaborn",
    "tqdm",
    "charset-normalizer==3.4.3",
    # "tensorflow==2.12.0",
    "stardist==0.9.1",
    # "torch",
    "pyro-ppl==1.9.1",
    "opencv-python>=4.10",
    "jupyter",
]

setup(
    name="spotiphy",
    version="0.3.1",
    packages=["spotiphy"],
    url="https://github.com/jyyulab/Spotiphy",
    license="Apache-2.0",
    author="Ziqian Zheng, Jiyuan Yang",
    author_email="zzheng92@wisc.edu",
    description="An integrated pipeline designed to deconvolute and decompose spatial transcriptomics data, "
    "and produce pseudo single-cell resolution images.",
    python_requires=">=3.9",
    install_requires=required_packages,
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
