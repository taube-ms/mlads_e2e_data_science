from setuptools import setup, find_packages

setup(
    name="mlads_ds",
    version="0.1.0",
    author="[Your Name]",
    author_email="[Your Email]",
    description="A modular data science pipeline for Titanic survival prediction",
    url="[Your Repository URL]",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"],
    python_requires=">=3.6",
)
