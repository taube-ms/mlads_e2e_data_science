from setuptools import setup, find_packages

setup(
    name="mlads_ds",  # Name of your package
    version="0.1.0",  # Initial version of your package
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    description="Data Science pipeline for Titanic dataset analysis",  # Brief description of your package
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # Set the content type to markdown as we are using a README.md
    url="https://github.com/yourusername/mlads_ds",  # Replace with the URL of your repository
    packages=find_packages(),  # Automatically find your package
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "xgboost",
        "lightgbm",
        "catboost"
        # Add other dependencies as needed
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",  # Development status of your package
        "Intended Audience :: Developers",  # Define the audience of your package
        "License :: OSI Approved :: MIT License",  # License of your package
        "Programming Language :: Python :: 3",  # Programming language
        "Programming Language :: Python :: 3.8",  # Specific version of the programming language
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",  # Minimum version requirement of Python
)
