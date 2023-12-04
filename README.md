# Titanic Survival Prediction: Data Science Pipeline

This repository contains a modular data science pipeline for predicting survival on the Titanic. It's structured into distinct components, each responsible for a specific aspect of the pipeline, from data loading to model evaluation.

## Repository Structure

- `mlads_ds/`: Main module directory containing the pipeline components.
    - `data_loader.py`: Contains the `DataLoader` class for loading the dataset.
    - `data_preprocessor.py`: Contains the `DataPreprocessor` class for data cleaning and preprocessing.
    - `feature_selector.py`: Contains the `FeatureSelector` class for selecting relevant features.
    - `model_trainer.py`: Contains the `ModelTrainer` class for training and tuning models.
    - `model_evaluator.py`: Contains the `ModelEvaluator` class for evaluating model performance.
- `notebooks/`: Jupyter notebooks for running the pipeline and experiments.
    - `run_pipeline.ipynb`: Notebook to run the entire pipeline and compare model performances.

## Installation

To set up the project, clone this repository to your local machine:

## git clone
```bash
git clone [Your Repository URL]
```

## create a vm
```bash
python -m venv venv
source venv/bin/activate  # For Unix or MacOS
venv\Scripts\activate  # For Windows
```

## install requirements
```bash
pip install -r requirements.txt
```


