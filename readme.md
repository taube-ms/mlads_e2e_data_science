# MLADS Data Science Project

## Overview
This project implements a comprehensive data science pipeline for the Titanic dataset. The pipeline includes data loading, preprocessing, feature engineering, model training, evaluation, and prediction. This README provides an overview of the project structure and instructions for running the pipeline.

## Project Structure
### Folders and Files
- `mlads_ds/`: Main module containing the data science pipeline components.
- `data_loader.py`: Contains the DataLoader class for loading the dataset.
- `data_preprocessor.py`: Contains the DataPreprocessor class for data preprocessing.
- `feature_engineer.py`: Contains the FeatureEngineer class for feature engineering.
- `model_trainer.py`: Contains the ModelTrainer class for training various ML models and hyperparameter tuning.
- `model_evaluator.py`: Contains the ModelEvaluator class for evaluating model performance.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and additional insights.
- `EDA.ipynb`: Jupyter notebook for Exploratory Data Analysis (EDA) on the Titanic dataset.
- `pipeline_run.py`: Python script to run the entire data science pipeline.
- `README.md`: This file, containing project information and instructions.

## Setup and Installation
Ensure you have Python installed on your system. The project requires various libraries such as pandas, scikit-learn, xgboost, lightgbm, and catboost. You can install these dependencies via pip:

```bash
pip install pandas scikit-learn xgboost lightgbm catboost