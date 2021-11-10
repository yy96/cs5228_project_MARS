cs5228_project
==============================

This is the repository for module CS5228 project.

## Task 1
The code repository contains the working code for task 1, our main code implementation is in `/src` directory while `/notebooks` contain the work during model development stage. Before running the code, you call install the necessary package by running
```
make requirements
```
This helps to install all the dependencies needed. 
- src/features: The folder contains the codes for feature preprocessing and engineering. You can get the processed training dataset and test dataset by running the following command
```
make train_data
make test_data
```
The outputs will be saved to `data/processed/train_engineered.csv` and `data/processed/test_engineered.csv`

- src/models: The folder contains the codes for model training and predicting results. You can get trained model files and the prediction results for random forest, xgboost and catboost by running the following command (Note this step should only be run after the processed data has been made using the previous step)
```
make model_predict
```
The pickle file for the models will be saved to `models` folder, while the predicted outcomes will be saved to `data/processed/df_rf.csv`, `data/processed/df_xgboost.csv` and `data/processed/df_catboost.csv` respectively.

The work during model development stage can be found in `/notebooks` directory
- `1. yy-data-analysis.ipynb`: This notebook contains the information on initial EDA work.
- `2. yy-feature-engineering.ipynb`: This notebook contains the feature preprocessing and feature engineering step.
- `3. yy-model-training.ipynb`: This notebook contains the work for model training and hyperparameter tuning.