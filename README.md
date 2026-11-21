
# 🤖 PAC Learning Agent V2

## Project Overview

PAC Learning Agent is an interactive Machine Learning
simulator developed using Python and Streamlit.

PAC stands for Probably Approximately Correct.
It evaluates how well machine learning algorithms
learn from training data and generalize to unseen data.

## Project Objectives

- Understand PAC Learning concepts.
- Evaluate model accuracy and error.
- Compare different ML algorithms.
- Analyze Epsilon and Delta parameters.
- Study the effect of training sample size.
- Visualize learning curves.
- Generate downloadable evaluation reports.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Machine Learning Algorithms

1. Decision Tree Classifier
2. K-Nearest Neighbors (KNN)
3. Logistic Regression

## Main Features

- CSV dataset upload.
- Automatic sample dataset.
- Target column selection.
- ML model comparison.
- PAC-style evaluation.
- Epsilon vs Delta visualization.
- Training sample-size analysis.
- Learning curve visualization.
- Confusion matrix.
- CSV evaluation report download.

## Project Structure

PAC_Learning_Agent/
|
|-- app.py
|-- README.md
|-- requirements.txt
|
|-- datasets/
    |-- housing_dataset  
    |-- breastcancer_datset

## Installation

### Step 1: Clone Repository

git clone YOUR_GITHUB_REPOSITORY_URL

### Step 2: Open Project Folder

cd PAC_Learning_Agent

### Step 3: Create Virtual Environment

python -m venv venv

### Step 4: Activate Environment

Windows:

venv\Scripts\activate

### Step 5: Install Dependencies

pip install -r requirements.txt

### Step 6: Run Application

streamlit run app.py

## PAC Parameters

Epsilon (ε):
Maximum acceptable prediction error.

Delta (δ):
Probability that the learning guarantee fails.

Confidence:
1 - Delta

PAC-style condition:
Upper Error Bound <= Epsilon

## Dataset

The application supports CSV datasets.

Users select a target column before training
the machine learning models.

The default dataset is Breast Cancer Wisconsin.

## Evaluation

The application evaluates:

- Accuracy
- Test error
- Upper confidence bound
- Confusion matrix
- Learning curves
- Training sample-size performance

## Important Note

This project demonstrates PAC-style statistical
evaluation using a held-out test dataset.

It is an educational simulator, not a formal
proof of PAC learnability.

## Future Improvements

- More machine learning algorithms.
- Larger datasets.
- Hyperparameter tuning.
- Better PAC sample complexity analysis.
- Advanced visualizations.

## Project Status

Version 2.0 - Educational ML Simulator
