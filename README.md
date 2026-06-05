# AI-Powered Career Path Suggestion System

An AI-powered Career Path Suggestion System that analyzes a user's skills and recommends relevant career opportunities. The project combines Natural Language Processing (NLP) and Machine Learning to match skill sets with suitable career roles and industry clusters.

## Overview

This project was built to understand and implement a complete Machine Learning workflow from data preprocessing to prediction. The system transforms textual skills into numerical representations using TF-IDF Vectorization and uses a Multinomial Naive Bayes model to predict the most relevant career paths.

The model is designed to efficiently handle thousands of career titles while providing accurate and fast recommendations.

## Features

* Skill-based career recommendations
* TF-IDF feature extraction for text processing
* Multinomial Naive Bayes classification
* Best match along with alternative career suggestions
* Career-to-cluster mapping
* Lightweight and fast prediction system
* Complete end-to-end Machine Learning pipeline

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib

## Project Workflow

### 1. Data Loading & Cleaning

* Load and inspect the dataset
* Remove missing or invalid records
* Prepare clean data for training

### 2. Data Preprocessing

* Clean and normalize text data
* Create Career-to-Cluster mapping
* Convert skills into TF-IDF vectors
* Split data into training and testing sets
* Save vectorizers and processed datasets

### 3. Model Training

* Train the Multinomial Naive Bayes classifier
* Learn relationships between skills and careers
* Save the trained model

### 4. Model Evaluation

* Evaluate performance on unseen data
* Analyze prediction accuracy

### 5. Career Prediction

* Accept user skills as input
* Predict the most relevant career roles
* Display recommended careers and corresponding clusters

## Project Structure

```text
Career_Path_Suggestion_System/
│
├── Career_Dataset_Clustered.csv
├── cleaned_careers_clustered.csv
│
├── step1_load_data.py
├── step2_preprocess.py
├── step3_train_model.py
├── step4_evaluate.py
├── step5_predict.py
│
├── tfidf_vectorizer.pkl
├── career_model.pkl
├── career_to_cluster.pkl
├── X_train_numeric.pkl
├── X_test_numeric.pkl
├── y_train.pkl
├── y_test.pkl
│
└── README.md
```

## Installation

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn joblib
```

## Running the Project

Execute the pipeline in the following order:

```bash
python step1_load_data.py
python step2_preprocess.py
python step3_train_model.py
python step4_evaluate.py
python step5_predict.py
```

## Example

**Input Skills**

```text
Python, SQL, Machine Learning, Data Visualization, Communication
```

**Output**

```text
Top Match: Data Scientist
Cluster: Data & Analytics

Alternative Suggestions:
- Machine Learning Engineer
- Data Analyst
- Business Intelligence Analyst
```

## Key Learnings

Through this project, I gained practical experience in:

* Data Cleaning and Preprocessing
* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Machine Learning Model Training
* Model Evaluation and Testing
* Building End-to-End ML Pipelines
* Model Serialization using Joblib

