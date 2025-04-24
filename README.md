# Sentiment Analysis using Naive Bayes & Logistic Regression
# Overview
This project performs sentiment analysis on text data using Naive Bayes and Logistic Regression models. It classifies text (e.g., reviews, tweets) as positive, negative, or neutral by analyzing word patterns. The models are trained and evaluated to compare their performance in predicting sentiment.
Features

# Dataset: IMBD DATA SET FROM KAGGLE.
Algorithms: Naive Bayes (Multinomial) and Logistic Regression for classification.
Preprocessing: Tokenization, stop-word removal, and TF-IDF vectorization.
Evaluation: Metrics like accuracy, precision, recall, and F1-score visualized with Matplotlib/Seaborn.

# Tech Stack

Python: Core programming language.
Scikit-learn: Machine learning models and evaluation metrics.
Pandas/NumPy: Data manipulation and numerical computations.
NLTK/Scikit-learn: Text preprocessing and feature extraction.
Matplotlib/Seaborn: Visualization of model performance.

# Installation

Clone the repository:git clone https://github.com/your-username/sentiment-analysis-nb-lr.git


Install dependencies:pip install scikit-learn pandas numpy nltk matplotlib seaborn


Download NLTK data:python -m nltk.downloader stopwords punkt


Run the main script:python sentiment_analysis.py



# Usage

Execute sentiment_analysis.py to preprocess data, train models, and evaluate performance.
The script loads a dataset, trains Naive Bayes and Logistic Regression models, and generates performance plots.
Modify dataset paths or hyperparameters (e.g., regularization in Logistic Regression) in the script for experimentation.

# Project Timeline

Duration: October 2024 – December 2024
Developed to compare traditional machine learning approaches for sentiment classification.

# Results

Both models achieve high accuracy, with Logistic Regression often outperforming Naive Bayes on complex datasets.
Visualizations highlight trade-offs in precision, recall, and F1-score.

# Contributing
Fork the repository, report issues, or submit pull requests for improvements.
License
This project is licensed under the MIT License.
