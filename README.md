# Personal Finance Daily Expenses Tracking Prediction Using Machine Learning

This repository contains a project that explores the prediction of daily expenses and the discovery of underlying spending patterns using machine learning. The project involves generating a synthetic dataset that simulates a year’s worth of financial transactions for a student, building supervised regression models for expense prediction, and applying unsupervised clustering techniques to reveal distinct spending behaviors.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
- [Experiments and Results](#experiments-and-results)
- [Usage](#usage)

## Introduction

In today's fast-paced world, personal financial management is essential. This project investigates whether machine learning can accurately predict daily expenses and identify distinct spending profiles. By simulating a comprehensive dataset of daily transactions (including income, fixed expenses, and variable daily spending), we apply both supervised and unsupervised learning methods. The goal is twofold: to forecast daily expenses using a Random Forest regression model and to uncover latent clusters in spending behavior through K-Means and Gaussian Mixture Models (GMM).

## Dataset

The dataset is generated using a custom Python script (`generate_data.py`) that simulates one full year (January 1, 2025 – December 31, 2025) of financial transactions for a student. The simulation includes:

- **Income:** Scholarship payments, parental support, and part-time job earnings (e.g., from Burger King and office assistant roles).
- **Expenses:** Fixed monthly costs (e.g., transport, entertainment subscriptions, groceries), daily variable expenses (meals, coffee, dinner, laundry), occasional special events (weekend dining out), and gym-related spending.

**Key features include:**
- **Date:** The transaction date.
- **Category:** Labels such as Income, Transport, Entertainment, Groceries, Meal, Coffee, Food & Drink, Gym, and Laundry.
- **Amount_NTD:** Transaction amount in New Taiwan Dollars.
- **Additional Metadata:** Includes a description, payment method, and transaction time.

The generated dataset is saved as `daily_expenses.csv` and is also available online on [GitHub](https://github.com/benedictdavon/AI-Capstone-Homework-1).

## Methodology

### Supervised Learning

To predict daily expenses, we employed a Random Forest regression model using the open-source [scikit-learn](https://scikit-learn.org/) library. Our supervised learning pipeline was implemented in Python and leveraged other widely adopted libraries such as Pandas and NumPy for data manipulation, and Matplotlib and Seaborn for visualization ([pandas.pydata.org](https://pandas.pydata.org/), [numpy.org](https://numpy.org/), [matplotlib.org](https://matplotlib.org/), [seaborn.pydata.org](https://seaborn.pydata.org/)). The entire codebase, including data generation and preprocessing, is available as open-source on GitHub.

**The primary steps in our supervised learning workflow are:**

- **Feature Engineering:**  
  We derived several informative features from the raw date:
  - **DayOfWeek:** An integer (Monday=0 to Sunday=6) representing the day of the week.
  - **IsWeekend:** A binary indicator (1 if Saturday or Sunday, 0 otherwise).
  - **Month and Day:** Directly extracted from the date.
  - **Lag1:** The expense value from the previous day to capture short-term dependencies.
  - **Rolling7:** A 7-day moving average of expenses to capture recent spending trends.
  - **LogExpense:** The logarithm (after adding 1) of the daily expense, used to reduce the influence of outliers.

- **Model Training and Evaluation:**  
  We trained the Random Forest model on four different feature sets:
  - **Feature Set A:** Excluding both Lag1 and Rolling7.
  - **Feature Set B:** Including Lag1 only.
  - **Feature Set C:** Including Rolling7 only.
  - **Feature Set D:** Including both Lag1 and Rolling7.

  A grid search combined with TimeSeriesSplit cross-validation was used for hyperparameter tuning. The evaluation metric was the Root Mean Squared Error (RMSE), and we also analyzed feature importances to understand the contribution of each predictor.

- **Additional Methods:**  
  Data resampling techniques ensured a balanced representation of the temporal data. Dimensionality reduction, specifically Principal Component Analysis (PCA), was applied to visualize high-dimensional features in two dimensions. This aided in both exploratory data analysis and in validating the clustering results.

### Unsupervised Learning

Clustering techniques were employed to explore underlying spending patterns:

- **K-Means Clustering:**  
  We applied K-Means (using scikit-learn) and evaluated the optimal number of clusters using multiple metrics, including the Elbow method, Silhouette score, Davies–Bouldin Index, and Calinski–Harabasz Index. Our experiments indicated an optimal k of 6 according to the Elbow method, although other metrics provided different perspectives, highlighting the complexity of the dataset's structure.

- **Gaussian Mixture Models (GMM):**  
  To obtain a probabilistic perspective on clustering, we applied Gaussian Mixture Models using scikit-learn. GMM provided soft cluster assignments along with confidence scores (GMM_Confidence), offering insights into the certainty of each point's cluster membership—especially for points near cluster boundaries or in outlier regions.

- **Visualization:**  
  PCA was employed to reduce the dimensionality of the feature space for visualization purposes, making it easier to interpret the clustering results.

**References for Public Libraries and Open-Source Tools:**

- **Python:** [python.org](https://www.python.org/)
- **Pandas:** [pandas.pydata.org](https://pandas.pydata.org/)
- **NumPy:** [numpy.org](https://numpy.org/)
- **scikit-learn:** [scikit-learn.org](https://scikit-learn.org/)
- **Matplotlib:** [matplotlib.org](https://matplotlib.org/)
- **Seaborn:** [seaborn.pydata.org](https://seaborn.pydata.org/)

Pretrained models were not directly used in this project; instead, the focus was on building and tuning models from scratch using custom-generated data. This approach ensured that our experiments were fully reproducible using open-source tools and publicly available libraries.

## Experiments and Results

### Supervised Learning Experiments

The Random Forest regression model was evaluated using different feature sets. Results showed that:
- Excluding temporal features (Feature Set A) resulted in higher RMSE.
- Incorporating a rolling average (Feature Set C) and combining it with lag features (Feature Set D) significantly improved predictive performance.
- Cross-validation confirmed model stability, and feature importance analysis consistently highlighted **DayOfWeek** and **IsWeekend** as the most influential predictors.

### Unsupervised Learning Experiments

Clustering analyses revealed that the dataset naturally segments into distinct groups. K-Means clustering, when using an optimal k of 6, identified clusters with unique spending profiles. For instance, some clusters represented consistent low-spending patterns while others captured extreme or outlier behaviors. GMM further confirmed these profiles by providing probabilistic cluster assignments with high confidence scores for most data points.

Additionally, experiments examining how cluster assignments change from k to k+1 demonstrated that most transitions are stable, except for specific increases (e.g., from k=7 to k=8 and k=20 to k=21) where an abrupt change indicates potential over-segmentation. Clustering the data based on only the initial X months and then evaluating on future data revealed that a longer training period generally leads to more stable and representative clusters.
