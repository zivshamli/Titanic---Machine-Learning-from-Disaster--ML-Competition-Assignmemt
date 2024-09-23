# Titanic Survival Prediction - Assignment 3

## Overview

The sinking of the RMS Titanic is one of the most well-known maritime disasters in history. On April 15, 1912, the Titanic sank during her maiden voyage after hitting an iceberg, resulting in the deaths of 1,502 passengers and crew out of the 2,224 aboard. The tragedy was worsened by a lack of lifeboats, and survival often came down to various factors such as age, gender, and socio-economic status.

In this project, we aim to build a predictive model that answers the question: **"What sorts of people were more likely to survive the Titanic disaster?"** Using the Titanic dataset, we apply machine learning techniques to analyze and predict survival outcomes based on passenger data such as name, age, gender, and socio-economic class.

## Project Goals

1. **Data Preprocessing**: Cleaning and preparing the dataset for analysis by handling missing values and removing irrelevant features.
2. **Feature Engineering**: Creating new features to improve model performance and identifying significant variables that impact survival.
3. **Model Optimization**: Testing and comparing models like **Bagging Classifier**, Stochastic Gradient Descent (SGD) Classifier, and Multi-layer Perceptron (MLP), to find the optimal solution.
4. **Hyperparameter Tuning**: Identifying the best hyperparameters for each model to maximize accuracy.
5. **Data Slicing**: Finding the best method to split the data between training and validation sets.

## Dataset

The dataset used in this project contains information about the passengers aboard the Titanic, including:

- **Name**
- **Age**
- **Gender**
- **Passenger class (socio-economic status)**
- **Fare**
- **Cabin**
- **Embarked (port of departure)**

## Process

1. **Data Cleaning**: Removed irrelevant features such as passenger names and cabin numbers that do not directly impact the survival prediction. Handled missing data in features like Age and Embarked.
   
2. **Feature Engineering**: Created new features, including family size, title extraction from names, and categorization of passengers based on fare ranges.

3. **Modeling**:
   - We applied ensemble techniques such as the **Bagging Classifier** to improve prediction accuracy and reduce overfitting.
   - Compared with other models, such as the **SGD Classifier** and **MLP (Multi-layer Perceptron)**, to identify the best performing model.
   - Fine-tuned hyperparameters using cross-validation to ensure the models were performing optimally.

4. **Results**: After model comparison and optimization, we identified the model with the highest prediction accuracy and analyzed feature importance to understand the factors most critical for survival prediction.

---

## How to Run

### Prerequisites

Make sure you have the following installed:

- **Python 3.x**
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib` (optional, for visualization)
  - `seaborn` (optional, for visualization)

Install the required libraries using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
