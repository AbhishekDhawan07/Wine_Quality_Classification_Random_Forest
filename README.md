# 🍷 Wine_Quality_Classification_Random_Forest

A supervised machine learning project that predicts the quality of red
wine using a **Random Forest Classifier**. This project demonstrates a
complete ML pipeline --- from data exploration and preprocessing to
model training, evaluation, and prediction.

------------------------------------------------------------------------

## 📋 Table of Contents

1.  [Project Overview](#project-overview)
2.  [Features](#features)
3.  [Dataset Description](#dataset-description)
4.  [Project Workflow](#project-workflow)
5.  [Tech Stack](#tech-stack)
6.  [Project Structure](#project-structure)
7.  [Getting Started](#getting-started)
8.  [Model Performance](#model-performance)
9.  [Sample Prediction](#sample-prediction)
10. [Key Insights](#key-insights)
11. [Author](#author)
12. [Contributing](#contributing)

------------------------------------------------------------------------

## Project Overview

This project builds a machine learning model to classify the **quality
of red wine** based on its chemical properties.

Using a **Random Forest Classifier**, the model learns patterns from
physicochemical features such as acidity, alcohol, and sulphates to
predict wine quality scores.

The project highlights an end-to-end ML workflow with a strong focus on
interpretability and performance.

------------------------------------------------------------------------

## Features

-   Complete **EDA (Exploratory Data Analysis)**
-   Data cleaning and preprocessing
-   Feature importance analysis using Random Forest
-   Train-test split with reproducibility
-   Model training using ensemble learning
-   Evaluation using accuracy and confusion matrix
-   Predictive system for new wine samples

------------------------------------------------------------------------

## Dataset Description

The dataset `winequality-red.csv` contains red wine samples with
chemical attributes.

**Dataset stats:** - Multiple records of red wine samples - Input
features: physicochemical properties - Output: quality score (integer)

**Key Features:** - Fixed acidity - Volatile acidity - Citric acid -
Residual sugar - Chlorides - Free sulfur dioxide - Total sulfur
dioxide - Density - pH - Sulphates - Alcohol

**Target Variable:** - `quality` (score typically between 0--10)

------------------------------------------------------------------------

## Project Workflow

### 🟢 Step 1 --- Data Loading

``` python
df = pd.read_csv("winequality-red.csv")
df.head()
```

------------------------------------------------------------------------

### 🔵 Step 2 --- Data Exploration

-   Check dataset shape, info, and summary statistics
-   Identify missing values
-   Analyze feature distributions
-   Visualize correlations

------------------------------------------------------------------------

### 🟡 Step 3 --- Data Preprocessing

-   Handle missing values (if any)
-   Feature scaling (if applied)
-   Separate features and target

``` python
X = df.drop("quality", axis=1)
y = df["quality"]
```

------------------------------------------------------------------------

### 🟠 Step 4 --- Train-Test Split

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

------------------------------------------------------------------------

### 🔴 Step 5 --- Model Training (Random Forest)

``` python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

------------------------------------------------------------------------

### 🟣 Step 6 --- Model Evaluation

``` python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

------------------------------------------------------------------------

### ⚫ Step 7 --- Prediction System

``` python
sample = X.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample)
```

------------------------------------------------------------------------

## Tech Stack

-   **Language**: Python
-   **Libraries**:
    -   pandas
    -   numpy
    -   matplotlib / seaborn
    -   scikit-learn
-   **Environment**: Jupyter Notebook

------------------------------------------------------------------------

## Project Structure

    Wine_Quality_Classification_Random_Forest/
    │
    ├── README.md
    │
    └── Random Forest Project - Wine Quality Classification/
        ├── Random Forest Project - Wine Quality Classification.ipynb
        └── winequality-red.csv

------------------------------------------------------------------------

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/<your-username>/Wine_Quality_Classification_Random_Forest.git
cd Wine_Quality_Classification_Random_Forest
```

### 2. Install dependencies

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Run the notebook

``` bash
jupyter notebook
```

------------------------------------------------------------------------

## Model Performance

-   Random Forest provides strong performance due to ensemble learning
-   Handles non-linearity and feature interactions well
-   More robust compared to single decision trees

------------------------------------------------------------------------

## Sample Prediction

``` python
Prediction Output: [5]
→ "Predicted Wine Quality Score is 5"
```

------------------------------------------------------------------------

## Key Insights

-   Alcohol and sulphates are often strong predictors of wine quality
-   Random Forest reduces overfitting compared to single trees
-   Dataset is clean and suitable for ML tasks
-   Feature importance helps interpret model decisions

------------------------------------------------------------------------

## Author

This project is created as a Machine Learning Portfolio project
demonstrating classification using ensemble methods.

------------------------------------------------------------------------

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and submit a pull request 🚀

-----------------------------------------------------------------------

> ⭐ If you found this project useful, consider starring the repository!
