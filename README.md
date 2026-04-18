# Real Estate Price Prediction Pipeline

## Project Overview

This project implements a complete **end-to-end Machine Learning pipeline** for analyzing and predicting real estate property prices. It covers data preprocessing, exploratory data analysis (EDA), model development, unsupervised learning, and evaluation, along with a simple **Streamlit web application** for interaction.

The goal is to build a reliable model that can estimate property prices based on various features such as location, area, and property characteristics.

---

##  Key Features

* Data Cleaning & Preprocessing
*  Exploratory Data Analysis (EDA)
*  Supervised Learning (Regression Models)
*  Unsupervised Learning (Clustering)
*  Model Evaluation & Comparison
*  Interactive Web App using Streamlit

---

## Project Structure

```
ml_project/
│
├── data/
│   ├── property_data.csv
│   └── processed_data.csv
│
├── notebooks/
│   ├── Phase1_EDA.ipynb
│   ├── Phase2_Modeling.ipynb
│   ├── Phase3_Unsupervised.ipynb
│   └── Phase4_Evaluation.ipynb
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Pipeline Workflow

### 🔹 Phase 1: Data Preprocessing & EDA

* Handling missing values
* Data cleaning and transformation
* Visualization of trends and relationships

### 🔹 Phase 2: Model Development

* Feature selection and engineering
* Training regression models
* Hyperparameter tuning

### 🔹 Phase 3: Unsupervised Learning

* Clustering techniques to identify patterns
* Grouping similar properties

### 🔹 Phase 4: Model Evaluation

* Performance comparison of models
* Metrics used:

  * R² Score
  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)

---

##  Dataset

The dataset contains real estate property information such as:

* Property prices
* Location details
* Area (size)
* Number of rooms
* Additional relevant features

---

## Running the Streamlit App

To launch the interactive application:

```
cd app
streamlit run app.py
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

##  Results & Insights

* Built and compared multiple regression models
* Identified the best-performing model based on evaluation metrics
* Extracted meaningful insights from clustering techniques

---

## Future Improvements

* Deploy the model online (e.g., cloud platform)
* Integrate real-time data
* Improve feature engineering
* Enhance UI/UX of the Streamlit app

---

## Authors

* **Zayna Qasim**
* **Hamna Shah**
* **Sukaina Nasir**


---

## ⭐ Acknowledgment

This project was developed as part of academic learning and hands-on practice in building real-world machine learning pipelines.
