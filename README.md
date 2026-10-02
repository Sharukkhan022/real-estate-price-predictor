# House Price Prediction Using Multiple Linear Regression

A machine learning project focused on predicting residential property prices using a multiple linear regression model. The project walks through the full workflow of working with a real-world housing dataset: gathering the data, exploring patterns, cleaning and preparing it, training the model, evaluating performance, and building a simple prediction application.

## Project Overview

This project uses a public housing dataset from Kaggle to estimate the price of a property based on a set of features such as:

- number of bedrooms
- number of bathrooms
- square footage
- sale year
- sale month
- city average price

The model predicts the target variable:

- Price (USD)

The solution is implemented in Python using:

- pandas for data processing
- scikit-learn for machine learning
- joblib for model serialization
- Streamlit for the web-based user interface

## Objective

The primary goal of this project is to build an accurate and usable house price prediction model that can estimate the value of a home based on key property and market characteristics.

## Dataset Source

The dataset was obtained from Kaggle and stored locally in the project under:

- `data/redfin_housing.csv`

It contains housing records with attributes such as address, city, listing date, bedroom count, bathroom count, square footage, and sale price.

## Workflow

### 1. Data Extraction

The dataset was downloaded from Kaggle and placed in the `data/` directory. The raw CSV file was then loaded into a pandas DataFrame for analysis.

### 2. Dataset Study and Exploration

During the exploratory data analysis phase, the project examined:

- column names
- sample rows
- null/missing values
- feature types
- city distribution
- date format and temporal values

This helped identify the structure of the dataset and understand which features were usable for modeling.

### 3. Data Cleaning and Feature Engineering

Before training the model, the dataset was cleaned and transformed to improve quality and make it suitable for regression.

The cleaning process included:

- converting the raw date column into `Sale_Year` and `Sale_Month`
- removing unnecessary text-based fields such as address and original date
- handling missing values in numeric features using median imputation
- computing average house price by city
- creating a `City_Avg_Price` feature for city-based pricing trends
- dropping irrelevant or unusable columns
- saving the cleaned output to `data/cleaned_redfin_housing.csv`

This step is essential because machine learning models perform better when the data is consistent, structured, and free from unnecessary noise.

### 4. Model Training

A multiple linear regression model was used to learn the relationship between independent variables and the target variable.

The training process included:

- separating features (`X`) and target (`y`)
- splitting the data into training and testing sets
- fitting the LinearRegression model
- generating predictions on the test set
- evaluating model performance using R-squared and RMSE

### 5. Model Evaluation

The model was evaluated using the following metrics:

- R-squared: 0.3804
- RMSE: 1304306.74

These values indicate how well the model explains the variation in house prices and how far the predicted values deviate from the actual prices.

### 6. Model Saving

The trained model was saved using joblib in the `models/` directory:

- `models/linear_regression_model.pkl`

### 7. Streamlit Web Application

A web application was created using Streamlit so users can enter property details and get an estimated house price prediction.

The app includes:

- number of bedrooms
- number of bathrooms
- area in SQFT
- sale year
- sale month
- city selection

The model loads from the trained pickle file and predicts the estimated price based on the user inputs.

## Project Structure

```text
HousePricePrediction_MultilinearRegression/
├── app.py
├── README.md
├── reqirements.txt
├── data/
│   ├── redfin_housing.csv
│   └── cleaned_redfin_housing.csv
├── models/
│   └── linear_regression_model.pkl
└── src/
    ├── clean_data.py
    └── train_model.py
```

## Technologies Used

- Python
- pandas
- scikit-learn
- joblib
- Streamlit
- Kaggle dataset

## Setup Instructions

1. Clone the repository.
2. Create a virtual environment if needed.
3. Install dependencies:

```bash
pip install -r reqirements.txt
```

4. Run the data cleaning step:

```bash
python src/clean_data.py
```

5. Train the model:

```bash
python src/train_model.py
```

6. Start the Streamlit app:

```bash
streamlit run app.py
```

## Notes

This project is a practical example of how a real-world housing dataset can be transformed into a predictive model using supervised machine learning. It demonstrates key steps common in data science projects, including data preparation, feature engineering, model training, and deployment as a user-facing application.

## Conclusion

This house price prediction project successfully demonstrates the use of multiple linear regression for estimating property values from multiple housing-related features. It combines data analysis, machine learning, and application development into a complete end-to-end workflow.
