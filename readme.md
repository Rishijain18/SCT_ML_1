# House Price Prediction using Linear Regression

This project implements a **Linear Regression** model to predict the prices of houses based on:

- 🏠 Square Footage (area)
- 🛏️ Number of Bedrooms
- 🛁 Number of Bathrooms

## 📊 Dataset
A CSV file containing real estate listings with:
- `area`, `bedrooms`, `bathrooms`, and `price`

## 🔍 Objective
Predict house prices using Linear Regression and evaluate with R² Score and Mean Squared Error.

## ⚙️ Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib

## 📈 Model Performance
- Evaluation Metrics:
  - **R² Score**: ~0.48–0.58 (depends on dataset quality)
  - **Mean Squared Error**: Varies by data scale

## 📎 Files Included
- `linear_regression.py`: Main Python script
- `house_data.csv`: Dataset file
- `README.md`: Project overview

## 🚀 How to Run
```bash
pip install pandas scikit-learn matplotlib
python linear_regression.py
