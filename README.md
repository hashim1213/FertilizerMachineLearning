# Fertilizer Optimization for Canola Using Machine Learning

## Introduction

This project aims to optimize the fertilizer blend for canola crops using historical yield data and soil test data

## Project Structure
```
├── app.py
├── models.py
├── train.py
├── requirements.txt
├── fertilizer_data.csv
├── fertilizer_model.pkl
├── templates
│ ├── index.html
│ ├── add_data.html
└── uploads
```

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:hashim1213/FertilizerMachineLearning.git
   cd FertilizerMachineLearning
   ```
2. Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```
3. Install the required packages:
```
pip install -r requirements.txt
```
4. Train the model:
```
python train.py
```
5. Running the Application
```
python app.py
```
Access the application in your web browser at http://localhost:5000.

# Application Workflow

app.py
- Import Libraries: Import necessary libraries including Flask, SQLAlchemy, Pandas, Joblib, and others.
- Initialize Flask App: Configure the app with the database and upload folder.
- Load Model: Load the pre-trained model if available.
Routes:
/: Main route for the fertilizer optimization form.
/add_data: Route for adding new training data and retraining the model.

train.py
- Load Dataset: Load data from fertilizer_data.csv.
- Prepare Features and Target: Select features (X) and target variables (y).
- Train-Test Split: Split the data into training and testing sets.
- Train Model: Train a RandomForestRegressor model.
- Save Model: Save the trained model as fertilizer_model.pkl.
  
# How It Works
- Nutrient Removal: Based on previous yield, calculate the nutrients removed from the soil.
- Nutrient Uptake: Based on yield goal, calculate the nutrients required for optimal growth.
- Soil Organic Matter (OM): Adjust nitrogen requirement based on soil organic matter.
- Soil Test Data: Use soil test data to adjust the required fertilizer blend.
  
Adding Training Data
- Form Inputs: Collect data on previous yield, yield goal, soil nutrients, location, and fertilizer amounts.
- Save Data: Save the new data to fertilizer_data.csv.
- Retrain Model: Retrain the RandomForestRegressor model with the updated data and save the model.
  
# Usage
1. Navigate to the home page to input the crop and soil data.
2. Click "Calculate" to get the optimal fertilizer blend.
3. To add new training data, navigate to the "Add Training Data" page.
   
# Dependencies
Flask
Flask-SQLAlchemy
Pandas
Joblib
Scikit-learn
Requests
Install all dependencies using:
```
pip install -r requirements.txt
```
