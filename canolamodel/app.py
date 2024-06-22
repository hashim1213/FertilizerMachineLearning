from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import pandas as pd
import joblib
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from models import db, Fertilizer

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fertilizers.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
db.init_app(app)

with app.app_context():
    db.create_all()

# Load the trained model
if os.path.exists('fertilizer_model.pkl'):
    model = joblib.load('fertilizer_model.pkl')
else:
    model = None

# Nutrient removal rates per bushel
removal_rates_per_bushel = {
    'N': 1.87,
    'P2O5': 0.78,
    'K2O': 0.38,
    'S': 0.22
}

# Calculate nutrient removal based on previous yield
def calculate_nutrient_removal(previous_yield):
    return {nutrient: previous_yield * rate for nutrient, rate in removal_rates_per_bushel.items()}

# Calculate nutrient uptake based on yield goal
def calculate_nutrient_uptake(yield_goal):
    uptake_per_bushel = {
        'N': 2.38,
        'P2O5': 0.90,
        'K2O': 2.93,
        'S': 0.86
    }
    return {nutrient: yield_goal * rate for nutrient, rate in uptake_per_bushel.items()}

# Calculate the required fertilizers
def integrated_calculation(previous_yield, yield_goal, OM):
    soil_OM_contribution_N = OM * 14 * 0.8
    removal_rates = calculate_nutrient_removal(previous_yield)
    uptake_rates = calculate_nutrient_uptake(yield_goal)
    uptake_rates['N'] += soil_OM_contribution_N
    return uptake_rates

# Function to get weather data
def get_weather_data(lat, lon):
    api_key = "9b6e507a7d41ce2286eb3741f686f05a"
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    return response.json()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            previous_yield = float(request.form['previous_yield'])
            yield_goal = float(request.form['yield_goal'])
            OM = float(request.form['OM'])
            use_soil_data = request.form.get('use_soil_data') == 'yes'
            location = request.form['location']
            
            soil_N = soil_P2O5 = soil_K2O = soil_S = 0
            if use_soil_data:
                soil_N = float(request.form['soil_N'])
                soil_P2O5 = float(request.form['soil_P2O5'])
                soil_K2O = float(request.form['soil_K2O'])
                soil_S = float(request.form['soil_S'])

            # Prepare the input for the model
            input_features = pd.DataFrame([[previous_yield, yield_goal, OM, soil_N, soil_P2O5, soil_K2O, soil_S, location]], 
                                           columns=['previous_yield', 'yield_goal', 'OM', 'soil_N', 'soil_P2O5', 'soil_K2O', 'soil_S', 'location'])
            input_features = pd.get_dummies(input_features, columns=['location'])

            # Align input features with the training features
            model_features = model.feature_names_in_
            input_features = input_features.reindex(columns=model_features, fill_value=0)

            print("Input features for model:", input_features)

            # Predict fertilizer blend
            if model:
                predictions = model.predict(input_features)
                predictions = predictions[0]  # Get the first prediction
                print("Predictions:", predictions)

                # Map predictions to fertilizers
                fertilizers = ['Ammonium_nitrate', 'Ammonium_sulfate', 'Anhydrous_ammonia', 'Urea', 'Diammonium_phosphate', 'MAP', 'Potassium_chloride', 'Potassium_sulfate', 'Gypsum']
                results = {fertilizer: round(amount, 2) for fertilizer, amount in zip(fertilizers, predictions)}

                # Calculate nutrient requirements
                if use_soil_data:
                    removal_rates = calculate_nutrient_removal(previous_yield)
                    required_fertilizer = {nutrient: max(0, removal_rates[nutrient] - soil_N - soil_P2O5 - soil_K2O - soil_S) for nutrient in removal_rates}
                else:
                    required_fertilizer = integrated_calculation(previous_yield, yield_goal, OM)

                print("Results:", results)
                print("Required fertilizer:", required_fertilizer)

                return render_template('index.html', results=results, total_cost=None, rates=None, required_fertilizer=required_fertilizer, use_soil_data=use_soil_data)
            else:
                return render_template('index.html', error="Model not trained. Please upload training data to train the model.")
        except Exception as e:
            error_message = str(e)
            print("Error:", error_message)
            return render_template('index.html', error=error_message)

    return render_template('index.html')

@app.route('/add_data', methods=['GET', 'POST'])
def add_data():
    data = []
    if os.path.exists('fertilizer_data.csv'):
        df = pd.read_csv('fertilizer_data.csv')
        data = df.to_dict(orient='records')
    
    if request.method == 'POST':
        try:
            # Collect form data
            new_data = {
                'previous_yield': float(request.form['previous_yield']),
                'yield_goal': float(request.form['yield_goal']),
                'OM': float(request.form['OM']),
                'soil_N': float(request.form['soil_N']),
                'soil_P2O5': float(request.form['soil_P2O5']),
                'soil_K2O': float(request.form['soil_K2O']),
                'soil_S': float(request.form['soil_S']),
                'location': request.form['location'],
                'Ammonium_nitrate': float(request.form['Ammonium_nitrate']),
                'Ammonium_sulfate': float(request.form['Ammonium_sulfate']),
                'Anhydrous_ammonia': float(request.form['Anhydrous_ammonia']),
                'Urea': float(request.form['Urea']),
                'Diammonium_phosphate': float(request.form['Diammonium_phosphate']),
                'MAP': float(request.form['MAP']),
                'Potassium_chloride': float(request.form['Potassium_chloride']),
                'Potassium_sulfate': float(request.form['Potassium_sulfate']),
                'Gypsum': float(request.form['Gypsum']),
            }

            # Load the existing dataset
            if os.path.exists('fertilizer_data.csv'):
                df = pd.read_csv('fertilizer_data.csv')
            else:
                df = pd.DataFrame(columns=new_data.keys())

            # Append new data using pd.concat
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

            # Save the updated dataset
            df.to_csv('fertilizer_data.csv', index=False)

            # Train the model
            X = df[['previous_yield', 'yield_goal', 'OM', 'soil_N', 'soil_P2O5', 'soil_K2O', 'soil_S', 'location']]
            X = pd.get_dummies(X, columns=['location'], drop_first=True)
            y = df[['Ammonium_nitrate', 'Ammonium_sulfate', 'Anhydrous_ammonia', 'Urea', 'Diammonium_phosphate', 'MAP', 'Potassium_chloride', 'Potassium_sulfate', 'Gypsum']]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            new_model = RandomForestRegressor(n_estimators=100, random_state=42)
            new_model.fit(X_train, y_train)
            model_accuracy = new_model.score(X_test, y_test)
            joblib.dump(new_model, 'fertilizer_model.pkl')

            global model
            model = new_model

            data = df.to_dict(orient='records')

            return render_template('add_data.html', message=f"Data added and model retrained successfully. Model accuracy: {model_accuracy:.2f}", data=data)
        except Exception as e:
            return render_template('add_data.html', message=str(e), data=data)

    return render_template('add_data.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
