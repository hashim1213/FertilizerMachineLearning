import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
data = pd.read_csv('fertilizer_data.csv')

# Features and target
X = data[['previous_yield', 'yield_goal', 'OM', 'soil_N', 'soil_P2O5', 'soil_K2O', 'soil_S', 'location']]
y = data[['Ammonium_nitrate', 'Ammonium_sulfate', 'Anhydrous_ammonia', 'Urea', 'Diammonium_phosphate', 'MAP', 'Potassium_chloride', 'Potassium_sulfate', 'Gypsum']]

# One-hot encode the 'location' column
X = pd.get_dummies(X, columns=['location'], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'fertilizer_model.pkl')

# Evaluate the model
print("Model score:", model.score(X_test, y_test))
