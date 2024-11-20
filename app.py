from flask import Flask, request, render_template
import numpy as np
import pandas as pd  # type: ignore
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Define column names for the input features
column_names = ['Area', 'Production', 'GDP', 'Annual Growth Rate', 'Inflation', 'Rainfall', 'Temperature']

# Load the dataset from a CSV file
data = pd.read_csv('crop_data.csv')

# Separate features (X) and target (y)
X = data[column_names]
y = data['Crop Price']

# Initialize and fit the pipeline with training data
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Fit the pipeline on the training data
X_scaled = my_pipeline.fit_transform(X)

# Train the Random Forest model
model = RandomForestRegressor()
model.fit(X_scaled, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form as a list of float values
    input_features = [float(x) for x in request.form.values()]
    
    # Convert input features to DataFrame to ensure correct column names
    input_df = pd.DataFrame([input_features], columns=column_names)
    
    # Preprocess the input data with the fitted pipeline
    input_prepared = my_pipeline.transform(input_df)
    
    # Make the prediction
    prediction = model.predict(input_prepared)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Predicted Crop Price: ₹{output}')

if __name__ == "__main__":
    app.run(debug=True)
