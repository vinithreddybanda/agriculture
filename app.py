from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

app = Flask(__name__)

# Define column names for the input features
column_names = ['Vegetable', 'Area (hectares)', 'Production (tons)', 'GDP (₹)', 
                'Annual Growth Rate (%)', 'Inflation Rate (%)', 
                'Rainfall (mm)', 'Temperature (°C)']

# Load data from a CSV file (Ensure your dataset is in the correct path)
data = pd.read_csv('./vegetable_prices_dataset.csv')

# Separate features and target
X_train = data[column_names]
y_train = data['Target Price (₹)']

# Define preprocessing for numerical and categorical columns
numeric_features = ['Area (hectares)', 'Production (tons)', 'GDP (₹)', 
                    'Annual Growth Rate (%)', 'Inflation Rate (%)', 
                    'Rainfall (mm)', 'Temperature (°C)']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['Vegetable']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the final pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    input_data = [x for x in request.form.values()]
    
    # Convert inputs to appropriate types
    try:
        area = float(input_data[1])
        production = float(input_data[2])
        gdp = float(input_data[3])
        growth_rate = float(input_data[4])
        inflation = float(input_data[5])
        rainfall = float(input_data[6])
        temperature = float(input_data[7])
    except ValueError:
        return render_template('index.html', prediction_text="Error: Invalid input. Please ensure all inputs are numbers.")
    
    # Validate input: Reject values less than or equal to zero
    error_message = []
    if area <= 0:
        error_message.append("Area")
    if production <= 0:
        error_message.append("Production")
    if gdp <= 0:
        error_message.append("GDP")
    if rainfall <= 0:
        error_message.append("Rainfall")
    if temperature <= 0:
        error_message.append("Temperature")
    
    if error_message:
        error_columns = ", ".join(error_message)
        return render_template('index.html', prediction_text=f"Error: {error_columns} must be positive values.")
    
    input_df = pd.DataFrame([input_data], columns=column_names)
    
    # Preprocess and make prediction
    prediction = pipeline.predict(input_df)
    output = round(prediction[0] * 1000, 2)  # Scale the predicted price appropriately

    return render_template('index.html', prediction_text=f'Predicted Vegetable Price: ₹{output}')

if __name__ == "__main__":
    app.run(debug=True)
