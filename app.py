<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Vegetable Price Prediction</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
  
  <!-- Embedded CSS for styling -->
  <style>
    /* Basic Reset */
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    /* Body Styling */
    body {
      font-family: Arial, sans-serif;
      background: url("{{ url_for('static', filename='images/Background.jpg') }}") no-repeat center center fixed;
      background-size: cover;
      color: #333;
    }

    /* Container */
    .container {
      max-width: 600px;
      margin: 50px auto;
      padding: 30px;
      background-color: rgba(255, 255, 255, 0.9);
      border-radius: 12px;
      box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
      border-top: 4px solid #4caf50;
    }

    /* Title */
    h1 {
      color: #00695c;
      font-size: 2.2em;
      text-align: center;
      margin-bottom: 30px;
      font-weight: bold;
    }

    /* Form Styling */
    .form-label {
      font-weight: bold;
      color: #555;
    }

    .form-control {
      border: 2px solid #cfd8dc;
      border-radius: 8px;
      padding: 12px;
      font-size: 1em;
      transition: border-color 0.3s;
    }

    .form-control:focus {
      border-color: #4caf50;
      box-shadow: 0 0 5px rgba(76, 175, 80, 0.3);
    }

    /* Button Styling */
    button[type="submit"] {
      width: 100%;
      padding: 12px;
      font-size: 1.1em;
      background-color: #4caf50;
      color: #ffffff;
      border: none;
      border-radius: 8px;
      font-weight: bold;
      cursor: pointer;
      transition: background-color 0.3s, transform 0.1s;
    }

    button[type="submit"]:hover {
      background-color: #388e3c;
      transform: scale(1.02);
    }

    button[type="submit"]:active {
      background-color: #2e7d32;
    }

    /* Result Styling */
    .result {
      margin-top: 20px;
      padding: 20px;
      background-color: #e8f5e9;
      color: #1b5e20;
      border: 1px solid #c8e6c9;
      border-radius: 8px;
      text-align: center;
      font-size: 1.2em;
      font-weight: bold;
    }

    /* Image Styling */
    .decorative-images {
      margin-top: 20px;
      text-align: center;
    }

    .decorative-images img {
      margin: 10px;
      width: 120px;
      height: auto;
      border-radius: 8px;
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
  </style>
</head>
<body>
  <div class="container">
    <h1 class="text-center mt-4">Vegetable Price Prediction</h1>
    
    <!-- User Input Form -->
    <form id="vegetableForm" method="POST" action="/predict">
      <!-- Dropdown for Vegetable Selection -->
      <div class="mb-3">
        <label for="vegetable" class="form-label">Select Vegetable</label>
        <select class="form-control" id="vegetable" name="vegetable" required>
          <option value="" disabled selected>Choose a vegetable...</option>
          <option value="Tomato">Tomato</option>
          <option value="Potato">Potato</option>
          <option value="Onion">Onion</option>
          <option value="Cabbage">Cabbage</option>
          <option value="Carrot">Carrot</option>
        </select>
      </div>
      <div class="mb-3">
        <label for="area" class="form-label">Area (in hectares)</label>
        <input type="number" class="form-control" id="area" name="area" required>
      </div>
      <div class="mb-3">
        <label for="production" class="form-label">Production (in tons)</label>
        <input type="number" class="form-control" id="production" name="production" required>
      </div>
      <div class="mb-3">
        <label for="gdp" class="form-label">GDP</label>
        <input type="number" class="form-control" id="gdp" name="gdp" required>
      </div>
      <div class="mb-3">
        <label for="growthRate" class="form-label">Annual Growth Rate (%)</label>
        <input type="number" class="form-control" id="growthRate" name="growthRate" step="0.1" required>
      </div>
      <div class="mb-3">
        <label for="inflation" class="form-label">Inflation Rate (%)</label>
        <input type="number" class="form-control" id="inflation" name="inflation" step="0.1" required>
      </div>
      <div class="mb-3">
        <label for="rainfall" class="form-label">Rainfall (mm)</label>
        <input type="number" class="form-control" id="rainfall" name="rainfall" required>
      </div>
      <div class="mb-3">
        <label for="temperature" class="form-label">Temperature (°C)</label>
        <input type="number" class="form-control" id="temperature" name="temperature" required>
      </div>

      <!-- Submit Button -->
      <button type="submit" class="btn btn-success">Predict Price</button>
    </form>

    <!-- Prediction Result -->
    {% if prediction_text %}
      <div class="result">
        <p>{{ prediction_text }}</p>
      </div>
    {% endif %}

    <!-- Optional Decorative Images -->
    <div class="decorative-images">
      <img src="{{ url_for('static', filename='images/vegetable1.png') }}" alt="Vegetable 1">
      <img src="{{ url_for('static', filename='images/vegetable2.webp') }}" alt="Vegetable 2">
    </div>
  </div>

  <!-- Bootstrap JS & jQuery -->
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
