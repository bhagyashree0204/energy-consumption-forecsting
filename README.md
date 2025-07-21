# energy-consumption-forecsting
python
CopyEdit
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('energy.csv', parse_dates=['Datetime'], index_col='Datetime')
data = data.resample('H').mean()
data = data.fillna(method='ffill')

data['Energy'].plot(figsize=(15,5), title="Energy Consumption Over Time")
python
CopyEdit
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

X = data[['hour', 'day']]
y = data['Energy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")
bash
CopyEdit
pip install flask joblib
Code (Save in app.py):
python
CopyEdit
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('energy_forecast_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['hour'], data['day']]])
    prediction = model.predict(features)
    return jsonify({'predicted_energy': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
