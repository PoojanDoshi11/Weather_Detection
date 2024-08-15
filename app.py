from flask import Flask, request, render_template
from model import WeatherPrediction

app = Flask(__name__)

WP = WeatherPrediction(data_path='weather.csv')
WP.run()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve and convert form data
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        precipitation = float(request.form['precipitation'])
        atmospheric_pressure = float(request.form['atmospheric_pressure'])
        uv_index = float(request.form['uv_index'])
        visibility = float(request.form['visibility'])
        
        # Make prediction
        prediction = WP.predict(
            temperature, 
            humidity, 
            wind_speed, 
            precipitation, 
            atmospheric_pressure, 
            uv_index, 
            visibility
        )
        
        # Render the result page
        return render_template(
            'result.html',
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            precipitation=precipitation,
            atmospheric_pressure=atmospheric_pressure,
            uv_index=uv_index,
            visibility=visibility,
            result=prediction
        )
    except (KeyError, ValueError) as e:
        # Handle missing or invalid form data
        return f"Error processing your request: {e}", 400

if __name__ == '__main__':
    app.run(debug=False, port=8000)
