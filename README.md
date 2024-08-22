# Weather Type Detection

The WeatherPrediction class provides a framework for training and evaluating a RandomForestClassifier model to predict weather types based on various meteorological features. It fetches data from a specified CSV file, processes it, trains the model, and allows for predictions on new input data.


## Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About

This project will detect wheather the weather is sunny, cloudy, snowy or rainy. This project enhanced my ML skills to implement. Here around the accuracy is near 89% which means my data is not overfitting. But then to if anyone have any issue then all the counters and criticizms are welcome but in the form of ADVICE.

## Features

- Load weather data from a CSV file.
- Preprocess the data by dropping unnecessary columns.
- Train a RandomForestClassifier on the preprocessed data.
- Evaluate the model's performance using accuracy and classification reports.
- Make predictions based on new input features.

## Installation

Provide instructions on how to install and set up your project locally. Include any dependencies that need to be installed and any configuration steps.

### Note

- Advice all to install following in an "env." folder. 

```bash
# Clone the repository
git clone https://github.com/PoojanDoshi11/Weather_Detection

# Navigate to the project directory
cd Weather_Detection

# Install dependencies
pip install -r requirements.txt
```

## Usage
-- Run the Flask app
```bash
python app.py
```
-- then follow the provided local host link
-- then enter the necessary data and please click on "Detect"
-- then enjoy the results
