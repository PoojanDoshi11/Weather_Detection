# Speech Type Detection

This project utilizes a machine learning model to detect hate speech and offensive language in social media posts. Implemented using Python and Flask, it provides a web interface for users to input text and receive predictions. The model classifies text into three categories: "Hate Speech," "Offensive Language," and "No Hate or Offensive Language," achieving an accuracy of 87.3%.

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
