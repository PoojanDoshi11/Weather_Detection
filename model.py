# # #!/usr/bin/env python
# # # coding: utf-8

# # # In[1]:


# # import pandas as pd
# # import matplotlib.pyplot as pyp


# # # In[2]:


# # df = pd.read_csv("weather.csv")


# # # In[3]:


# # df.head()


# # # In[4]:


# # col = df.columns


# # # In[5]:


# # y = df['Weather Type'] 
# # y
# # df.drop(columns=['Weather Type'], inplace=True)
# # df


# # # In[6]:


# # object_columns = df.select_dtypes(include=['object']).columns.tolist()
# # df.drop(columns=object_columns, inplace=True)


# # # In[7]:


# # df


# # # In[8]:


# # from sklearn.preprocessing import MinMaxScaler

# # # Initialize the MinMaxScaler
# # scaler = MinMaxScaler()

# # # Fit and transform the features
# # df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# # # In[9]:


# # df_normalized


# # # In[11]:


# # from sklearn.model_selection import train_test_split

# # # Split the data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)


# # # In[12]:


# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import classification_report, accuracy_score

# # # Initialize the Random Forest classifier
# # rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# # # Train the model
# # rf_model.fit(X_train, y_train)

# # # Make predictions
# # y_pred = rf_model.predict(X_test)

# # # Calculate accuracy
# # accuracy = accuracy_score(y_test, y_pred)
# # print(f'Accuracy: {accuracy:.2f}')

# # # Print classification report
# # report = classification_report(y_test, y_pred)
# # print('Classification Report:')
# # print(report)


# # # In[ ]:




# #!/usr/bin/env python
# # coding: utf-8
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import joblib  # Add this import to use joblib for saving/loading

# def load_data(file_path):
#     """Load data from a CSV file."""
#     return pd.read_csv(file_path)

# def preprocess_data(df):
#     """Preprocess the DataFrame by handling missing values and scaling features."""
#     # Drop non-numeric columns
#     object_columns = df.select_dtypes(include=['object']).columns.tolist()
#     df.drop(columns=object_columns, inplace=True)
    
#     # Scale features
#     scaler = MinMaxScaler()
#     df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
#     # Save the scaler
#     joblib.dump(scaler, 'scaler.joblib')
    
#     return df_normalized

# def split_data(df, target_column):
#     """Split the DataFrame into features and target, then into training and testing sets."""
#     y = df[target_column]
#     X = df.drop(columns=[target_column])
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     return X_train, X_test, y_train, y_test

# def train_model(X_train, y_train):
#     """Train the Random Forest classifier model."""
#     rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf_model.fit(X_train, y_train)
    
#     # Save the trained model
#     joblib.dump(rf_model, 'rf_model.joblib')
    
#     return rf_model

# def evaluate_model(model, X_test, y_test):
#     """Evaluate the model and print accuracy and classification report."""
#     y_pred = model.predict(X_test)
    
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f'Accuracy: {accuracy:.2f}')
    
#     report = classification_report(y_test, y_pred)
#     print('Classification Report:')
#     print(report)

# def main():
#     # File path
#     file_path = "weather.csv"
    
#     # Load data
#     df = load_data(file_path)
    
#     # Preprocess data
#     df_normalized = preprocess_data(df)
    
#     # Split data
#     target_column = 'Weather Type'
#     X_train, X_test, y_train, y_test = split_data(df, target_column)
    
#     # Train model
#     model = train_model(X_train, y_train)
    
#     # Evaluate model
#     evaluate_model(model, X_test, y_test)

# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

class WeatherPrediction:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestClassifier()
        self.data = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        dataset = pd.read_csv(self.data_path)
        # Drop columns if they exist
        columns_to_drop = ['Cloud Cover', 'Location', 'Season']
        for column in columns_to_drop:
            if column in dataset.columns:
                dataset.drop(columns=[column], inplace=True)
        self.data = dataset
    
    def preprocess_data(self):
        Y = np.array(self.data['Weather Type'])
        X = np.array(self.data.drop(columns=['Weather Type']))
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.x_test)
        cr = classification_report(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        print(cr)

    def predict(self, temperature, humidity, wind_speed, precipitation, atmospheric_pressure, uv_index, visibility):
        input_data = np.array([[temperature, humidity, wind_speed, precipitation, atmospheric_pressure, uv_index, visibility]])
        prediction = self.model.predict(input_data)
        return prediction[0]
    
    def run(self):
        self.load_data()
        self.preprocess_data()
        self.train_model()
        self.evaluate_model()

# # Usage example
# weather_model = WeatherPrediction('weather.csv')
# weather_model.load_data()
# weather_model.preprocess_data()
# weather_model.train_model()
# weather_model.evaluate_model()

# # Predicting weather type for a new set of values
# predicted_weather = weather_model.predict(3.0,85,6.0,96.0,984.46,1,3.5)
# print(f'Predicted Weather Type: {predicted_weather}')

# 14.0,73,9.5,82.0,partly cloudy,1010.82,2,Winter,3.5,inland,Rainy