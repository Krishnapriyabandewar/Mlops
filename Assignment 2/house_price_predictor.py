# house_price_predictor.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class HousePricePredictor:
    def __init__(self):
        # Load dataset
        self.data = pd.read_csv('train.csv')

        # Select features and target
        self.features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
        self.target = 'SalePrice'
        
        # Prepare the data
        X = self.data[self.features]
        y = self.data[self.target]

        # Preprocess: scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def predict(self, input_data):
        # Preprocess the input data
        input_data_scaled = self.scaler.transform([input_data])
        
        # Make prediction
        predicted_price = self.model.predict(input_data_scaled)
        return predicted_price[0]
