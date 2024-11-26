from fastapi import FastAPI
from pydantic import BaseModel
from house_price_predictor import HousePricePredictor

app = FastAPI()

# Define a Pydantic model for the input data
class HouseFeatures(BaseModel):
    OverallQual: int
    GrLivArea: int
    GarageCars: int
    TotalBsmtSF: int
    FullBath: int
    YearBuilt: int

predictor = HousePricePredictor()  # Initialize your predictor class

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Predictor API!"}

@app.post("/predict")
async def predict(house_features: HouseFeatures):
    # Convert the input data to a format suitable for prediction
    data = [
        house_features.OverallQual,
        house_features.GrLivArea,
        house_features.GarageCars,
        house_features.TotalBsmtSF,
        house_features.FullBath,
        house_features.YearBuilt
    ]
    prediction = predictor.predict(data)  # Make sure your predictor class has a predict method
    return {"predicted_price": prediction}
