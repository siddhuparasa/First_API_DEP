from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "Hi siddhu your first  deployment is successful"}

@app.post("/predict")
def predict(features: list):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
