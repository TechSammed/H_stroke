# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 22:45:05 2025

@author: samme
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

# --- CORS Middleware Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load the trained ML model ---
model = joblib.load("best_stroke_model.pkl")

@app.get("/")
def home():
    return {"message": "Welcome to the Stroke Risk Prediction API"}

@app.post("/predict", response_class=JSONResponse)
async def predict(
    request: Request,
    feature0: float = Form(...), feature1: float = Form(...),
    feature2: float = Form(...), feature3: float = Form(...),
    feature4: float = Form(...), feature5: float = Form(...),
    feature6: float = Form(...), feature7: float = Form(...),
    feature8: float = Form(...), feature9: float = Form(...),
    feature10: float = Form(...), feature11: float = Form(...),
    feature12: float = Form(...), feature13: float = Form(...),
    feature14: float = Form(...), feature15: float = Form(...)
):
    # Collect the input features into a list
    input_features = [feature0, feature1, feature2, feature3, feature4, feature5,
                      feature6, feature7, feature8, feature9, feature10, feature11,
                      feature12, feature13, feature14, feature15]
    
    # Perform the prediction using the loaded model
    prediction = model.predict(np.array(input_features).reshape(1, -1))[0]

    # Return the prediction result as JSON
    return JSONResponse(content={"prediction": round(prediction, 2)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
