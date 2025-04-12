# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 22:45:05 2025

@author: samme
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  # ✅ NEW: Import CORS middleware

# Initialize FastAPI app
app = FastAPI()

# ✅ Add CORS middleware to allow frontend or external requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],  # "" means any origin. Use your frontend URL here in production.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
# Load the trained ML model
model = joblib.load("best_stroke_model.pkl")

# Set up Jinja2 template directory for rendering HTML
templates = Jinja2Templates(directory="templates")

# Home route to load the index page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route to handle the form input and generate prediction
@app.post("/predict", response_class=HTMLResponse)
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
  # Collect features into an array for prediction
    input_features = [feature0, feature1, feature2, feature3, feature4, feature5,
                      feature6, feature7, feature8, feature9, feature10, feature11,
                      feature12, feature13, feature14, feature15]
    
    # Make the prediction
    prediction = model.predict(np.array(input_features).reshape(1, -1))[0]
    
    # Return prediction result back to the HTML page
    return templates.TemplateResponse("index.html", {
    "request": request,
    "prediction": f"Predicted Stroke Risk: {round(prediction, 2)}%"
})

