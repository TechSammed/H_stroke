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

app = FastAPI()

# Load the trained model
model = joblib.load("best_stroke_model.pkl")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
    input_features = [feature0, feature1, feature2, feature3, feature4, feature5,
                      feature6, feature7, feature8, feature9, feature10, feature11,
                      feature12, feature13, feature14, feature15]
    
    prediction = model.predict(np.array(input_features).reshape(1, -1))[0]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": f"Predicted Stroke Risk: {round(prediction, 2)}%"
    })
