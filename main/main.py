"""
main.py

This module defines a FastAPI application for Animal Image Classification.
"""

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from model.model import predict

app = FastAPI()

# Mount the static files (CSS) directory
app.mount("/static", StaticFiles(directory="main/static"), name="static")

# Initialize Jinja2 template engine
templates = Jinja2Templates(directory="main/templates")


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """
    Renders the index.html template.
    """
    return templates.TemplateResponse("index.html", {"request": request})


class PredictionResult(BaseModel):
    """
    Model to represent the prediction result.
    """

    prediction: str


@app.post("/predict", response_model=PredictionResult)
async def predict_endpoint(file: UploadFile = File(...)) -> PredictionResult:
    """
    Predicts the class of an uploaded image.
    """
    image_bytes = await file.read()
    prediction = predict(image_bytes)
    return PredictionResult(prediction=prediction)
