"""
main.py

This module defines a FastAPI application for Animal Image Classification.
"""

from urllib.parse import quote
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
from model.model import predict

app = FastAPI()

# Mount the static files (CSS) directory
app.mount("/main/static", StaticFiles(directory="main/static"), name="static")

# Initialize Jinja2 template engine
templates = Jinja2Templates(directory="main/templates")


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """
    Renders the index.html template.
    """
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/model", response_class=HTMLResponse)
async def read_home(request: Request):
    """
    Renders the home.html template.
    """
    return templates.TemplateResponse("model.html", {"request": request})


@app.get("/features", response_class=HTMLResponse)
async def read_model_features(request: Request):
    """
    Renders the modelfeatures.html template.
    """
    return templates.TemplateResponse("features.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def read_about(request: Request):
    """
    Renders the about.html template.
    """
    return templates.TemplateResponse("about.html", {"request": request})


class PredictionResponse(BaseModel):
    """
    Model to represent the prediction result.
    """

    prediction: str
    wikipedia_url: str
    wikipedia_description: str


def fetch_wikipedia_description(animal_name: str) -> str:
    """
    Fetches the first paragraph of the Wikipedia page for a given animal name.
    """
    try:
        url=f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(animal_name.replace(' ', '_'))}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("extract", "No description available.")
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500, detail="Error fetching Wikipedia data"
        ) from e


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Predicts the class of an uploaded image, fetches a description from Wikipedia
    """
    image_bytes = await file.read()
    prediction = predict(image_bytes)
    wikipedia_url = (
        f"https://en.wikipedia.org/wiki/{quote(prediction.replace(' ', '_'))}"
    )
    wikipedia_description = fetch_wikipedia_description(prediction)

    return PredictionResponse(
        prediction=prediction,
        wikipedia_url=wikipedia_url,
        wikipedia_description=wikipedia_description,
    )
