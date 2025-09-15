from fastapi import FastAPI
import mlflow.sklearn
from pydantic import BaseModel

app =  FastAPI()

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
model_name = "tracking-quickstart"
model_version = "latest"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

class Param(BaseModel):
    size: float
    rooms: int
    is_garden: int

class ParamModel(BaseModel):
	name: str
	version: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(features: Param):
    size = features.size
    nb_rooms = features.rooms
    is_garden = features.is_garden
    results = model.predict([[size, nb_rooms, is_garden]])
    return {"y_pred": results[0]}

@app.post("/update-model")
def update(features: ParamModel):
	name = features.name
	version = features.version
	model_uri = f"models:/{name}/{version}"
	model = mlflow.sklearn.load_model(model_uri)
