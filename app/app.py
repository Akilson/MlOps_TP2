from fastapi import FastAPI
import mlflow.sklearn
from pydantic import BaseModel
import time
import requests
import random

app =  FastAPI()

#mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
#mlflow.set_tracking_uri("http://host.docker.internal:8080")
mlflow.set_tracking_uri(uri="http://mlflow-service:8080")
model_name = "tracking-quickstart"
model_version = "latest"
model_uri = f"models:/{model_name}/{model_version}"
mlflow_uri = "http://mlflow-service:8080"
for _ in range(20):  # max 20 tries
    try:
        r = requests.get(f"{mlflow_uri}/api/2.0/mlflow/registered-models/list")
        if r.status_code == 200:
            break
    except requests.exceptions.ConnectionError:
        pass
    print("Waiting for MLflow server...")
    time.sleep(2)

current_model = mlflow.sklearn.load_model(model_uri)
next_model = mlflow.sklearn.load_model(model_uri)
p = random.uniform(0, 1)

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
    if random.uniform(0, 1) < p:
        results = current_model.predict([[size, nb_rooms, is_garden]])
    else:
        results = next_model.predict([[size, nb_rooms, is_garden]])
    return {"y_pred": results[0]}

@app.post("/update-model")
def update(features: ParamModel):
	name = features.name
	version = features.version
	model_uri = f"models:/{name}/{version}"
	next_model = mlflow.sklearn.load_model(model_uri)

@app.get("/accept-next-model")
def accept_next_mode():
	current_model = next_model
	return {"message": "Next model promoted as current"}
