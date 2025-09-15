import mlflow.sklearn

model_name = "tracking-quickstart"
model_version = "latest"

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Load the model from the Model Registry
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

results = model.predict([[150, 2, 0]])

print(results)
