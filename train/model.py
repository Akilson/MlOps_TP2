import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/houses.csv")
X = df[["size", "nb_rooms", "garden"]]
y = df["price"]

# Define model with hyperparameters
params = {
    "n_estimators":300,     # number of trees
    "max_depth":20,         # limit depth to avoid overfitting
    "min_samples_split":4,  # minimum samples required to split a node
    "random_state":42
}
model = RandomForestRegressor(**params)

# Train the model
model.fit(X, y)

# Set our tracking server uri for logging
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_tracking_uri(uri="http://mlflow-service:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Infer the model signature
    signature = infer_signature(X, model.predict(X))

    # Log the model, which inherits the parameters and metric
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="house",
        signature=signature,
        input_example=X,
        registered_model_name="tracking-quickstart",
    )

    # Set a tag that we can use to remind ourselves what this model was for
    mlflow.set_logged_model_tags(
        model_info.model_id, {"Training Info": "Basic LR model for iris data"}
    )

results = model.predict([[150, 2, 0]])

print(results)
