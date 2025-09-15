FROM python:3.12-slim
WORKDIR /app
COPY ./train /app/train
RUN pip install mlflow scikit-learn pandas
CMD ["python", "/app/train/model.py"]

