# MLflow tracking

Training runs are logged to MLflow. By default, the trainer uses a local SQLite backend (mlflow.db) and local artifact directory. To view runs:

    mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

Open http://127.0.0.1:5000. To use a remote or containerized tracking server, set MLFLOW_TRACKING_URI before running the training script.
