import pickle
import mlflow
from sklearn.metrics import accuracy_score, classification_report

def evaluate():
    with open("mlflow/models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("mlflow/data/X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open("mlflow/data/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    mlflow.set_tracking_uri("http://mlflow:600")
    mlflow.set_experiment("FakeNews_KNN")
    best_model_path = "mlflow/models/best_model.pkl"
    with mlflow.start_run(run_name = 'evaluate_best_model'):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_artifact(best_model_path)
        report = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metric(f"{label}_precision", metrics["precision"])
                mlflow.log_metric(f"{label}_recall", metrics["recall"])
