import pickle
import mlflow
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

def hyperparameter_tuning():
    with open("mlflow/data/X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open("mlflow/data/y_train.pkl", "rb") as f:
        y_train = pickle.load(f)

    # Định nghĩa grid các bộ tham số cần thử
    param_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }

    grid = ParameterGrid(param_grid)

    best_score = 0
    best_params = {}
    best_model = None

    mlflow.set_tracking_uri("http://mlflow:600")
    mlflow.set_experiment("FakeNews_KNN")

    for params in grid:
        run_name = f"KNN_k={params['n_neighbors']}_w={params['weights']}_p={params['p']}"
        with mlflow.start_run(run_name = run_name):
            model = KNeighborsClassifier(**params)
            model.fit(X_train, y_train)

            acc = model.score(X_train, y_train)
            mlflow.log_params(params)
            mlflow.log_metric("train_accuracy", acc)

            if acc > best_score:
                best_score = acc
                best_params = params
                best_model = model

    print("Best params:", best_params)
    with open("mlflow/models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
