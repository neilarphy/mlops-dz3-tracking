from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np
import mlflow

from config import config
from data import get_data

mlflow.set_experiment("mlops-tracking-itmo")
mlflow.log_param("logreg_C", config["logistic_regression"]["C"])
mlflow.log_param("logreg_max_iter", config["logistic_regression"]["max_iter"])


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)

    coeffs = model.coef_
    np.save("logreg_coeffs.npy", coeffs)
    mlflow.log_artifact("logreg_coeffs.npy", artifact_path="artifacts")


def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    # Здесь необходимо получить метрики и логировать их в трекер
    print(accuracy_score(y_true=y_test, y_pred=y_pred))
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    roc = roc_auc_score(y_test, model.predict_proba(x_test), multi_class="ovr")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("conf_matrix_logreg.png")
    plt.close()
    mlflow.log_artifact("conf_matrix_logreg.png", artifact_path="artifacts")

if __name__ == "__main__":
    logistic_regression_model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
        C=config["logistic_regression"]["C"]
    )

    data = get_data()
    train(logistic_regression_model, data["x_train"], data["y_train"])
    test(logistic_regression_model, data["x_test"], data["y_test"])
