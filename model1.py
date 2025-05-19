from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

from config import config
from data import get_data

from clearml import Task

task = Task.init(
    project_name="ML Tracking ITMO",
    task_name="Logistic Regression")
task.connect(config)

def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)
    
    coeffs = model.coef_
    task.get_logger().report_text(f"Coefficients: {coeffs.tolist()}")
    np.save("logreg_coeffs.npy", coeffs)
    task.upload_artifact("logreg_coeffs", "logreg_coeffs.npy")

def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    # Здесь необходимо получить метрики и логировать их в трекер
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    roc = roc_auc_score(y_test, model.predict_proba(x_test), multi_class="ovr")

    task.get_logger().report_scalar("Accuracy", "Validation", acc, iteration=0)
    task.get_logger().report_scalar("F1 Score", "Validation", f1, iteration=0)
    task.get_logger().report_scalar("ROC AUC", "Validation", roc, iteration=0)
    print(accuracy_score(y_true=y_test, y_pred=y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("conf_matrix_logreg.png")
    task.upload_artifact("conf_matrix_logreg", "conf_matrix_logreg.png")
    plt.close()


if __name__ == "__main__":
    logistic_regression_model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
        C=config["logistic_regression"]["C"]
    )

    data = get_data()
    train(logistic_regression_model, data["x_train"], data["y_train"])
    test(logistic_regression_model, data["x_test"], data["y_test"])
