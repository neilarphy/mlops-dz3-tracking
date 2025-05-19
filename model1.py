from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np
import wandb

from config import config
from data import get_data

wandb.init(
    project="mlops-tracking-itmo",
    name="Logistic Regression"
)

wandb.config.update(config)

def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)

    coeffs = model.coef_
    np.save("logreg_coeffs.npy", coeffs)

    coeff_artifact = wandb.Artifact("logreg_coeffs", type="model")
    coeff_artifact.add_file("logreg_coeffs.npy")
    wandb.log_artifact(coeff_artifact)

def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    # Здесь необходимо получить метрики и логировать их в трекер
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    roc = roc_auc_score(y_test, model.predict_proba(x_test), multi_class="ovr")
    print(accuracy_score(y_true=y_test, y_pred=y_pred))

    wandb.log({
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": roc
    }, step=0)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("conf_matrix_logreg.png")
    plt.close()

    cm_artifact = wandb.Artifact("conf_matrix_logreg", type="image")
    cm_artifact.add_file("conf_matrix_logreg.png")
    wandb.log_artifact(cm_artifact)

if __name__ == "__main__":
    logistic_regression_model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
        C=config["logistic_regression"]["C"]
    )

    data = get_data()
    train(logistic_regression_model, data["x_train"], data["y_train"])
    test(logistic_regression_model, data["x_test"], data["y_test"])
