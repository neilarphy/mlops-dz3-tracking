from sklearn.tree import DecisionTreeClassifier
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
    name="Decision Tree"
)

wandb.config.update(config)

def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)

    n_leaves = model.get_n_leaves()
    wandb.log({"num_leaves": n_leaves}, step=0)

def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    # Здесь необходимо получить метрики и логировать их в трекер
    print(accuracy_score(y_true=y_test, y_pred=y_pred))
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    roc = roc_auc_score(y_test, model.predict_proba(x_test), multi_class="ovr")

    wandb.log({
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": roc
    }, step=0)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("conf_matrix_tree.png")
    plt.close()

    cm_artifact = wandb.Artifact("conf_matrix_tree", type="image")
    cm_artifact.add_file("conf_matrix_tree.png")
    wandb.log_artifact(cm_artifact)


if __name__ == "__main__":
    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"],
        criterion=config["decision_tree"]["criterion"]
    )

    data = get_data()
    train(decision_tree_model, data["x_train"], data["y_train"])
    test(decision_tree_model, data["x_test"], data["y_test"])
