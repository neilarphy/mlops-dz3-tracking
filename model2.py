from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import mlflow

from config import config
from data import get_data

mlflow.set_experiment("mlops-tracking-itmo")

def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)
    n_leaves = model.get_n_leaves()
    mlflow.log_metric("num_leaves", n_leaves)

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
    plt.savefig("conf_matrix_tree.png")
    plt.close()

    mlflow.log_artifact("conf_matrix_tree.png", artifact_path="artifacts")

if __name__ == "__main__":
    with mlflow.start_run(run_name="Decision Tree"):
        mlflow.log_param("random_state", config["random_state"])

        for key, value in config["decision_tree"].items():
            mlflow.log_param(f'tree_{key}', value)

        decision_tree_model = DecisionTreeClassifier(
            random_state=config["random_state"],
            max_depth=config["decision_tree"]["max_depth"],
            criterion=config["decision_tree"]["criterion"]
        )

        data = get_data()
        train(decision_tree_model, data["x_train"], data["y_train"])
        test(decision_tree_model, data["x_test"], data["y_test"])
