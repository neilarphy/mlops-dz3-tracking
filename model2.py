from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from config import config
from data import get_data
from clearml import Task

task = Task.init(
    project_name="ML Tracking ITMO",
    task_name="Decision Tree")

task.connect(config)

def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)

    n_leaves = model.get_n_leaves()
    task.get_logger().report_scalar("Num Leaves", "Tree", n_leaves, iteration=0)


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
    plt.savefig("conf_matrix_tree.png")
    task.upload_artifact("conf_matrix_tree", "conf_matrix_tree.png")
    plt.close()

if __name__ == "__main__":
    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"],
        criterion=config["decision_tree"]["criterion"]
    )

    data = get_data()
    train(decision_tree_model, data["x_train"], data["y_train"])
    test(decision_tree_model, data["x_test"], data["y_test"])
