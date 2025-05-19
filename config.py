# Вы можете свободно менять параметры моделей и добавлять новые
config = {
    "random_state": 42,
    "data": {
        "test_size": 0.2,
    },
    "logistic_regression": {
        "max_iter": 200,
        "C": 1.0
    },
    "decision_tree": {
        "max_depth": 10,
        "criterion": "gini"
    }
}
