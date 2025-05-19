## 🧪 ClearML experiment: Logistic Regression

🔗 [Открыть эксперимент в ClearML](https://app.clear.ml/projects/c48a81df5dba48cb8a2c9c58db0b12e5/experiments/6b9927a291544e9794a898a4fdee62c3/output/execution)

### 📋 Описание эксперимента:

**Модель:** `LogisticRegression`

**Параметры из config:**
- `max_iter`: максимальное число итераций обучения
- `C`: коэффициент регуляризации

**Логируются метрики:**
- Accuracy (точность)
- F1 Score (макро)
- ROC AUC (мультиклассовая)
- Confusion matrix (как артефакт `.png`)

**Логируются параметры модели:**
- Коэффициенты регрессии (`coef_`) — доступны в:
  - текстовом виде через консоль
  - бинарном виде как `.npy`-артефакт (`logreg_coeffs.npy`)

**Артефакты:**
- Матрица ошибок (`conf_matrix_logreg.png`)
- Коэффициенты регрессии (`logreg_coeffs.npy`)

## 🧪 ClearML experiment: Decision Tree

🔗 [Открыть эксперимент в ClearML](https://app.clear.ml/projects/c48a81df5dba48cb8a2c9c58db0b12e5/experiments/714ddbacb15f403fbfbc8bd3b903433b/output/execution)

### 📋 Описание эксперимента:

**Модель:** `DecisionTreeClassifier`

**Параметры из config:**
- `max_depth`: максимальная глубина дерева
- `criterion`: критерий разделения (gini)

**Логируются метрики:**
- Accuracy (точность)
- F1 Score (макро)
- ROC AUC (мультиклассовая)
- Confusion matrix (как артефакт `.png`)
- Количество листьев (`get_n_leaves()`)

**Артефакты:**
- Матрица ошибок (`conf_matrix_tree.png`)

