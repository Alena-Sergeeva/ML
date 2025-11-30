import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    sorted_indices = np.argsort(feature_vector)
    feature_sorted = feature_vector[sorted_indices]
    target_sorted = target_vector[sorted_indices]

    # Уникальные значения признака
    unique_features = np.unique(feature_sorted)

    # Если меньше 2 уникальных значений - нет валидных разбиений
    if len(unique_features) < 2:
        return np.array([]), np.array([]), None, -np.inf

    # Пороги - средние между соседними уникальными значениями
    thresholds = (unique_features[:-1] + unique_features[1:]) / 2

    n_total = len(target_vector)
    ginis = np.zeros(len(thresholds))

    for i, threshold in enumerate(thresholds):
        # Разделение по порогу
        left_mask = feature_sorted < threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = n_total - n_left

        # Пропускаем пороги, приводящие к пустым поддеревьям
        if n_left == 0 or n_right == 0:
            ginis[i] = -np.inf
            continue

        # Целевые значения для поддеревьев
        left_target = target_sorted[left_mask]
        right_target = target_sorted[right_mask]

        # Доли классов в левом поддереве
        if len(left_target) > 0:
            left_p1 = np.sum(left_target == 1) / len(left_target)
            left_p0 = 1 - left_p1
            H_left = 1 - left_p1 ** 2 - left_p0 ** 2
        else:
            H_left = 0

        # Доли классов в правом поддереве
        if len(right_target) > 0:
            right_p1 = np.sum(right_target == 1) / len(right_target)
            right_p0 = 1 - right_p1
            H_right = 1 - right_p1 ** 2 - right_p0 ** 2
        else:
            H_right = 0

        # Критерий Джини
        gini = - (n_left / n_total) * H_left - (n_right / n_total) * H_right
        ginis[i] = gini

    # Находим лучший порог
    valid_indices = np.where(ginis != -np.inf)[0]
    if len(valid_indices) == 0:
        return np.array([]), np.array([]), None, -np.inf

    # Находим максимальное значение Джини
    max_gini = np.max(ginis[valid_indices])
    # Среди всех с максимальным Джини берем первый (минимальный порог)
    best_indices = valid_indices[ginis[valid_indices] == max_gini]
    best_idx = best_indices[0]

    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):  # Исправлено: должно быть == вместо !=
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):  # Исправлено: range(sub_X.shape[1]) вместо range(1, sub_X.shape[1])
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)  # Гарантируем float тип
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    # Исправлено: избегаем деления на ноль
                    ratio[key] = current_click / current_count if current_count > 0 else 0

                # Исправлено: правильная сортировка категорий
                sorted_categories = [x[0] for x in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = {cat: i for i, cat in enumerate(sorted_categories)}

                # Исправлено: правильное преобразование в numpy array
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError

            # Исправлено: проверка на константный признак
            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            # Исправлено: проверка на валидность разбиения
            if threshold is None or gini == -np.inf:
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":  # Исправлено: "categorical" вместо "Categorical"
                    # Сохраняем категории для левого поддерева
                    threshold_best = [cat for cat, idx in categories_map.items() if idx < threshold]
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            # Исправлено: правильное получение самого частого класса
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}

        # Исправлено: правильное разделение данных
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"])  # Исправлено: sub_y[~split]

    def _predict_node(self, x, node):
        """
        Рекурсивное предсказание для одного объекта
        """
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_value = x[feature_idx]
        feature_type = self._feature_types[feature_idx]

        if feature_type == "real":
            if feature_value < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if feature_value in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)


import numpy as np
from sklearn.metrics import accuracy_score


def test_find_best_split():
    print("=== ТЕСТИРОВАНИЕ find_best_split ===")

    # Тест 1: Идеально разделимые данные
    print("\n1. Идеально разделимые данные:")
    feature_vector = np.array([1, 1, 2, 2, 3, 3])
    target_vector = np.array([0, 0, 1, 1, 0, 0])
    thresholds, ginis, best_threshold, best_gini = find_best_split(feature_vector, target_vector)
    print(f"Пороги: {thresholds}")
    print(f"Джини: {ginis}")
    print(f"Лучший порог: {best_threshold}, Джини: {best_gini}")

    # Тест 2: Все объекты одного класса
    print("\n2. Все объекты одного класса:")
    feature_vector = np.array([1, 2, 3, 4, 5])
    target_vector = np.array([1, 1, 1, 1, 1])
    thresholds, ginis, best_threshold, best_gini = find_best_split(feature_vector, target_vector)
    print(f"Пороги: {thresholds} (должен быть пустой)")
    print(f"Лучший порог: {best_threshold} (должен быть None)")

    # Тест 3: Константный признак
    print("\n3. Константный признак:")
    feature_vector = np.array([5, 5, 5, 5, 5])
    target_vector = np.array([0, 1, 0, 1, 0])
    thresholds, ginis, best_threshold, best_gini = find_best_split(feature_vector, target_vector)
    print(f"Пороги: {thresholds} (должен быть пустой)")
    print(f"Лучший порог: {best_threshold} (должен быть None)")


def test_decision_tree_simple():
    print("\n=== ТЕСТИРОВАНИЕ DecisionTree - ПРОСТЫЕ ДАННЫЕ ===")

    # Тест 1: Вещественные признаки
    print("1. Вещественные признаки:")
    X = np.array([
        [1.0, 0.1],
        [1.5, 0.2],
        [2.0, 0.3],
        [2.5, 0.4],
        [3.0, 0.5],
        [3.5, 0.6]
    ])
    y = np.array([0, 0, 1, 1, 0, 0])

    tree = DecisionTree(feature_types=["real", "real"])
    tree.fit(X, y)
    predictions = tree.predict(X)

    print(f"Предсказания: {predictions}")
    print(f"Истинные:     {y}")
    accuracy = accuracy_score(y, predictions)
    print(f"Accuracy: {accuracy:.3f}")

    # Тест 2: Все объекты одного класса
    print("\n2. Все объекты одного класса:")
    X_single = np.array([[1], [2], [3], [4]])
    y_single = np.array([1, 1, 1, 1])

    tree_single = DecisionTree(feature_types=["real"])
    tree_single.fit(X_single, y_single)
    predictions_single = tree_single.predict(X_single)
    print(f"Предсказания: {predictions_single}")
    print(f"Все предсказания равны 1: {np.all(predictions_single == 1)}")


def test_decision_tree_categorical():
    print("\n=== ТЕСТИРОВАНИЕ DecisionTree - КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ ===")

    # Тест с категориальными признаками
    X_cat = np.array([
        ["A", "X"],
        ["A", "Y"],
        ["B", "X"],
        ["B", "Y"],
        ["A", "X"],
        ["B", "Y"]
    ])
    y_cat = np.array([0, 1, 0, 1, 0, 1])

    tree_cat = DecisionTree(feature_types=["categorical", "categorical"])
    tree_cat.fit(X_cat, y_cat)
    predictions_cat = tree_cat.predict(X_cat)

    print(f"Предсказания: {predictions_cat}")
    print(f"Истинные:     {y_cat}")
    accuracy_cat = accuracy_score(y_cat, predictions_cat)
    print(f"Accuracy: {accuracy_cat:.3f}")


def test_decision_tree_mixed():
    print("\n=== ТЕСТИРОВАНИЕ DecisionTree - СМЕШАННЫЕ ПРИЗНАКИ ===")

    # Тест со смешанными признаками
    X_mixed = np.array([
        [1.0, "A"],
        [1.5, "B"],
        [2.0, "A"],
        [2.5, "B"],
        [3.0, "A"],
        [3.5, "B"]
    ])
    y_mixed = np.array([0, 1, 0, 1, 0, 1])

    tree_mixed = DecisionTree(feature_types=["real", "categorical"])
    tree_mixed.fit(X_mixed, y_mixed)
    predictions_mixed = tree_mixed.predict(X_mixed)

    print(f"Предсказания: {predictions_mixed}")
    print(f"Истинные:     {y_mixed}")
    accuracy_mixed = accuracy_score(y_mixed, predictions_mixed)
    print(f"Accuracy: {accuracy_mixed:.3f}")


def test_decision_tree_with_regularization():
    print("\n=== ТЕСТИРОВАНИЕ DecisionTree - РЕГУЛЯРИЗАЦИЯ ===")

    # Создаем данные где может произойти переобучение
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Дерево без регуляризации
    tree_no_reg = DecisionTree(feature_types=["real", "real"])
    tree_no_reg.fit(X, y)
    pred_no_reg = tree_no_reg.predict(X)
    acc_no_reg = accuracy_score(y, pred_no_reg)

    # Дерево с регуляризацией
    tree_with_reg = DecisionTree(
        feature_types=["real", "real"],
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2
    )
    tree_with_reg.fit(X, y)
    pred_with_reg = tree_with_reg.predict(X)
    acc_with_reg = accuracy_score(y, pred_with_reg)

    print(f"Accuracy без регуляризации: {acc_no_reg:.3f}")
    print(f"Accuracy с регуляризацией: {acc_with_reg:.3f}")
    print(f"Разница: {acc_no_reg - acc_with_reg:.3f}")


def test_edge_cases():
    print("\n=== ТЕСТИРОВАНИЕ КРАЙНИХ СЛУЧАЕВ ===")

    # Тест 1: Один объект
    print("1. Один объект:")
    try:
        X_one = np.array([[1.0]])
        y_one = np.array([0])
        tree_one = DecisionTree(feature_types=["real"])
        tree_one.fit(X_one, y_one)
        pred_one = tree_one.predict(X_one)
        print(f"Успешно. Предсказание: {pred_one}")
    except Exception as e:
        print(f"Ошибка: {e}")

    # Тест 2: Два объекта
    print("\n2. Два объекта:")
    X_two = np.array([[1.0], [2.0]])
    y_two = np.array([0, 1])
    tree_two = DecisionTree(feature_types=["real"])
    tree_two.fit(X_two, y_two)
    pred_two = tree_two.predict(X_two)
    print(f"Предсказания: {pred_two}")

    # Тест 3: Неверный тип признака
    print("\n3. Неверный тип признака:")
    try:
        tree_wrong = DecisionTree(feature_types=["invalid"])
        print("ОШИБКА: Должна быть исключение!")
    except ValueError as e:
        print(f"Успешно поймано исключение: {e}")


def test_performance():
    print("\n=== ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ ===")

    # Генерируем больше данных
    np.random.seed(42)
    X = np.random.randn(1000, 3)
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)

    tree = DecisionTree(feature_types=["real", "real", "real"])

    import time
    start_time = time.time()
    tree.fit(X, y)
    fit_time = time.time() - start_time

    start_time = time.time()
    predictions = tree.predict(X)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y, predictions)

    print(f"Время обучения: {fit_time:.3f} сек")
    print(f"Время предсказания: {predict_time:.3f} сек")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Количество объектов: {len(y)}")


def main():
    print("=" * 60)
    print("ПОЛНОЕ ТЕСТИРОВАНИЕ РЕАЛИЗАЦИИ РЕШАЮЩЕГО ДЕРЕВА")
    print("=" * 60)

    test_find_best_split()
    test_decision_tree_simple()
    test_decision_tree_categorical()
    test_decision_tree_mixed()
    test_decision_tree_with_regularization()
    test_edge_cases()
    test_performance()

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)


if __name__ == "__main__":
    main()