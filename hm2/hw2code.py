import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    if np.all(feature_vector == feature_vector[0]):
        return np.array([]), np.array([]), None, -np.inf

    sorted_idx = np.argsort(feature_vector)
    sorted_feat = feature_vector[sorted_idx]
    sorted_target = target_vector[sorted_idx]

    uniq_vals, uniq_idx = np.unique(sorted_feat, return_index=True)
    if len(uniq_vals) < 2:
        return np.array([]), np.array([]), None, -np.inf

    thresholds = (uniq_vals[:-1] + uniq_vals[1:]) / 2.0

    n_total = len(target_vector)
    cumsum_target = np.cumsum(sorted_target)
    total_ones = cumsum_target[-1]

    split_positions = uniq_idx[1:]

    n_left = split_positions
    n_right = n_total - n_left

    valid = (n_left > 0) & (n_right > 0)
    if not np.any(valid):
        return np.array([]), np.array([]), None, -np.inf

    n_left = n_left[valid]
    n_right = n_right[valid]
    pos_valid = split_positions[valid]

    ones_left = cumsum_target[pos_valid - 1]
    zeros_left = n_left - ones_left

    ones_right = total_ones - ones_left
    zeros_right = n_right - ones_right

    gini_left = 1.0 - (ones_left / n_left) ** 2 - (zeros_left / n_left) ** 2
    gini_right = 1.0 - (ones_right / n_right) ** 2 - (zeros_right / n_right) ** 2

    Q = -(n_left / n_total) * gini_left - (n_right / n_total) * gini_right

    best_i = np.argmax(Q)
    gini_best = Q[best_i]
    thresh_idx = np.where(valid)[0][best_i]
    threshold_best = thresholds[thresh_idx]

    valid_thresholds = thresholds[valid]
    valid_ginis = Q

    return valid_thresholds, valid_ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split or 2
        self._min_samples_leaf = min_samples_leaf or 1

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p1 = np.mean(y)
        p0 = 1 - p1
        return 1 - p1**2 - p0**2

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if len(sub_y) == 0:
            node["type"] = "terminal"
            node["class"] = 0
            return

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best = None, None, -np.inf
        best_split = None
        best_categories_map = None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            feature_vector = sub_X[:, feature]

            if feature_type == "real":
                thresholds, ginis, thresh, gini = find_best_split(feature_vector, sub_y)
                if len(thresholds) == 0:
                    continue
                if gini > gini_best:
                    gini_best = gini
                    feature_best = feature
                    threshold_best = thresh
                    best_split = feature_vector < thresh
                    best_categories_map = None

            elif feature_type == "categorical":
                categories = np.unique(feature_vector)
                if len(categories) < 2:
                    continue

                category_means = {}
                for cat in categories:
                    mask = (feature_vector == cat)
                    if np.any(mask):
                        category_means[cat] = np.mean(sub_y[mask])
                    else:
                        category_means[cat] = 0.0

                sorted_categories = sorted(categories, key=lambda x: category_means[x])
                categories_map = {cat: idx for idx, cat in enumerate(sorted_categories)}
                mapped_feature = np.array([categories_map[x] for x in feature_vector])

                thresholds, ginis, thresh, gini = find_best_split(mapped_feature, sub_y)
                if len(thresholds) == 0:
                    continue

                if gini > gini_best:
                    gini_best = gini
                    feature_best = feature
                    threshold_best = thresh
                    split_mask = mapped_feature < thresh
                    best_split = split_mask
                    left_ranks = set(np.where(mapped_feature < thresh)[0])
                    left_categories = [cat for cat, rank in categories_map.items() if rank < thresh]
                    best_categories_map = left_categories

            else:
                raise ValueError("Unknown feature type")

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = best_categories_map

        left_mask = best_split
        right_mask = ~best_split

        if (np.sum(left_mask) < self._min_samples_leaf) or (np.sum(right_mask) < self._min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["left_child"] = {}
        node["right_child"] = {}

        self._fit_node(sub_X[left_mask], sub_y[left_mask], node["left_child"], depth + 1)
        self._fit_node(sub_X[right_mask], sub_y[right_mask], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        feature = node["feature_split"]
        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

