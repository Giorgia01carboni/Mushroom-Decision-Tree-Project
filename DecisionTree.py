import numpy as np
from collections import Counter
from Node import Node


class DecisionTree():
    def __init__(self, criterion, root=None, max_depth=8, min_samples_split=2, min_samples_per_leaf=1) -> None:
        self.root = root
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_per_leaf = min_samples_per_leaf
        self.criterion = criterion  # gini or entropy

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth=0):
        """check stopping conditions, find best split, split the current node into left and right, recursively build subtrees"""
        n_samples = X.shape[0]

        # Check stop conditions.
        if self._pure_node(y) or depth >= self.max_depth or len(y) < self.min_samples_split:
            most_common_class = self._find_class(y)
            return Node(impurity=0.0, most_common_value=most_common_class)

        # find best split
        best_feature, best_thresh = self._find_best_split(X, y)

        # check if best_feature and best_thresh were found
        if best_feature is None or best_thresh is None:
            most_common_class = self._find_class(y)
            return Node(most_common_value=most_common_class)

        # split the current node into left and right
        left_child, right_child = self._split(X, best_feature, best_thresh)

        # check minimum number of samples per leaf
        if len(left_child) <= self.min_samples_per_leaf or len(right_child) <= self.min_samples_per_leaf:
            most_common_class = self._find_class(y)
            return Node(most_common_value=most_common_class)

        # recursively build subtrees
        left_subtree = self._build_tree(X[left_child], y[left_child], depth+1)
        right_subtree = self._build_tree(X[right_child], y[right_child], depth+1)
        impurity = self._gini(y) if self.criterion == 'gini' else self._entropy(y) if self.criterion == 'entropy' \
            else self._misclassification_error(y)

        return Node(left=left_subtree, right=right_subtree, feature_idx=best_feature, threshold=best_thresh,
                    impurity=impurity)

# ----------------- Split Section -----------------

    def _feature_type(self, column):
        """ Distinguish between categorical or numerical features. """
        if all(isinstance(val, (int, float)) for val in column):
            return 'numerical'
        if any(isinstance(val, list) for val in column):
            return 'categorical_multi'

    def _find_best_split(self, X, y):
        """"""
        best_ig = -float('inf')  # information gain
        best_gini = best_misclass = -float('inf')
        best_feature = best_thresh = None

        for feature_idx in range(X.shape[1]):
            col = X[:, feature_idx]
            feature_type = self._feature_type(col)

            if feature_type == 'categorical_multi':
                thresholds = {item
                              for cell in col
                              for item in cell}

            elif feature_type == 'numerical':
                numeric_values = [val for val in X[:, feature_idx] if isinstance(val, (int, float))]

                # Check if we have enough unique numbers to make a split
                if not numeric_values or len(np.unique(numeric_values)) < 2:
                    continue  # Skip this feature

                unique_numeric_vals = np.unique(numeric_values)
                sorted_vals = np.sort(unique_numeric_vals)
                thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2

            for threshold in thresholds:
                left_child, right_child = self._split(X, feature_idx, threshold)

                # if any of the node is empty because of a split, skip this (no information gain from this)
                if left_child is None or right_child is None:
                    continue

                if self.criterion == 'entropy':
                    ig = self._information_gain(y, y[left_child], y[right_child])
                    if ig > best_ig:
                        best_ig = ig
                        best_feature = feature_idx
                        best_thresh = threshold

                elif self.criterion == 'gini':
                    parent_gini = self._gini(y)
                    left_gini = self._gini(y[left_child])
                    right_gini = self._gini(y[right_child])
                    weighted_gini = (len(left_child) / len(y)) * left_gini + (
                                len(right_child) / len(y)) * right_gini
                    if parent_gini - weighted_gini > best_gini:
                        best_gini = parent_gini - weighted_gini
                        best_feature = feature_idx
                        best_thresh = threshold

                elif self.criterion == 'misclassification_error':
                    parent_loss = self._misclassification_error(y)
                    left_loss = self._misclassification_error(y[left_child])
                    right_loss = self._misclassification_error(y[right_child])
                    weighted_loss = (len(left_child) / len(y)) * left_loss + \
                                    (len(right_child) / len(y)) * right_loss
                    if parent_loss - weighted_loss > best_misclass:
                        best_misclass = parent_loss - weighted_loss
                        best_feature = feature_idx
                        best_thresh = threshold

        return best_feature, best_thresh

    def _split(self, X, feature_idx, best_split) -> tuple:
        """split dataset based on best_split.
        Split condition for categorical features: check membership.
        Split condition for numerical features: check inequality.
        Return left and right child"""

        left_child = []
        right_child = []
        feature_type = self._feature_type(X[:, feature_idx])

        if feature_type == 'numerical':
            for i, val in enumerate(X[:, feature_idx]):
                if isinstance(val, (int, float)) and val <= best_split:
                    left_child.append(i)
                else:
                    right_child.append(i)
            left_child = np.array(left_child)
            right_child = np.array(right_child)

        elif feature_type == 'categorical_multi':
            for i, cell in enumerate(X[:, feature_idx]):
                categories = set(cell) if isinstance(cell, (list, set)) else set()
                if best_split in categories:
                    left_child.append(i)
                else:
                    right_child.append(i)

        if len(left_child) == 0 or len(right_child) == 0:
            return None, None

        return left_child, right_child

# ----------------- Helper Functions and Criteria -----------------
    def predict(self, X):
        """Traverse the tree from root to leaf to predict the class of input x"""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Tree traversal procedure:
        starting from input array x (containing an entire row of data),
        we check if we hit a leaf (in this case, return the corresponding most common class),
        else, if the corresponding column value (from row x) is less than the considered threshold,
        we traverse the left subtree,
        else, we traverse the right subtree"""

        if node.is_leaf():
            return node.most_common_value

        feature_val = x[node.feature_idx]

        # Check if feature_val is multi-category (list or set)
        if isinstance(feature_val, (list, set)):
            return self._traverse_tree(x, node.left if node.threshold in feature_val else node.right)

        # Numerical
        else:
            return self._traverse_tree(x, node.left if feature_val <= node.threshold else node.right)

    def _gini(self, y) -> float:
        """
        Probability of a random sample being misclassified if it were labeled
        according to the distribution in the node
        """
        gini = 0.0
        for feature in np.unique(y):
            prob = len(y[y == feature]) / len(y)
            gini += prob ** 2
        return 1 - gini

    def _entropy(self, y) -> float:
        """Measure node impurity. Lower entropy => node is homogenous => better split"""
        entropy = 0.0
        for feature in np.unique(y):
            prob = len(y[y == feature]) / len(y)
            entropy -= prob * np.log2(prob)
        return entropy

    def _misclassification_error(self, y) -> float:
        """
        Proportion of incorrect predictions in a node.
        Smaller error => Better split.
        Formula: misclassification_loss(t) = 1 - max(p_i),
        where t = node, p_i probability of class i in node t.
        """
        max_prob = 0.0

        for label in np.unique(y):
            prob = len(y[y == label]) / len(y)
            if prob > max_prob:
                max_prob = prob
        return 1.0 - max_prob

    def _information_gain(self, y, left, right) -> float:
        ig = 0.0

        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(left)
        right_entropy = self._entropy(right)

        weight_left = len(left) / len(y)
        weight_right = len(right) / len(y)

        ig = parent_entropy - (weight_left * left_entropy + weight_right * right_entropy)

        return ig

    def _pure_node(self, y) -> bool:
        """Check if the samples in the node belong all to the same class. Avoids further splitting if that's true."""
        return len(np.unique(y)) == 1

    def _find_class(self, y):
        """return the most common class in the node"""
        return Counter(y).most_common(1)[0][0]
