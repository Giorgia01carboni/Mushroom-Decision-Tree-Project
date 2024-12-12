import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/primary_data.csv', delimiter=';')
#print(df.columns)
X = df.drop(['class'], axis=1)
y = df['class']
print(X.head())
print(y.head())

#Perform hot-encoding
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class Node():
  def __init__(self, left=None, right=None, feature_idx=None, feature_val=None, threshold=None, impurity=None, most_common_value=None) -> None:
    self.left = left
    self.right = right
    self.feature_idx = feature_idx #dataset column index
    self.feature_val = feature_val #dataset rows indices
    self.threshold = threshold
    self.impurity = impurity
    self.most_common_value = most_common_value

  def is_leaf(self):
    return (self.left == None) and (self.right == None)


class DecisionTree():
  def __init__(self, root=None, max_depth=8, min_samples_split=2, min_samples_per_leaf=1, criterion='gini') -> None:
    self.root = root
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.min_samples_per_leaf = min_samples_per_leaf
    self.criterion = criterion # gini or entropy

  def fit(self, X, y):
    self.root = self._build_tree(X, y, depth=0)

  def _build_tree(self, X, y, depth = 0):
    """check stopping conditions, find best split, split the current node into left and right, recursively build subtrees"""
    n_samples = X.shape[0]

    # if all instances in X belong to the the same class, return a node with that class
    if self._pure_node(y):
      most_common_class = self._find_class(y)
      return Node(impurity=0.0, most_common_value=most_common_class)

    # check halt conditions
    if depth >= self.max_depth or n_samples < self.min_samples_split:
      most_common_class = self._find_class(y)
      return Node(impurity=0.0, most_common_value=most_common_class)

    # find best split
    best_feature, best_thresh = self._find_best_split(X, y)

    # check if best_feature and best_thresh were found
    if best_feature is None or best_thresh is None:
      most_common_class = self._find_class(y)
      return Node(most_common_value = most_common_class)

    # split the current node into left and right
    left_child, right_child = self._split(X, best_feature, best_thresh)

    # check minimum number of samples per leaf
    if len(left_child) < self.min_samples_per_leaf or len(right_child) < self.min_samples_per_leaf:
      most_common_class = self._find_class(y)
      return Node(most_common_value=most_common_class)

    # increase tree depth
    depth += 1

    # recursively build subtrees
    left_subtree = self._build_tree(X[left_child], y[left_child], depth)
    right_subtree = self._build_tree(X[right_child], y[right_child], depth)

    return Node(left=left_subtree, right=right_subtree, feature_idx=best_feature, threshold=best_thresh, impurity=self._gini(y))

  def _find_best_split(self, X, y):
    """"""
    best_ig = -float('inf')
    best_gini = -float('inf')
    best_feature = None
    best_thresh = None

    if self.criterion == 'entropy':
      for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
          left_child, right_child = self._split(X, feature_idx, threshold)
          ig = self._information_gain(y, y[left_child], y[right_child])
          if ig > best_ig:
            best_ig = ig
            best_feature = feature_idx
            best_thresh = threshold
    elif self.criterion == 'gini':
      for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
          left_child, right_child = self._split(X, feature_idx, threshold)
          parent_gini = self._gini(y)
          left_gini = self._gini(y[left_child])
          right_gini = self._gini(y[right_child])
          weighted_gini = (len(left_child) / len(y)) * left_gini + (len(right_child) / len(y)) * right_gini
          if parent_gini - weighted_gini > best_gini:
            best_gini = parent_gini - weighted_gini
            best_feature = feature_idx
            best_thresh = threshold

    return best_feature, best_thresh


  def _split(self, X, feature_idx, best_split) -> tuple:
    """split dataset based on best_split. Return left and right child"""
    left_child = []
    right_child = []

    left_child = np.where(X[:, feature_idx] <= best_split)[0]
    right_child = np.where(X[:, feature_idx] > best_split)[0]
    #left_child = np.where((X[:, feature_idx] >= best_split) if isinstance(best_split, (int, float)) else (X[:, feature_idx] == best_split))[0]
    #right_child = np.where((X[:, feature_idx] < best_split) if isinstance(best_split, (int, float)) else (X[:, feature_idx] != best_split))[0]


    return left_child, right_child


  def _predict(self, x):
    """Traverse the tree from root to leaf to predict the class of input x"""


  def _gini(self, y) -> float:
    """Probability of a random sample being misclassified if it were labeled according to the distribution in the node"""
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
    return y.mode()[0]