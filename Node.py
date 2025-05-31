class Node():
    def __init__(self, left=None, right=None, feature_idx=None, threshold=None, impurity=None,
                 most_common_value=None) -> None:
        self.left = left
        self.right = right
        self.feature_idx = feature_idx  # dataset column index (for example, if feature_idx = 4 we use column cap-color)
        self.threshold = threshold
        self.impurity = impurity
        self.most_common_value = most_common_value

    def is_leaf(self):
        return (self.left == None) and (self.right == None)