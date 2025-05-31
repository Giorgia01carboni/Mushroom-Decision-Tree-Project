# Decision Tree Classifier for Mushroom Edibility

## Project Overview
This project implements a decision tree classifier from scratch to determine whether mushrooms are edible or poisonous, based on the Secondary Mushroom Dataset. It explores different splitting criteria (Gini impurity, Information Gain via Entropy, and Misclassification Error) and includes hyperparameter tuning to optimize performance.

**Author:** Giorgia Carboni

## Files in this Repository
* `Node.py`: Defines the structure for individual nodes within the decision tree.
* `DecisionTree.py`: Contains the core implementation of the decision tree algorithm.
* `training.py`: Main script for data preprocessing, hyperparameter tuning, model training, and evaluation.
* `Decision_Tree.pdf`: Project report.
* `secondary_data.csv`: The dataset used for training and evaluating the models.
*  `primary_data.csv`: Noisy dataset.
* `.gitignore`: Specifies intentionally untracked files by Git.

## How to Run
1.  Ensure you have Python >=3.6 installed along with the necessary libraries (see Dependencies).
2.  Place `secondary_data.csv` in the same directory as `training.py`.
3.  Run the main script from the terminal:
    ```bash
    python training.py
    ```
    This will perform data preprocessing, hyperparameter tuning for the defined criteria, train the final models, and print evaluation metrics.

## Dependencies
* Python 3.11 (or version >= 3.6)
* pandas
* numpy
* scikit-learn (for `train_test_split` and `metrics`)
* seaborn (for plotting confusion matrices)
* matplotlib (for plotting)
