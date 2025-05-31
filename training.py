import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from DecisionTree import DecisionTree


# Convert numerical ranges or list-like strings into a float value (e.g.: average)
def parse_numerical_range(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            value = value[1:-1]
            parts = [p.strip() for p in value.split(',')]
            numbers = [float(p) for p in parts if p]
            if numbers:
                return sum(numbers) / len(numbers)
            else:
                return np.nan
        elif value:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return np.nan

# Convert categorical list-like strings into actual Python lists
def parse_categorical_cell(value):
    if pd.isna(value):
        return []
    value_str = str(value).strip()
    if value_str.startswith('[') and value_str.endswith(']'):
        # If the cell contains a list, remove any bracket and split by comma
        return [item.strip() for item in value_str[1:-1].split(',') if item.strip()]
    else:
        return [value_str] if value_str else []

# Ensure a cell value is a list
def ensure_list(cell):
    if isinstance(cell, (list, set, tuple)):
        return list(cell)
    elif pd.isna(cell):
        return []
    else:
        # anything else (string, int, etc.) becomes a single‐item list
        return [cell]

# Load dataset
try:
    df = pd.read_csv('secondary_data.csv', delimiter=';', na_values=[' ', '?'])
except FileNotFoundError:
    print("Error. File not found.")
    exit()

# Dataset info: notice how many missing value per column
print("\nDataframe info: ")
df.info()
# sns.histplot(df['class'])
# plt.show()

# Clean the dataframe from:
# 1. Missing values: use Mode Imputation
# 2. Drop columns with ≥ 85% missing values
# 3. Deal with multi-value in cells (eg: ['a', 'b', 'c'])

# For numerical features, if they have more than one value per cell take the average.
# If there's any missing value return it for Mode IMputation
numerical_cols = ['cap-diameter', 'stem-height', 'stem-width']
for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].apply(parse_numerical_range)
        df[col].fillna(df[col].median(), inplace=True)
    else:
        print(f"Column {col} not found in dataset.")

# Drop columns with > 85% missing values
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100
columns_over_85_missing = missing_percent[missing_percent > 85].index
if not columns_over_85_missing.empty:
    print("Dropping the following columns for having ≥85% missing values: ", columns_over_85_missing.tolist())
    df = df.drop(columns=columns_over_85_missing)

# Get categorical columns without labels
categorical_cols = [col for col in df.columns if col not in numerical_cols]
if 'class' in categorical_cols:
    categorical_cols.remove('class')

for col in categorical_cols:
    if col in df.columns:
        mode = df[col].mode()
        if not mode.empty:
            df[col].fillna(mode[0], inplace=True)
        else:
            print('Warning: empty column.')
    else:
        print(f"{col} not found in dataframe.")

# Parse the imputed columns
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].apply(parse_categorical_cell)

for col in categorical_cols:
    df[col] = df[col].apply(ensure_list)


X = df.drop(['class'], axis=1)
#Map the class parameters in binary form:
# 1 = poisonous, 0 = edible
y = df['class'].map({'p': 1, 'e': 0})

# Split dataset between train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train if y_train.nunique() > 1 else None) # 0.25 of 0.8 = 0.2 overall for validation


print(f"Original dataset shape: {df.shape}")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ----------------- Hyperparameter Tuning -----------------

# Available criteria: gini, entropy, misclassification_error
criteria = ['gini', 'entropy', 'misclassification_error']
max_depth = [5, 10, 12]
min_samples_split = [2, 5]
min_samples_per_leaf = [1, 2]

best_params = {}
best_val_metric = -1.0

for criterion in criteria:
    for depth in max_depth:
        for split_samples in min_samples_split:
            for samples_leaf in min_samples_per_leaf:
                param_grid = {
                    'criterion': criterion,
                    'max_depth': depth,
                    'min_samples_split': split_samples,
                    'min_samples_per_leaf': samples_leaf
                }

                print(f"Tuning with: {param_grid}")
                dtt = DecisionTree(criterion=criterion, max_depth=depth, min_samples_split=split_samples, min_samples_per_leaf=samples_leaf)
                dtt.fit(X_train.values, y_train.values)

                dtt_val_predictions = dtt.predict(X_val.values)
                dtt_val_accuracy = metrics.accuracy_score(y_val, dtt_val_predictions)
                print(f"Validation set accuracy: {dtt_val_accuracy:.4f}")

                if dtt_val_accuracy > best_val_metric:
                    best_val_metric = dtt_val_accuracy
                    best_params = param_grid
                    print(f"\nFound new best validation accuracy: {best_val_metric:.4f}")
print(f"\n\nBest parameters found: {best_params}, best validation accuracy registered:{best_val_metric:.4f}")

# Retrain on full train+val
X_train_val = np.concatenate((X_train.values, X_val.values), axis=0)
y_train_val = np.concatenate((y_train.values, y_val.values), axis=0)
final_dtt = DecisionTree(criterion=best_params['criterion'],
                         max_depth=best_params['max_depth'],
                         min_samples_split=best_params['min_samples_split'],
                         min_samples_per_leaf=best_params['min_samples_per_leaf'])
final_dtt.fit(X_train_val, y_train_val)

# Evaluate
dtt_test_pred = final_dtt.predict(X_test.values)
test_accuracy = metrics.accuracy_score(y_test, dtt_test_pred)
dtt_train_pred = final_dtt.predict(X_train_val)
train_error = 1 - metrics.accuracy_score(y_train_val, dtt_train_pred)

print(f"\nFinal model test accuracy: {test_accuracy:.4f}")
print(f"Training error (0-1 Loss) on train+val set: {train_error:.4f}")

# Confusion Matrix and report
cm = confusion_matrix(y_test, dtt_test_pred)
print("Confusion Matrix: \n\n", cm)
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, linewidths=0.5,linecolor="red", fmt= '.0f',ax=ax)
plt.show()

print(classification_report(y_test, dtt_test_pred))
f1_score = f1_score(y_test, dtt_test_pred)
print("F1 Score:",f1_score)