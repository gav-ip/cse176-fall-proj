import numpy as np
from datasetFunc import filter_digits, generate_int
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat

data = loadmat('MNISTmini.mat')
digits = [2, 3]


print("Keys in .mat file: ", data.keys())
print(f"train_fea1 shape: {data['train_fea1'].shape}")
print(f"train_gnd1 shape: {data['train_gnd1'].shape}")
print(f"test_fea1 shape: {data['test_fea1'].shape}")
print(f"test_gnd1 shape: {data['test_gnd1'].shape}")

x_train_full, y_train_full = filter_digits(data['train_fea1'], data['train_gnd1'], [digits[0], digits[1]])
x_test_full, y_test_full = filter_digits(data['test_fea1'], data['test_gnd1'], [digits[0], digits[1]])
print(f"Filtered training set: {x_train_full.shape[0]} samples")
print(f"Filtered test set: {x_test_full.shape[0]} samples")

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size = 0.25,
    random_state = generate_int(),
    stratify = y_train_full
)

print(f"Final training set: {x_train.shape[0]} samples")
print(f"Validation set: {x_val.shape[0]} samples")
print(f"Test set: {x_test_full.shape[0]} samples")