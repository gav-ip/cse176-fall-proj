import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

def load_binary_digits(data_path, digit1, digit2, random_seed):
    """
    Loads MNISTmini dataset, filters for two specific digits, shuffles,
    splits into train/val/test (1000 samples each), and standardizes features.
    
    Args:
        data_path (str): Path to the .mat file.
        digit1 (int): First digit to filter (mapped to 0).
        digit2 (int): Second digit to filter (mapped to 1).
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Load the dataset
    mnist = loadmat(data_path)

    # Combine train and test features/labels
    X_all = np.vstack([mnist['train_fea1'], mnist['test_fea1']])
    y_all = np.hstack([mnist['train_gnd1'].flatten(), mnist['test_gnd1'].flatten()])

   # Separate the two digits
    X_digit1 = X_all[y_all == digit1]
    y_digit1 = y_all[y_all == digit1]

    X_digit2 = X_all[y_all == digit2]
    y_digit2 = y_all[y_all == digit2]

    # Fixed train/val/test split (500 per digit per split)
    X_train_raw = np.vstack([X_digit1[:500], X_digit2[:500]])
    y_train_raw = np.hstack([y_digit1[:500], y_digit2[:500]])

    X_val_raw = np.vstack([X_digit1[500:1000], X_digit2[500:1000]])
    y_val_raw = np.hstack([y_digit1[500:1000], y_digit2[500:1000]])

    X_test_raw = np.vstack([X_digit1[1000:1500], X_digit2[1000:1500]])
    y_test_raw = np.hstack([y_digit1[1000:1500], y_digit2[1000:1500]])

    # Shuffle splits
    X_train, y_train = shuffle(X_train_raw, y_train_raw, random_state=42)
    X_val, y_val = shuffle(X_val_raw, y_val_raw, random_state=42)
    X_test, y_test = shuffle(X_test_raw, y_test_raw, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert labels to 0/1 (digit1=0, digit2=1)
    y_train = (y_train == digit2).astype(int)
    y_val = (y_val == digit2).astype(int)
    y_test = (y_test == digit2).astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test

