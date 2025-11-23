import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

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
    mnist_data = loadmat(data_path)

    # Extract all features and labels
    train_features_full = mnist_data['train_fea1']
    train_labels_full = mnist_data['train_gnd1'].flatten()
    test_features_full = mnist_data['test_fea1']
    test_labels_full = mnist_data['test_gnd1'].flatten()

    # Combine to search across the entire dataset
    all_features_full = np.vstack([train_features_full, test_features_full])
    all_labels_full = np.hstack([train_labels_full, test_labels_full])

    # Filter for specific digits
    mask = (all_labels_full == digit1) | (all_labels_full == digit2)
    indices = np.where(mask)[0]

    # Shuffle the indices to mix the classes
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Select the first 3000 samples found
    if len(indices) < 3000:
        print(f"Warning: Only found {len(indices)} samples for digits {digit1} and {digit2}")
        selected_indices = indices
    else:
        selected_indices = indices[:3000]

    selected_features = all_features_full[selected_indices]
    selected_labels = all_labels_full[selected_indices]

    # Split into sets (hardcoded 1000 split as per original script)
    X_train_raw = selected_features[:1000]
    y_train_raw = selected_labels[:1000]

    X_val_raw = selected_features[1000:2000]
    y_val_raw = selected_labels[1000:2000]

    X_test_raw = selected_features[2000:3000]
    y_test_raw = selected_labels[2000:3000]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Relabel classes to 0 and 1 for binary classification
    # digit1 -> 0, digit2 -> 1
    y_train = (y_train_raw == digit2).astype(int)
    y_val = (y_val_raw == digit2).astype(int)
    y_test = (y_test_raw == digit2).astype(int)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

