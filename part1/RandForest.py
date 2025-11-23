import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from scipy.io import loadmat

# -----------------------------
# 1. Load MNISTmini dataset
# -----------------------------
data_path = "MNISTmini.mat"
mnist = loadmat(data_path)

# Combine train and test features/labels
X_all = np.vstack([mnist['train_fea1'], mnist['test_fea1']])
y_all = np.hstack([mnist['train_gnd1'].flatten(), mnist['test_gnd1'].flatten()])

# Assigned digits for binary classification
digit1 = 2
digit2 = 3

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
y_train_bin = (y_train == digit2).astype(int)
y_val_bin = (y_val == digit2).astype(int)
y_test_bin = (y_test == digit2).astype(int)

print(f"Train shape: {X_train.shape}, Class distribution: {np.bincount(y_train_bin)}")
print(f"Val shape: {X_val.shape}, Class distribution: {np.bincount(y_val_bin)}")
print(f"Test shape: {X_test.shape}, Class distribution: {np.bincount(y_test_bin)}")

# -----------------------------
# 2. Random Forest
# -----------------------------
n_trees_list = [10, 50, 100, 200]
train_errors = []
val_errors = []

for n_trees in n_trees_list:
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rf.fit(X_train, y_train_bin)
    
    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)
    
    train_errors.append(1 - accuracy_score(y_train_bin, y_train_pred))
    val_errors.append(1 - accuracy_score(y_val_bin, y_val_pred))
    
    print(f"n_estimators={n_trees}: Train Error={train_errors[-1]:.4f}, Val Error={val_errors[-1]:.4f}")

best_idx = np.argmin(val_errors)
best_n_trees = n_trees_list[best_idx]
print(f"Best n_estimators: {best_n_trees} (Validation Error={val_errors[best_idx]:.4f})")

# Train final random forest
final_rf = RandomForestClassifier(n_estimators=best_n_trees, random_state=42)
final_rf.fit(X_train, y_train_bin)

y_test_pred = final_rf.predict(X_test)
test_acc = accuracy_score(y_test_bin, y_test_pred)
test_error = 1 - test_acc
print(f"Test Accuracy: {test_acc:.4f}, Test Error: {test_error:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test_bin, y_test_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=[str(digit1), str(digit2)],
            yticklabels=[str(digit1), str(digit2)])
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
