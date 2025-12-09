import pandas as pd
import pickle
import sys

# Usage check
if len(sys.argv) < 2:
    print("Usage: python predict_from_pickle.py input.csv")
    sys.exit(1)

input_file = sys.argv[1]
print(f"Loading data from {input_file}...")

# Load data
df = pd.read_csv(input_file)

# Load saved model + scaler
with open("fraud_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
scaler = saved["scaler"]

# Remove Class column if present
if "Class" in df.columns:
    df = df.drop("Class", axis=1)

# Ensure required columns exist
required = ["Time", "Amount"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Input file must contain '{col}' column.")

# Scale Time + Amount
df_scaled = df.copy()
df_scaled[["Time", "Amount"]] = scaler.transform(df_scaled[["Time", "Amount"]])

# Predict probabilities + class
probs = model.predict_proba(df_scaled)[:, 1]
preds = model.predict(df_scaled)

# Create clean output ONLY
output = pd.DataFrame({
    "fraud_probability": probs,
    "prediction": preds
})

# Save output
output_file = "predictions_output.csv"
output.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
