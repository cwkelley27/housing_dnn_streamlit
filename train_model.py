import os
import joblib
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("data/Housing_Hamilton_Compressed.csv.gz", compression="gzip")

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# ----------------------------
# Select features and target
# ----------------------------
features = [
    "CALC_ACRES",
    "LAND_USE_CODE_DESC",
    "NEIGHBORHOOD_CODE_DESC",
    "ZONING_DESC",
    "PROPERTY_TYPE_CODE_DESC"
]
target = "APPRAISED_VALUE"

df = df[features + [target]].copy()

# ----------------------------
# Clean data
# ----------------------------
df = df[df[target].notna()]
df = df[df[target] > 0]
df = df.dropna()

print("Cleaned shape:", df.shape)

# ----------------------------
# One-hot encode categorical variables
# ----------------------------
df_encoded = pd.get_dummies(
    df,
    columns=[
        "LAND_USE_CODE_DESC",
        "NEIGHBORHOOD_CODE_DESC",
        "ZONING_DESC",
        "PROPERTY_TYPE_CODE_DESC"
    ],
    drop_first=True
)

X = df_encoded.drop(columns=[target])
y = df_encoded[target]

feature_names = X.columns.tolist()

# ----------------------------
# Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Scale inputs
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Build neural network
# ----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ----------------------------
# Train model
# ----------------------------
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    verbose=1
)

# ----------------------------
# Evaluate model
# ----------------------------
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print("Test MAE:", mae)

# ----------------------------
# Save artifacts
# ----------------------------
import os
from pathlib import Path

os.makedirs("artifacts", exist_ok=True)

model.save("artifacts/housing_model.h5")
joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(feature_names, "artifacts/feature_names.pkl")

print("Current working directory:", os.getcwd())
print("Artifacts folder path:", Path("artifacts").resolve())
print("Files in artifacts:", os.listdir("artifacts"))
print("Artifacts saved successfully.")

