import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# Load dataset
data = pd.read_csv(r"C:\Users\patel\Downloads\traffic_volume.csv")

# Define target variable
target = "traffic_volume"

# Ensure target column exists
if target not in data.columns:
    raise KeyError(f"Column '{target}' not found in dataset.")

# Define feature columns
possible_features = ["holiday", "temp", "rain", "snow", "weather", "year", "month", "day", "hours", "minutes", "seconds"]
features = [col for col in possible_features if col in data.columns]  # Use only existing features

X = data[features]
y = data[target]

# Identify categorical columns
categorical_features = [col for col in ["holiday", "weather"] if col in X.columns]

# Apply One-Hot Encoding
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

if categorical_features:
    encoded_features = encoder.fit_transform(X[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)

    # Convert to DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    # Drop original categorical columns and merge encoded data
    X = X.drop(columns=categorical_features).reset_index(drop=True)
    X = pd.concat([X, encoded_df], axis=1)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoder
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("✅ Model and encoder saved!")
