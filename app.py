import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='template')

# Load trained model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    # Read user input
    input_feature = [x for x in request.form.values()]
    feature_values = [np.array(input_feature)]

    # Define feature names
    original_features = ["holiday", "temp", "rain", "snow", "weather", "year", "month", "day", "hours", "minutes", "seconds"]
    
    # Convert input into DataFrame
    data = pd.DataFrame(feature_values, columns=original_features)

    # Ensure categorical features exist in input
    categorical_features = ["holiday", "weather"]  # Categorical columns used in training

    for col in categorical_features:
        if col not in data.columns:
            data[col] = "Unknown"  # Default category for missing features

    # Apply One-Hot Encoding (Same as Training)
    encoded_features = encoder.transform(data[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)

    # Convert encoded data to DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    # Drop original categorical columns and merge encoded data
    data = data.drop(columns=categorical_features).reset_index(drop=True)
    data = pd.concat([data, encoded_df], axis=1)

    # Ensure feature names match the trained model's input
    data = data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict
    prediction = model.predict(data)
    text = "Estimated Traffic Volume: "
    return render_template("output.html", result=text + str(prediction))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
