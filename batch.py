from flask import Flask, request, jsonify, send_file
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)

# Load model and encoders
model = joblib.load("random_forest_best_model.pkl")
binary_encodings = joblib.load("binary_encodings.pkl")
label_encodings = joblib.load("label_encodings.pkl")
best_features = joblib.load("selected_features.pkl")
mae = joblib.load("final_mae.pkl")
adjusted_r2 = joblib.load("final_adjusted_r2.pkl")
explainer = shap.TreeExplainer(model)

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        # Check if file is uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})
        
        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Read uploaded file
        df = pd.read_csv(filepath)
        input_df = df.copy()
        
        # Encode categorical features
        for col, mapping in binary_encodings.items():
            if col in input_df:
                input_df[col] = input_df[col].map(mapping).fillna(0)
        
        for col, mapping in label_encodings.items():
            if col in input_df:
                input_df[col] = input_df[col].astype("category").cat.set_categories(mapping.values()).cat.codes.fillna(-1)
        
        # Ensure feature alignment
        input_df = input_df.reindex(columns=best_features, fill_value=0)
        
        # Perform predictions
        predictions = model.predict(input_df)
        df["Predicted_Claim_Cost"] = predictions
        
        # Compute SHAP values
        shap_values = explainer.shap_values(input_df)
        
        # Calculate metrics
        actual_values = df["claim_cost"] if "claim_cost" in df else None
        final_mae = mean_absolute_error(actual_values, predictions) if actual_values is not None else None
        final_r2 = r2_score(actual_values, predictions) if actual_values is not None else None
        final_adj_r2 = adjusted_r2(final_r2, input_df.shape[0], len(best_features)) if actual_values is not None else None
        
        # Save results to Excel
        results_filepath = os.path.join(RESULTS_FOLDER, "batch_predictions.xlsx")
        df.to_excel(results_filepath, index=False)
        
        # Create Actual vs. Predicted dictionary
        actual_vs_predicted = dict(zip(actual_values, predictions)) if actual_values is not None else None
        actual_vs_predicted_float = {float(key): float(value) for key, value in actual_vs_predicted.items()} if actual_vs_predicted else None
        
        # Response
        response = {
            "mae": final_mae,
            "adjusted_r2": final_adj_r2,
            "actual_vs_predicted": actual_vs_predicted_float,
            "download_link": f"/download/{os.path.basename(results_filepath)}"
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    filepath = os.path.join(RESULTS_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "File not found"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)