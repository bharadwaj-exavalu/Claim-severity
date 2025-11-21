from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import shap

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="Claim Severity API", version="1.0")

# -----------------------------
# Load model and dependencies
# -----------------------------
model = joblib.load("random_forest_best_model.pkl")
binary_encodings = joblib.load("binary_encodings.pkl")
label_encodings = joblib.load("label_encodings.pkl")
best_features = joblib.load("selected_features.pkl")
mae = joblib.load("final_mae.pkl")
adjusted_r2 = joblib.load("final_adjusted_r2.pkl")
X_train = pd.read_csv("X_train.csv").drop(columns=["Unnamed: 0"], errors="ignore")

actual_vs_predicted = joblib.load("actual_vs_predicted.pkl")
actual_vs_predicted = {float(k): float(v) for k, v in actual_vs_predicted.items()}

explainer = shap.TreeExplainer(model)

# -----------------------------
# Request Body Schema
# -----------------------------
from pydantic import RootModel

class PredictInput(RootModel[dict]):
    pass


# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def home():
    return {"status": "running", "message": "Claim Severity API is live"}

# -----------------------------
# Predict endpoint
# -----------------------------
@app.post("/predict")
def predict(input_data: PredictInput):
    try:
        data = input_data.root

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Binary encoding
        for col, mapping in binary_encodings.items():
            if col in input_df:
                input_df[col] = input_df[col].map(mapping).fillna(0)

        # Label encoding
        for col, mapping in label_encodings.items():
            if col in input_df:
                input_df[col] = (
                    input_df[col]
                    .astype("category")
                    .cat.set_categories(mapping.values())
                    .cat.codes
                    .fillna(-1)
                )

        # Align features
        input_df = input_df.reindex(columns=best_features, fill_value=0)

        # Prediction
        prediction = model.predict(input_df)[0]

        # SHAP values
        shap_values = explainer.shap_values(input_df)

        shap_list = sorted(
            [
                (feature, data.get(feature, None), float(value))
                for feature, value in zip(best_features, shap_values[0])
            ],
            key=lambda x: abs(x[2]),
            reverse=True,
        )[:5]

        return {
            "prediction": float(prediction),
            "mae": float(mae),
            "adjusted_r2": float(adjusted_r2),
            "top_5_shap_values": shap_list,
            "all_actual_vs_predicted": actual_vs_predicted,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Run (only if running directly)
# -----------------------------
# Use: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
