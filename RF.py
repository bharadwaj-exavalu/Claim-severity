import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data.csv")
df.drop('Claim_ID', axis=1, inplace=True, errors='ignore')  # Drop ID column if exists
df.fillna(0, inplace=True)  # Fill missing values

# Select only numerical features
df1 = df.select_dtypes(include=['int64', 'float64'])

# Define features and target
X = df1.drop('claim_cost', axis=1)
y = df1['claim_cost']

# Split data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Drop index column if it exists
if "Unnamed: 0" in X_train.columns:
    X_train.drop("Unnamed: 0", axis=1, inplace=True)
    X_test.drop("Unnamed: 0", axis=1, inplace=True)

# **Function to Compute Adjusted R²**
def adjusted_r2(r2, n, p):
    """Compute Adjusted R². Prevents division errors if n <= p + 1"""
    return r2 if n <= p + 1 else 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# **Forward Selection Function Using Adjusted R²**
def forward_selection(X_train, X_test, y_train, y_test):
    selected_features = []
    remaining_features = list(X_train.columns)
    best_adj_r2 = -np.inf  # Best Adjusted R² score so far

    while remaining_features:
        best_feature = None
        for feature in remaining_features:
            trial_features = selected_features + [feature]
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X_train[trial_features], y_train)
            y_pred = rf.predict(X_test[trial_features])
            r2 = r2_score(y_test, y_pred)

            # Compute Adjusted R² using **training set** size
            n, p = X_train.shape[0], len(trial_features)
            adj_r2 = adjusted_r2(r2, n, p)

            # Keep track of best feature
            if adj_r2 > best_adj_r2 or best_adj_r2 == -np.inf:  
                best_adj_r2 = adj_r2
                best_feature = feature

        if best_feature:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            print(f"Added {best_feature}, Adjusted R² improved to: {best_adj_r2:.4f}")
        else:
            break  # Stop if no feature improves Adjusted R²

    return selected_features

# **Run Forward Selection**
best_features = forward_selection(X_train, X_test, y_train, y_test)

# **Train Final Model with Selected Features**
final_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
final_model.fit(X_train[best_features], y_train)

# **Evaluate Model**
y_pred = final_model.predict(X_test[best_features])
final_r2 = r2_score(y_test, y_pred)
final_adj_r2 = adjusted_r2(final_r2, X_train.shape[0], len(best_features))  # Consistent `n`
final_mae = mean_absolute_error(y_test, y_pred)
final_mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error

# **Save Model & Selected Features**
joblib.dump(final_model, "random_forest_best_model.pkl")
joblib.dump(best_features, "selected_features.pkl")
joblib.dump(final_adj_r2, "final_adjusted_r2.pkl")
joblib.dump(final_mae, "final_mae.pkl")

# **Save Actual vs. Predicted Values as Key-Value Pairs**
# Explicitly convert both key and value to float
actual_vs_predicted = dict(zip(y_test, y_pred))
actual_vs_predicted_float = {float(key): float(value) for key, value in actual_vs_predicted.items()}

# Save the dictionary with float keys and values
joblib.dump(actual_vs_predicted_float, "actual_vs_predicted.pkl")  # Save the dictionary with floats

# **Plot Actual vs. Predicted values**
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Claim Cost")
plt.ylabel("Predicted Claim Cost")
plt.title(f"Actual vs. Predicted Claim Costs (Adjusted R² = {final_adj_r2:.3f})")
plt.show()

# **Error Histogram**
plt.figure(figsize=(8, 6))
sns.histplot(y_test - y_pred, bins=30, kde=True)
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.show()

# **Print Results**
print(f"\nSelected Features: {best_features}")
print(f"Final Adjusted R² Score: {final_adj_r2:.4f}")
print(f"Final Mean Absolute Error: {final_mae:.2f}")
print(f"Final Mean Absolute Percentage Error: {final_mape:.2f}%")
print("Model saved successfully!")
print("Actual vs. Predicted values saved successfully as 'actual_vs_predicted.pkl'!")
