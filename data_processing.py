import pandas as pd

# Load dataset
df = pd.read_csv("synthetic_claim_data.csv")

# Encode categorical variables (Binary Encoding)
binary_mappings = {
    "Claimant_Injuries": {"None": 0, "Minor": 1, "Moderate": 2, "Severe": 3},
    "Repairable_Flag": {"No": 0, "Yes": 1},
    "Total_Loss_Flag": {"No": 0, "Yes": 1},
    "Minor_Involved": {"No": 0, "Yes": 1},
    "Initial_Attorney_Involvement": {"No": 0, "Yes": 1},
    "Non_Drivable_Flag": {"No": 0, "Yes": 1},
    "Hospital_Visit": {"No": 0, "Yes": 1},
    "Pedestrian_Involvement": {"No": 0, "Yes": 1},
    "Performance_Vehicle": {"No": 0, "Yes": 1}
}

df.replace(binary_mappings, inplace=True)

# Label encode remaining categorical variables
for col in ["Primary_Cause_of_Accident", "Initial_Class_of_Claim", "Claimant_State", 
            "Primary_Accident_Description", "Rate_Class", "Vehicle_Type"]:
    df[col] = df[col].astype("category").cat.codes

# Rename target column
df.rename({"Claim_Amount": "claim_cost"}, axis=1, inplace=True)

# Remove outliers using IQR only for numerical columns
def remove_outliers_iqr(dataframe):
    numerical_cols = dataframe.select_dtypes(include=['number'])  # Select only numeric columns
    Q1 = numerical_cols.quantile(0.25)
    Q3 = numerical_cols.quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Apply outlier removal only on numerical columns
    df_cleaned = dataframe[~((numerical_cols < lower_bound) | (numerical_cols > upper_bound)).any(axis=1)]
    
    return df_cleaned

df = remove_outliers_iqr(df)

# Save cleaned data
df.to_csv("data.csv", index=False)

print(f"Data saved successfully with {df.shape[0]} rows and {df.shape[1]} columns.")