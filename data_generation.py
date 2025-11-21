import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
n = 10000  # Number of records

data = {
    "Claim_ID": [f"CLM{str(i).zfill(6)}" for i in range(1, n+1)],
    "Initial_Reserve_Amount": np.random.randint(500, 50000, n),
    "Primary_Cause_of_Accident": np.random.choice(["Rear-end Collision", "Side-impact", "Head-on", "Rollover"], n),
    "Claimant_Age": np.random.randint(18, 80, n),
    "Initial_Class_of_Claim": np.random.choice(["Bodily Injury", "Property Damage", "Comprehensive"], n),
    "Repairable_Flag": np.random.choice(["Yes", "No"], n, p=[0.7, 0.3]),
    "Total_Loss_Flag": np.random.choice(["Yes", "No"], n, p=[0.2, 0.8]),
    "Claimant_State": np.random.choice(["CA", "TX", "NY", "FL", "IL"], n),
    "Minor_Involved": np.random.choice(["Yes", "No"], n, p=[0.1, 0.9]),
    "Days_to_Int_Reserve_since_FNOL": np.random.randint(1, 30, n),
    "Primary_Accident_Description": np.random.choice(["Highway Accident", "Intersection Collision", "Parking Lot Incident"], n),
    "Initial_Attorney_Involvement": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
    "Total_People_Involved": np.random.randint(1, 5, n),
    "Non_Drivable_Flag": np.random.choice(["Yes", "No"], n, p=[0.4, 0.6]),
    "Number_of_Injuries": np.random.randint(0, 5, n),
    "Rate_Class": np.random.choice(["Standard", "High Risk", "Preferred"], n),
    "Hospital_Visit": np.random.choice(["Yes", "No"], n, p=[0.2, 0.8]),
    "Vehicle_Type": np.random.choice(["Sedan", "SUV", "Truck", "Motorcycle"], n),
    "Pedestrian_Involvement": np.random.choice(["Yes", "No"], n, p=[0.05, 0.95]),
    "Combined_BI_Limits": np.random.choice([25000, 50000, 100000, 300000], n),
    "Performance_Vehicle": np.random.choice(["Yes", "No"], n, p=[0.1, 0.9]),
    "Claimant_Injuries": np.random.choice(["Minor", "Moderate", "Severe", "None"], n, p=[0.3, 0.4, 0.2, 0.1])
}

# Ensure Repairable_Flag and Total_Loss_Flag are logically consistent
data["Repairable_Flag"] = np.where(np.array(data["Total_Loss_Flag"]) == "Yes", "No", data["Repairable_Flag"])

# Base claim amount
data["Base_Claim_Amount"] = np.random.randint(10000, 50000, n)

# Multipliers for different conditions
severe_multiplier = np.where(np.array(data["Claimant_Injuries"]) == "Severe", np.random.uniform(5, 8, n), 1)
moderate_multiplier = np.where(np.array(data["Claimant_Injuries"]) == "Moderate", np.random.uniform(2, 4, n), 1)
minor_multiplier = np.where(np.array(data["Claimant_Injuries"]) == "Minor", np.random.uniform(1.2, 1.8, n), 1)
total_loss_multiplier = np.where(np.array(data["Total_Loss_Flag"]) == "Yes", np.random.uniform(10, 15, n), 1)
attorney_multiplier = np.where(np.array(data["Initial_Attorney_Involvement"]) == "Yes", np.random.uniform(1.8, 2.5, n), 1)
non_drivable_multiplier = np.where(np.array(data["Non_Drivable_Flag"]) == "Yes", np.random.uniform(1.3, 2, n), 1)
pedestrian_multiplier = np.where(np.array(data["Pedestrian_Involvement"]) == "Yes", np.random.uniform(2.5, 3.5, n), 1)
hospital_multiplier = np.where(np.array(data["Hospital_Visit"]) == "Yes", np.random.uniform(2, 3, n), 1)
minor_involved_multiplier = np.where(np.array(data["Minor_Involved"]) == "Yes", np.random.uniform(1.2, 1.5, n), 1)
performance_vehicle_multiplier = np.where(np.array(data["Performance_Vehicle"]) == "Yes", np.random.uniform(1.5, 2, n), 1)
rate_class_multiplier = np.where(np.array(data["Rate_Class"]) == "High Risk", np.random.uniform(1.5, 2, n), 1)
accident_cause_multiplier = np.where(np.array(data["Primary_Cause_of_Accident"]) == "Head-on", np.random.uniform(1.8, 2.5, n), 1)
claim_class_multiplier = np.where(np.array(data["Initial_Class_of_Claim"]) == "Bodily Injury", np.random.uniform(2.5, 3.5, n), 1)
repairable_multiplier = np.where(np.array(data["Repairable_Flag"]) == "No", np.random.uniform(1.8, 2.5, n), 1)

# State-based multiplier (for example, claims in CA and NY may be higher)
state_multiplier = np.where(pd.Series(data["Claimant_State"]).isin(["CA", "NY"]), np.random.uniform(1.3, 2, n), 1)

# Compute final Claim Amount
data["Claim_Amount"] = (data["Base_Claim_Amount"] * severe_multiplier * moderate_multiplier * minor_multiplier *
                        total_loss_multiplier * attorney_multiplier * non_drivable_multiplier *
                        pedestrian_multiplier * hospital_multiplier * minor_involved_multiplier *
                        performance_vehicle_multiplier * rate_class_multiplier * accident_cause_multiplier *
                        claim_class_multiplier * repairable_multiplier * state_multiplier).astype(int)

# Convert to DataFrame
df = pd.DataFrame(data)
df.drop("Base_Claim_Amount",axis=1,inplace=True)
# Save to CSV
df.to_csv("synthetic_claim_data.csv", index=False)

print("Synthetic claim data generated and saved to 'synthetic_claim_data.csv'.")
