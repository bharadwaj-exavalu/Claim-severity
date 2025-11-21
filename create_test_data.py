import pandas as pd
import random

# Define possible values for each categorical column
data_options = {
    "Initial_Class_of_Claim": ["Bodily Injury", "Property Damage", "Comprehensive"],
    "Claimant_Injuries": ["Minor", "Moderate", "Severe", "None"],
    "Repairable_Flag": ["Yes", "No"],
    "Initial_Attorney_Involvement": ["Yes", "No"],
    "Primary_Cause_of_Accident": ["Rear-end Collision", "Side-impact", "Head-on", "Rollover"],
    "Rate_Class": ["Standard", "High Risk", "Preferred"],
    "Non_Drivable_Flag": ["Yes", "No"],
    "Claimant_State": ["CA", "TX", "NY", "FL", "IL"],
    "Primary_Accident_Description": ["Highway Accident", "Intersection Collision", "Parking Lot Incident"]
}

# Generate synthetic data
num_samples = 200  # Change this value for more rows
data = {
    column: [random.choice(values) for _ in range(num_samples)]
    for column, values in data_options.items()
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("testing.csv", index=False)

# Display sample output
print(df.head())
