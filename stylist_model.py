import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("stylist_data.csv")

# Encode categorical variables
encoder_season = LabelEncoder()
encoder_occasion = LabelEncoder()
encoder_body_type = LabelEncoder()
encoder_clothing_type = LabelEncoder()

data["Season"] = encoder_season.fit_transform(data["Season"])
data["Occasion"] = encoder_occasion.fit_transform(data["Occasion"])
data["Body_Type"] = encoder_body_type.fit_transform(data["Body_Type"])
data["Clothing_Type"] = encoder_clothing_type.fit_transform(data["Clothing_Type"])

# Define features and target
X = data[["Season", "Occasion", "Body_Type"]]
y = data["Clothing_Type"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model and encoders
with open("stylist_model.pkl", "wb") as f:
    pickle.dump((model, encoder_season, encoder_occasion, encoder_body_type, encoder_clothing_type), f)

print("✅ Model trained and saved successfully!")
