import os
import random
import pickle
import requests
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load dataset
data = pd.read_csv("stylist_data.csv")

# Encode categorical variables
encoder_season = LabelEncoder()
encoder_occasion = LabelEncoder()
encoder_body_type = LabelEncoder()
encoder_clothing_type = LabelEncoder()

data['Season'] = encoder_season.fit_transform(data['Season'])
data['Occasion'] = encoder_occasion.fit_transform(data['Occasion'])
data['Body_Type'] = encoder_body_type.fit_transform(data['Body_Type'])
data['Clothing_Type'] = encoder_clothing_type.fit_transform(data['Clothing_Type'])

# Train the model
X = data[['Season', 'Occasion', 'Body_Type']]
y = data['Clothing_Type']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the trained model and encoders
with open('stylist_model.pkl', 'wb') as f:
    pickle.dump((model, encoder_season, encoder_occasion, encoder_body_type, encoder_clothing_type), f)

# Load the trained model and encoders
with open('stylist_model.pkl', 'rb') as f:
    model, encoder_season, encoder_occasion, encoder_body_type, encoder_clothing_type = pickle.load(f)

# Pexels API Key (replace with your own)
PEXELS_API_KEY = "cbdMZ4fXmxRRBmVUR8MNMVLahFZ30RCapVqI3uz3oYMOL5U9EaJDl6Ry"

def get_random_image(query):
    """Fetch a random clothing image from Pexels."""
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=10"
    headers = {"Authorization": PEXELS_API_KEY}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        results = response.json().get("photos", [])
        if results:
            return random.choice(results)["src"]["medium"]  # Return a random image URL
    return "static/images/default.jpg"  # Fallback image

@app.route("/", methods=["GET", "POST"])
def index():
    prediction, image_url = None, None

    if request.method == "POST":
        season = request.form["season"]
        occasion = request.form["occasion"]
        body_type = request.form["body_type"]

        # Encode input
        input_data = [
            encoder_season.transform([season])[0],
            encoder_occasion.transform([occasion])[0],
            encoder_body_type.transform([body_type])[0],
        ]

        # Predict clothing type
        prediction_index = model.predict([input_data])[0]
        prediction = encoder_clothing_type.inverse_transform([prediction_index])[0]

        # Fetch image from Pexels
        image_url = get_random_image(prediction)

    return render_template("index.html", prediction=prediction, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
