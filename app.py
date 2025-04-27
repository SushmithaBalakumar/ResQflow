from flask import Flask, render_template
import random

app = Flask(__name__)

# Utility function to simulate data for demo purposes
def generate_sample_data():
    lat = round(random.uniform(8.0, 35.0), 4)      # Latitude within India
    lon = round(random.uniform(68.0, 97.0), 4)     # Longitude within India

    prediction = random.choices(
        ["Low Flood Risk", "Medium Flood Risk", "High Flood Risk"],
        weights=[2, 3, 5], k=1
    )[0]

    region = random.choice(["Maharashtra", "Assam", "Bihar", "Tamil Nadu", "West Bengal", "Uttar Pradesh"])

    food_needed = random.randint(4000, 7000)
    medical_needed = random.randint(3000, 6000)

    faster_response = random.randint(75, 95)
    inventory_managed = random.randint(60, 90)
    strategic_allocation = random.randint(70, 98)

    return {
        "latitude": lat,
        "longitude": lon,
        "prediction": prediction,
        "region": region,
        "food_needed": food_needed,
        "medical_needed": medical_needed,
        "faster_response": faster_response,
        "inventory_managed": inventory_managed,
        "strategic_allocation": strategic_allocation
    }

@app.route("/")
def dashboard():
    data = generate_sample_data()
    return render_template("index.html", **data)

if __name__ == "_main_":
    app.run(debug=True)
