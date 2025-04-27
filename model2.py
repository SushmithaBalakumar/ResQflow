import pandas as pd
import numpy as np
import random
import os
import joblib
from supabase import create_client
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 🔑 Supabase setup
url = 'https://bnxzvzqapyiovxnimbtl.supabase.co'
key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJueHp2enFhcHlpb3Z4bmltYnRsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUyNTY3NTYsImV4cCI6MjA2MDgzMjc1Nn0.2Stk0zVhwfXCPX1x8o5v2TOhqCytUFepFxbF5YO6Wgo'
supabase = create_client(url, key)

# 📦 Fetch dataset
response = supabase.table('resqflow').select('*').execute()
df = pd.DataFrame(response.data)
df.columns = df.columns.map(str.strip)
df.dropna(inplace=True)
df = df.loc[:, ~df.columns.duplicated()]

# 🎯 Target column
target_column = 'Flood Occurred'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

# 🔤 Label encode object columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    if col != target_column:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

if df[target_column].dtype == 'object':
    target_encoder = LabelEncoder()
    df[target_column] = target_encoder.fit_transform(df[target_column].astype(str))
else:
    target_encoder = None

# 📊 Features and target
X = df.drop(target_column, axis=1)
y = df[target_column].astype(int)

# 📏 Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ⚖ Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 🎓 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 🌲 Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 🧠 Predictions
y_pred = model.predict(X_test)

# 📈 Evaluation
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 🌊 Flood prediction logic
threshold_high = 0.5
threshold_low = 0.25

def get_flood_level(probabilities):
    if len(probabilities) == 2:
        if probabilities[1] >= threshold_high:
            return "High Flood Risk"
        elif probabilities[1] >= threshold_low:
            return "Low Flood Risk"
        else:
            return "No Flood Risk"
    return "Prediction Error"

def generate_surrounding_coords(center_lat, center_lon, lat_range=0.01, lon_range=0.01, steps=5):
    lat_steps = np.linspace(center_lat - lat_range, center_lat + lat_range, steps)
    lon_steps = np.linspace(center_lon - lon_range, center_lon + lon_range, steps)
    return [(round(lat, 6), round(lon, 6)) for lat in lat_steps for lon in lon_steps]

def generate_samples_for_coords(df, X_columns, coords):
    samples = []
    for lat, lon in coords:
        sample = {}
        for col in X_columns:
            if col == 'Latitude':
                sample[col] = lat
            elif col == 'Longitude':
                sample[col] = lon
            elif df[col].dtype in [np.float64, np.int64]:
                sample[col] = round(random.uniform(df[col].min(), df[col].max()), 4)
            else:
                sample[col] = random.choice(df[col].unique())
        samples.append(sample)
    return pd.DataFrame(samples)

# 🗺 Generate predictions for multiple locations
if 'Latitude' not in X.columns or 'Longitude' not in X.columns:
    raise ValueError("Dataset must contain 'Latitude' and 'Longitude' columns.")

central_lat = df['Latitude'].mean()
central_lon = df['Longitude'].mean()
coords = generate_surrounding_coords(central_lat, central_lon, lat_range=0.02, lon_range=0.02, steps=5)

multi_samples = generate_samples_for_coords(df, X.columns, coords)
multi_samples_scaled = scaler.transform(multi_samples)
multi_probs = model.predict_proba(multi_samples_scaled)
multi_preds = [get_flood_level(prob) for prob in multi_probs]

# 🔁 Supabase insert loop
for i, ((lat, lon), pred) in enumerate(zip(coords, multi_preds)):
    flood_occurred = 1 if "Flood" in pred else 0

    # Mock region based on latitude/longitude (you can enhance this)
    region = "Unknown"
    if lat >= 26 and lon >= 75:
        region = "Rajasthan"
    elif lat <= 10:
        region = "Kerala"

    # Estimate food/medical/resource needs
    population_factor = random.randint(5000, 10000)
    food_needed = population_factor
    medical_needed = int(population_factor * 0.8)
    faster_response = random.randint(70, 95)
    inventory_managed = random.randint(65, 90)
    strategic_allocation = random.randint(80, 95)

    data = {
        "latitude": lat,
        "longitude": lon,
        "prediction": pred,
        "region": region,
        "food_needed": food_needed,
        "medical_needed": medical_needed,
        "faster_response": faster_response,
        "inventory_managed": inventory_managed,
        "strategic_allocation": strategic_allocation
    }

    response = supabase.table("flood_data").insert(data).execute()
    print(f"✅ Inserted data for (Lat: {lat}, Lon: {lon}) ➜ {pred}")

# 💾 Save model + tools
os.makedirs("saved_model_rf", exist_ok=True)
joblib.dump(model, "saved_model_rf/random_forest_model.pkl")
joblib.dump(scaler, "saved_model_rf/scaler.pkl")
joblib.dump(label_encoders, "saved_model_rf/label_encoders.pkl")
if target_encoder:
    joblib.dump(target_encoder, "saved_model_rf/target_encoder.pkl")

print("\n✅ Model, scaler, and encoders saved in 'saved_model_rf/' folder.")
