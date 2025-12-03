# train_flight_price_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

# --------------------------
# Load flights dataset
# --------------------------
flights = pd.read_csv("data/flights.csv")

# --------------------------
# Preprocess city names
# --------------------------
def city_only(s):
    if pd.isna(s): return s
    return str(s).split("(")[0].strip()

flights['from_city'] = flights['from'].apply(city_only)
flights['to_city'] = flights['to'].apply(city_only)

# --------------------------
# Encode target: agency
# --------------------------
le_agency = LabelEncoder()
flights['agency_encoded'] = le_agency.fit_transform(flights['agency'])

# --------------------------
# Features and Targets
# --------------------------
X = flights[['from_city','to_city','distance','flightType']]
y = flights[['price','time','agency_encoded']]

# OneHotEncode categorical features
cat_features = ['from_city','to_city','flightType']
num_features = ['distance']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# --------------------------
# Multi-output RandomForest
# --------------------------
rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))

pipeline = Pipeline([
    ('preproc', preprocessor),
    ('model', rf_model)
])

# --------------------------
# Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

# --------------------------
# Save model and label encoder
# --------------------------
joblib.dump(pipeline, "models/flight_price_model.joblib")
joblib.dump(le_agency, "models/label_encoder_agency.joblib")

print("Flight price model and agency label encoder saved successfully!")

# --------------------------
# Optional: test predictions
# --------------------------
y_pred = pipeline.predict(X_test)
price_pred, time_pred, agency_pred_encoded = y_pred[:,0], y_pred[:,1], y_pred[:,2].round().astype(int)
agency_pred = le_agency.inverse_transform(agency_pred_encoded)
for i in range(min(5, len(price_pred))):
    print(f"Predicted -> Price: {price_pred[i]:.2f}, Time: {time_pred[i]:.2f}, Agency: {agency_pred[i]}")
