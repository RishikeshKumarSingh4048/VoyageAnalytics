from flask import Flask, render_template, request, flash
import joblib
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# --------------------------
# Load Models
# --------------------------
models_dir = "models"

# Gender model
gender_clf = joblib.load(os.path.join(models_dir, "gender_classifier.joblib"))
le_gender = joblib.load(os.path.join(models_dir, "label_encoder_gender.joblib"))

# Flight type model
flight_clf = joblib.load(os.path.join(models_dir, "flight_type_model.joblib"))

# Hotel recommender
hotel_nn = joblib.load(os.path.join(models_dir, "hotel_recommender_nn.joblib"))
user_hotel_matrix = pd.read_csv("outputs/user_hotel_matrix.csv", index_col=0)

# Flight price model
flight_price_model = joblib.load(os.path.join(models_dir, "flight_price_model.joblib"))
le_agency = joblib.load(os.path.join(models_dir, "label_encoder_agency.joblib"))

# Load CSVs for dropdowns
users_df = pd.read_csv("data/users.csv")
flights_df = pd.read_csv("data/flights.csv")

# --------------------------
# Utility functions
# --------------------------
def recommend_hotels(user_id, top_n=5):
    if user_id not in user_hotel_matrix.index:
        return []
    user_vector = user_hotel_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = hotel_nn.kneighbors(user_vector, n_neighbors=top_n+1)
    similar_users = [user_hotel_matrix.index[i] for i in indices.flatten() if user_hotel_matrix.index[i] != user_id]
    recommended_hotels = set()
    for uid in similar_users:
        hotels_rated = user_hotel_matrix.loc[uid]
        top_hotels = hotels_rated.sort_values(ascending=False).head(top_n).index.tolist()
        recommended_hotels.update(top_hotels)
        if len(recommended_hotels) >= top_n:
            break
    return list(recommended_hotels)[:top_n]

# Dropdown options
options = {
    "user_ids": sorted(users_df['code'].unique()),
    "from_city": sorted(flights_df['from'].apply(lambda x: str(x).split("(")[0].strip()).unique()),
    "to_city": sorted(flights_df['to'].apply(lambda x: str(x).split("(")[0].strip()).unique()),
    "flight_types": sorted(flights_df['flightType'].unique())
}

# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_input = {}
    try:
        if request.method == "POST":
            task = request.form.get("task")
            user_input = request.form.to_dict()

            # ---------- Gender Prediction ----------
            if task == "gender":
                X = pd.DataFrame([{
                    'flights_count': float(user_input.get("flights_count", 0)),
                    'hotels_count': float(user_input.get("hotels_count", 0)),
                    'avg_flight_price': float(user_input.get("avg_flight_price", 0)),
                    'avg_hotel_price_per_night': float(user_input.get("avg_hotel_price_per_night", 0)),
                    'avg_flight_time': float(user_input.get("avg_flight_time", 0)),
                    'avg_hotel_days': float(user_input.get("avg_hotel_days", 0)),
                    'avg_distance': float(user_input.get("avg_distance", 0)),
                    'total_spent_on_hotels': float(user_input.get("total_spent_on_hotels", 0))
                }])
                pred = gender_clf.predict(X)[0]
                prediction = f"Predicted Gender: {le_gender.inverse_transform([pred])[0]}"

            # ---------- Flight Type Prediction ----------
            elif task == "flight_type":
                X = pd.DataFrame([{
                    "price": float(user_input.get("price", 0)),
                    "time": float(user_input.get("time", 0)),
                    "distance": float(user_input.get("distance", 0)),
                    "from_city": user_input.get("from_city", ""),
                    "to_city": user_input.get("to_city", ""),
                }])
                pred = flight_clf.predict(X)[0]
                prediction = f"Predicted Flight Type: {pred}"

            # ---------- Flight Price Prediction ----------
            elif task == "flight_price":
                X = pd.DataFrame([{
                    "distance": float(user_input.get("distance", 0)),
                    "from_city": user_input.get("from_city", ""),
                    "to_city": user_input.get("to_city", ""),
                    "flightType": user_input.get("flight_type", "")
                }])
                y_pred = flight_price_model.predict(X)
                price_pred, time_pred, agency_encoded = y_pred[0]
                agency_pred = le_agency.inverse_transform([int(round(agency_encoded))])[0]
                prediction = f"Predicted Price: ${price_pred:.2f}, Time: {time_pred:.2f} hrs, Agency: {agency_pred}"

            # ---------- Hotel Recommendations ----------
            elif task == "hotel_recommend":
                user_id = int(user_input.get("user_id", -1))
                recs = recommend_hotels(user_id)
                if recs:
                    prediction = f"Recommended Hotels for User {user_id}: {', '.join(map(str, recs))}"
                else:
                    prediction = f"No recommendations found for User {user_id}"

    except Exception as e:
        flash(f"Error: {str(e)}")

    return render_template("index.html", prediction=prediction, user_input=user_input, options=options)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
