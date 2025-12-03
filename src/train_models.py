import pandas as pd, numpy as np, os, joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import NearestNeighbors

# -------------------------------
# Create directories for outputs/models
# -------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -------------------------------
# Load CSVs
# -------------------------------
flights = pd.read_csv("data/flights.csv")
hotels = pd.read_csv("data/hotels.csv")
users = pd.read_csv("data/users.csv")

# -------------------------------
# Utility: extract city only
# -------------------------------
def city_only(s):
    if pd.isna(s): return s
    return str(s).split("(")[0].strip()

flights['from_city'] = flights['from'].apply(city_only)
flights['to_city'] = flights['to'].apply(city_only)
hotels['hotel_city'] = hotels['place'].apply(city_only)

# -------------------------------
# Aggregate flights by user
# -------------------------------
fl_agg = flights.groupby('userCode').agg(
    flights_count=('travelCode','count'),
    avg_flight_price=('price','mean'),
    avg_flight_time=('time','mean'),
    avg_distance=('distance','mean'),
    most_common_agency=('agency', lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
).reset_index().rename(columns={'userCode':'code'})

# Aggregate hotels by user
ht_agg = hotels.groupby('userCode').agg(
    hotels_count=('travelCode','count'),
    avg_hotel_price_per_night=('price','mean'),
    avg_hotel_days=('days','mean'),
    total_spent_on_hotels=('total','sum')
).reset_index().rename(columns={'userCode':'code'})

# Merge with users
user_features = users.merge(fl_agg, on='code', how='left').merge(ht_agg, on='code', how='left')

# Fill numeric NAs
num_defaults = {
    'flights_count':0,'hotels_count':0,'avg_flight_price':0,'avg_hotel_price_per_night':0,
    'avg_flight_time':0,'avg_hotel_days':0,'avg_distance':0,'total_spent_on_hotels':0
}
user_features = user_features.fillna(num_defaults)

# -------------------------------
# Gender Classifier
# -------------------------------
print("Training gender classifier...")
user_features['gender_filled'] = user_features['gender'].fillna("none")
le_gender = LabelEncoder()
y_gender = le_gender.fit_transform(user_features['gender_filled'])

X_gender = user_features.drop(columns=['name','gender','gender_filled','code'], errors='ignore')

numeric_features = X_gender.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_gender.select_dtypes(include=['object']).columns.tolist()

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
preproc = ColumnTransformer([('num', num_pipe, numeric_features), ('cat', cat_pipe, categorical_features)])

gender_clf = Pipeline([('preproc', preproc), ('rf', RandomForestClassifier(n_estimators=200, random_state=42))])

Xg_train, Xg_test, yg_train, yg_test = train_test_split(X_gender, y_gender, test_size=0.2, random_state=42, stratify=y_gender)
gender_clf.fit(Xg_train, yg_train)

yg_pred = gender_clf.predict(Xg_test)
print("Gender classifier accuracy:", accuracy_score(yg_test, yg_pred))
print(classification_report(yg_test, yg_pred, target_names=le_gender.classes_))

joblib.dump(gender_clf, "models/gender_classifier.joblib")
joblib.dump(le_gender, "models/label_encoder_gender.joblib")
print("Saved gender model + encoder -> models/")

# -------------------------------
# Flight Type Classifier
# -------------------------------
print("Training flight-type classifier...")
df = flights.dropna(subset=['flightType'])
y_f = df['flightType']
X_f = df[['price','time','distance','from_city','to_city']]

num_cols = ['price','time','distance']
cat_cols = ['from_city','to_city']

num_pipe_f = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
cat_pipe_f = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
pre_f = ColumnTransformer([('num', num_pipe_f, num_cols), ('cat', cat_pipe_f, cat_cols)])

flight_clf = Pipeline([('pre', pre_f), ('rf', RandomForestClassifier(n_estimators=200, random_state=42))])

Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_f, y_f, test_size=0.2, random_state=42, stratify=y_f)
flight_clf.fit(Xf_train, yf_train)

yf_pred = flight_clf.predict(Xf_test)
print("Flight-type accuracy:", accuracy_score(yf_test, yf_pred))
print(classification_report(yf_test, yf_pred))

joblib.dump(flight_clf, "models/flight_type_model.joblib")
print("Saved flight_type_model.joblib")

# -------------------------------
# Hotel Recommender Model
# -------------------------------
print("Training hotel recommender...")
# Create user-hotel matrix (ratings = total spent / days)
hotels['rating'] = hotels['total'] / hotels['days']
user_hotel_matrix = hotels.pivot_table(index='userCode', columns='travelCode', values='rating', fill_value=0)

# Fit NearestNeighbors (user-based collaborative filtering)
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(user_hotel_matrix.values)

# Save recommender model and matrix
joblib.dump(nn_model, "models/hotel_recommender_nn.joblib")
user_hotel_matrix.to_csv("outputs/user_hotel_matrix.csv")
print("Saved hotel recommender model + user-hotel matrix")

# Function to recommend hotels for a user
def recommend_hotels(user_id, top_n=5):
    if user_id not in user_hotel_matrix.index:
        print(f"User {user_id} not found")
        return []
    user_vector = user_hotel_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = nn_model.kneighbors(user_vector, n_neighbors=top_n+1)  # +1 to skip self
    similar_users = [user_hotel_matrix.index[i] for i in indices.flatten() if user_hotel_matrix.index[i] != user_id]
    recommended_hotels = set()
    for uid in similar_users:
        hotels_rated = user_hotel_matrix.loc[uid]
        top_hotels = hotels_rated.sort_values(ascending=False).head(top_n).index.tolist()
        recommended_hotels.update(top_hotels)
        if len(recommended_hotels) >= top_n:
            break
    return list(recommended_hotels)[:top_n]

# Example usage
print("Recommended hotels for user 0:", recommend_hotels(0))

# -------------------------------
# Save processed CSVs
# -------------------------------
user_features.to_csv("outputs/user_features.csv", index=False)
flights.to_csv("outputs/flights_processed.csv", index=False)
hotels.to_csv("outputs/hotels_processed.csv", index=False)
print("Saved outputs/ CSVs and models/")
