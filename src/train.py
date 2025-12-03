import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import mlflow
import os

def train_model():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'data', 'processed_flights.csv')
    model_path = os.path.join(base_dir, 'models', 'xgb_model.joblib')
    mlruns_dir = os.path.join(base_dir, 'mlruns')

    # Ensure mlruns directory exists
    os.makedirs(mlruns_dir, exist_ok=True)

    # Set MLflow tracking URI to local directory (Windows compatible)
    mlflow.set_tracking_uri(f"file:///{mlruns_dir.replace(os.sep, '/')}")

    # Set experiment name (unchanged)
    mlflow.set_experiment("flight_price_prediction_demo1")

    df = pd.read_csv(data_path)
    X = df.drop(columns=['price'])
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Experiment 1
    with mlflow.start_run(run_name="experiment_1"):
        params1 = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100
        }
        model1 = xgb.XGBRegressor(**params1)
        model1.fit(X_train, y_train)
        preds1 = model1.predict(X_test)
        
        mae1 = mean_absolute_error(y_test, preds1)
        rmse1 = mean_squared_error(y_test, preds1) ** 0.5
        r2_1 = r2_score(y_test, preds1)

        mlflow.log_params(params1)
        mlflow.log_metric("mae", mae1)
        mlflow.log_metric("rmse", rmse1)
        mlflow.log_metric("r2", r2_1)

        print(f"Experiment 1 - MAE: {mae1}, RMSE: {rmse1}, R2: {r2_1}")

    # Experiment 2
    with mlflow.start_run(run_name="experiment_2"):
        params2 = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 8,
            'n_estimators': 200
        }
        model2 = xgb.XGBRegressor(**params2)
        model2.fit(X_train, y_train)
        preds2 = model2.predict(X_test)
        
        mae2 = mean_absolute_error(y_test, preds2)
        rmse2 = mean_squared_error(y_test, preds2) ** 0.5
        r2_2 = r2_score(y_test, preds2)

        mlflow.log_params(params2)
        mlflow.log_metric("mae", mae2)
        mlflow.log_metric("rmse", rmse2)
        mlflow.log_metric("r2", r2_2)

        print(f"Experiment 2 - MAE: {mae2}, RMSE: {rmse2}, R2: {r2_2}")

    # Choose the best model based on MAE
    if mae1 < mae2:
        best_model = model1
        print("Best model: Experiment 1")
    else:
        best_model = model2
        print("Best model: Experiment 2")

    # Save the best model
    joblib.dump(best_model, model_path)
    print(f"Best model saved at {model_path}")

if __name__ == "__main__":
    train_model()
