import pandas as pd
import joblib
import os
import numpy as np

def load_model_and_encoders():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'models', 'xgb_model.joblib')
    encoder_path = os.path.join(base_dir, 'models', 'label_encoders.joblib')

    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    return model, encoders

def prepare_input(user_input, encoders):
    df_input = pd.DataFrame([user_input])

    for col, encoder in encoders.items():
        if col in df_input.columns:
            if df_input[col].values[0] not in encoder.classes_:
                raise ValueError(f"Invalid input for {col}: {df_input[col].values[0]}")
            df_input[col] = encoder.transform(df_input[col])

    return df_input

def predict_price(user_input):
    model, encoders = load_model_and_encoders()
    df_input = prepare_input(user_input, encoders)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'data', 'processed_flights.csv')
    df_train = pd.read_csv(data_path)
    column_order = df_train.drop(columns=['price']).columns

    df_input = df_input.reindex(columns=column_order, fill_value=0)

    prediction = model.predict(df_input)[0]
    return prediction

def print_valid_options(encoders):
    print("Valid options for the inputs:")
    for col, encoder in encoders.items():
        if col in ['from', 'to', 'flightType', 'agency']:
            print(f"{col}: {list(encoder.classes_)}")
    print("-" * 50)

if __name__ == "__main__":
    model, encoders = load_model_and_encoders()

    # Print available options
    print_valid_options(encoders)

    # Example user input with full names as in dataset
    user_input = {
        'from': 'Sao Paulo (SP)',
        'to': 'Rio de Janeiro (RJ)',
        'flightType': 'economic',
        'agency': 'Rainbow',
        'day': 15,
        'month': 9,
        'year': 2025
    }

    try:
        price = predict_price(user_input)
        print(f"Predicted flight price: R$ {price:,.2f}")
    except ValueError as e:
        print(f"Error: {e}")