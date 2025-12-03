import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data():
    # Get the current working directory reliably
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'data', 'flights.csv')
    processed_data_path = os.path.join(base_dir, 'models', 'processed_flights.csv')
    encoder_path = os.path.join(base_dir, 'models', 'label_encoders.joblib')

    print(f"Loading data from: {data_path}")  # Debugging line

    # Load dataset
    df = pd.read_csv(data_path)

    # Drop unnecessary columns
    df = df.drop(columns=['Unnamed: 8', 'Unnamed: 9'], errors='ignore')

    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows with invalid dates if any
    df = df.dropna(subset=['date'])

    # Extract day, month, year from date
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Drop original 'date' column
    df = df.drop(columns=['date'])

    # Initialize LabelEncoders dictionary
    label_encoders = {}
    categorical_cols = ['from', 'to', 'flightType', 'agency']

    # Apply label encoding to categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Save the label encoders for future use
    joblib.dump(label_encoders, encoder_path)

    # Save the processed data
    df.to_csv(processed_data_path, index=False)

    print(f"Data preprocessing completed. Processed data saved at {processed_data_path}")
    print(f"Label encoders saved at {encoder_path}")

if __name__ == "__main__":
    preprocess_data()
