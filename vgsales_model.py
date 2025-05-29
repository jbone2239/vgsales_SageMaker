import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from io import StringIO

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "vgsales_model.pkl"))
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        columns = ['Platform', 'Genre', 'Publisher', 'Year']
        input_df = pd.read_csv(StringIO(request_body), names=columns)
        return input_df
    else:
        raise ValueError("Unsupported content type: " + request_content_type)

def output_fn(prediction, content_type):
    if content_type == "text/csv":
        return ','.join(str(x) for x in prediction)
    else:
        raise ValueError("Unsupported content type: " + content_type)

def predict_fn(input_data, model):
    model_columns = joblib.load(os.path.join("/opt/ml/model", "model_columns.pkl"))
    le = joblib.load(os.path.join("/opt/ml/model", "label_encoder.pkl"))

    input_data_encoded = pd.get_dummies(input_data)
    for col in model_columns:
        if col not in input_data_encoded:
            input_data_encoded[col] = 0
    input_data_encoded = input_data_encoded[model_columns]

    predictions = model.predict(input_data_encoded)
    return le.inverse_transform(predictions)

def main():
    df = pd.read_csv("/opt/ml/input/data/training/vgsales.csv")
    df = df.dropna(subset=['Platform', 'Genre', 'Publisher', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales'])

    def get_top_region(row):
        return max(['NA_Sales', 'EU_Sales', 'JP_Sales'], key=lambda region: row[region])
    df['Top_Region'] = df.apply(get_top_region, axis=1)

    df = df.drop(columns=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])

    X = df[['Platform', 'Genre', 'Publisher', 'Year']]
    y = df['Top_Region']

    X_encoded = pd.get_dummies(X)
    model_columns = X_encoded.columns.tolist()
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_encoded, y_encoded)

    model_dir = "/opt/ml/model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "vgsales_model.pkl"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    joblib.dump(model_columns, os.path.join(model_dir, "model_columns.pkl"))

if __name__ == "__main__":
    main()
