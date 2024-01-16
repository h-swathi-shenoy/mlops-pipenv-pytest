import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib
import json
from src.preprocessor import TransformationPipeline
from src.preprocessor import RegressorModel


def load_model():
    """
    Keras Model load from artifacts dir
    """
    model_dir = Path.cwd().joinpath("artifacts")
    model = tf.keras.models.load_model(model_dir)
    return model


def load_data():
    """
    Data(laptopPrice.csv) load from current dir
    """
    root_dir = Path.cwd()
    data_dir = root_dir.joinpath("data")
    data = pd.read_csv(data_dir.joinpath("laptopPrice.csv"))
    return data


def retrain(test_size: 0.1, model_dir: Path):
    """
    When data refreshed, retrain the model with New data includes transform, split,retrain model
    """
    data = load_data()
    X = data.drop(columns=["Price"])
    y = data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    transformation = TransformationPipeline(X_train)
    preprocessor = transformation.preprocess()
    proc_obj = preprocessor.fit(X_train)
    joblib.dump(proc_obj, model_dir.joinpath("preprocessor.pkl"))
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    model_obj = RegressorModel(X_train, y_train, X_test, y_test, model_dir)
    model_obj.model_train()


def transform_input(vals: pd.DataFrame):
    """
    Takes the inputs from user and transforms using Transformer model stored in artifacts
    """
    artifacts_dir = Path.cwd().joinpath("artifacts")
    model = joblib.load(artifacts_dir.joinpath("preprocessor.pkl"))
    transformed_vals = model.transform(vals)
    return transformed_vals


def human_readable_payload(predict_value):
    """Takes numpy array and returns back human readable dictionary"""

    laptop_price = float(np.round(predict_value, 2))
    result = {
        "Price": laptop_price,
    }
    return result


def predict(inputs):
    """Takes inputs(18 values)  and predicts price"""

    clf = load_model()  # loadmodel
    print("model_loaded")
    vals_df = pd.DataFrame(inputs)
    scaled_inputs = transform_input(vals_df)
    scaled_height_prediction = clf.predict(scaled_inputs)  # scaled prediction
    payload = human_readable_payload(scaled_height_prediction)
    return payload


if __name__ == "__main__":
    data = load_data()
    X = data.drop(columns=["Price"])
    y = data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    json_vals = json.dumps(json.loads(X_test.iloc[0:1].to_json(orient="records")))
    json_values = {"test_data": json_vals}
    print(json.loads(json_values["test_data"]))
