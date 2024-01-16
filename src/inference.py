import requests, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    """
    Data(laptopPrice.csv) load from current dir
    """
    root_dir = Path.cwd()
    data_dir = root_dir.joinpath("data")
    data = pd.read_csv(data_dir.joinpath("laptopPrice.csv"))
    return data


if __name__ == "__main__":
    data = load_data()
    X = data.drop(columns=["Price"])
    y = data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    url = "http://0.0.0.0:8080/predict/"
    json_vals = json.loads(json.dumps(X_test.iloc[0:1].to_json(orient="records")))
    json_values = {"test_data": json_vals}
    x = requests.post(url, json=json_values)
    print(x.json()["Price"])
