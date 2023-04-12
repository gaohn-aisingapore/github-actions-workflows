import threading
import time

import numpy as np
import requests
from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Generate synthetic data for linear regression
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Train and evaluate the linear regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    x_input = np.array(data["x"]).reshape(-1, 1)
    y_output = model.predict(x_input)
    return jsonify(y_output.tolist())


@app.route("/test", methods=["GET"])
def test():
    return "Test successful"


def run_app():
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    t = threading.Thread(target=run_app)
    t.start()

    time.sleep(5)  # Wait for the server to start

    response = requests.get("http://0.0.0.0:5000/test")
    print(response.text)

    # Shutdown the server
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()
