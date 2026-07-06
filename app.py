
from flask import Flask, request, render_template
import pandas as pd
import pickle
import sklearn

print("Scikit-learn Version:", sklearn.__version__)

# Load trained model and preprocessor
dtr = pickle.load(open("dtr.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

# Create Flask app
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from HTML form
        Year = float(request.form["Year"])
        average_rain_fall_mm_per_year = float(request.form["average_rain_fall_mm_per_year"])
        pesticides_tonnes = float(request.form["pesticides_tonnes"])
        avg_temp = float(request.form["avg_temp"])
        Area = request.form["Area"]
        Item = request.form["Item"]

        # Create DataFrame (same column names as training data)
        features = pd.DataFrame({
            "Year": [Year],
            "average_rain_fall_mm_per_year": [average_rain_fall_mm_per_year],
            "pesticides_tonnes": [pesticides_tonnes],
            "avg_temp": [avg_temp],
            "Area": [Area],
            "Item": [Item]
        })

        # Transform data
        transformed_features = preprocessor.transform(features)

        # Predict
        prediction = float(dtr.predict(transformed_features)[0])

        prediction = round(prediction, 2)

        return render_template(
            "index.html",
            prediction=prediction
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=False)
    