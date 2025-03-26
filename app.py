from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model_path = "model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"⚠️ Model file '{model_path}' not found! Train the model first.")

with open(model_path, "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if form contains the correct field
        if "workout_duration" not in request.form:
            return render_template("index.html", error="⚠️ Form field 'workout_duration' is missing!")

        # Get input and validate
        workout_duration = request.form["workout_duration"].strip()

        if not workout_duration.replace(".", "", 1).isdigit():  # Allow decimals
            return render_template("index.html", error="⚠️ Please enter a valid number!")

        workout_duration = float(workout_duration)

        # Predict using the model
        prediction = model.predict(np.array([[workout_duration]]))[0]

        return render_template("index.html", prediction=round(prediction, 2), entered_value=workout_duration)

    except Exception as e:
        return render_template("index.html", error=f"⚠️ An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
