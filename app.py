from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
nb_model = pickle.load(open("naive_bayes_model.pkl", "rb"))
svm_model = pickle.load(open("svm_model.pkl", "rb"))

# Assume label encoders are also saved (for occupation, location, etc.)
occupation_encoder = pickle.load(open("le_occupation.pkl", "rb"))
location_encoder = pickle.load(open("le_location.pkl", "rb"))
scheme_encoder = pickle.load(open("le_scheme.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    age = int(request.form["age"])
    occupation = request.form["occupation"]
    salary = int(request.form["salary"])
    location = request.form["location"]

    # Encode categorical variables
    occ_encoded = occupation_encoder.transform([occupation])[0]
    loc_encoded = location_encoder.transform([location])[0]

    features = np.array([[age, occ_encoded, salary, loc_encoded]])

    # Get NB predictions (top 5)
    probs = nb_model.predict_proba(features)[0]
    top_indices = np.argsort(probs)[-5:][::-1]
    nb_predictions = [nb_model.classes_[i] for i in top_indices]

    # Final SVM prediction
    final_scheme = svm_model.predict([nb_predictions])[0]

    return f"<h1>Hi {name}, we recommend: <b>{final_scheme}</b></h1>"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
