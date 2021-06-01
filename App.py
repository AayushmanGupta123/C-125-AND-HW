from flask import Flask,jsonify,request
from Digit import get_pred
from Alphabet import get_alpha
app = Flask(__name__)
@app.route('/')
def index():
    return "Welcome To My Home Page"
@app.route("/predict-digit",methods = ['POST'])
def predict_data():
    image = request.files.get("digit3.jpeg")
    prediction = get_pred(image)
    return jsonify ({"prediction":prediction}),200

@app.route("/predict-alpha",methods = ['POST'])
def predict_data():
    image = request.files.get("digit3.jpeg")
    prediction = get_pred(image)
    return jsonify ({"prediction":prediction}),200

if __name__ == "__main__":
    app.run(debug = True)