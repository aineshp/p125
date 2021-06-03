from flask import Flask, jsonify, request
from c125 import  getprediction

app = Flask(__name__)

@app.route("/predict-letter", methods=["POST"])
def predict_data():
  image = request.files.get("letter")
  prediction = getprediction(image)
  return jsonify({
    "prediction": prediction
  }), 200

if __name__ == "__main__":
  app.run(debug=True)