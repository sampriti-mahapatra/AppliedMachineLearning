import joblib
import os
from flask import Flask, request, jsonify
from score import score

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
model = joblib.load(MODEL_PATH)


@app.route('/score', methods=['POST'])
def score_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    threshold = data.get('threshold', 0.5)

    prediction, propensity = score(text, model, threshold)

    return jsonify({
        'prediction': int(prediction),
        'propensity': propensity
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
