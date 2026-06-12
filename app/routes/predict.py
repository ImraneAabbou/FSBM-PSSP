from flask import request, jsonify, render_template
from app.routes import bp
from app.utils.model_utils import predict, plot_prediction
import numpy as np


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    if not data or "sequence" not in data:
        return jsonify({"error": "Missing 'sequence' in request body"}), 400

    seq = data["sequence"].strip().upper()
    if len(seq) < 5:
        return jsonify({"error": "Sequence must be at least 5 amino acids"}), 400

    valid = "".join(c for c in seq if c in "ACDEFGHIKLMNPQRSTVWY")
    if len(valid) < 5:
        return jsonify({"error": "Invalid characters — use only standard 20 amino acids"}), 400

    pred_seq, probs = predict(valid)
    plot_b64 = plot_prediction(valid[:len(pred_seq)], probs)

    return jsonify({
        "prediction": pred_seq,
        "plot": plot_b64,
        "probs": probs.tolist()
    })
