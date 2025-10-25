# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from io import BytesIO
import warnings
import dill  # ✅ use dill instead of pickle
from model_class import ExoplanetEnsemble  # Your ensemble class

warnings.filterwarnings('ignore')

# ============================================================
# 1️⃣ Flask app initialization
# ============================================================
app = Flask(__name__)
CORS(app)

# ============================================================
# 2️⃣ Feature names (must match training order)
# ============================================================
FEATURE_NAMES = [
    'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_impact',
    'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag'
]

# ============================================================
# 3️⃣ Load the ensemble model
# ============================================================
model_path = "ensemble_model.pkl"

try:
    with open(model_path, 'rb') as f:
        model = dill.load(f)  # ✅ use dill to load
    print("✅ Ensemble model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading ensemble model: {e}")
    model = None

# ============================================================
# 4️⃣ Health check route
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

# ============================================================
# 5️⃣ Single prediction route
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract features in order
        features = []
        for name in FEATURE_NAMES:
            if name not in data:
                return jsonify({"error": f"Missing feature: {name}"}), 400
            features.append(float(data[name]))

        X = np.array([features])

        # Make prediction with confidence
        result = model.predict_with_confidence(X)

        prediction = int(result['prediction'])
        confidence = float(result['confidence'] * 100)  # percentage

        is_planet = prediction == 1
        status = "positive" if is_planet else "negative"
        message = "High probability of exoplanet detection" if is_planet else "Low probability of exoplanet detection"
        details = "The analyzed parameters suggest strong exoplanet characteristics" if is_planet else "The analyzed parameters do not indicate clear exoplanet signals"

        return jsonify({
            "status": status,
            "message": message,
            "details": details,
            "confidence": round(confidence, 1),
            "individual_predictions": result['individual_predictions'],
            "individual_probabilities": result['individual_probabilities']
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ============================================================
# 6️⃣ CSV prediction route
# ============================================================
@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV: {e}"}), 400

    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

    X = df[FEATURE_NAMES]

    try:
        results = model.predict_with_confidence(X)
        if not isinstance(results, list):
            results = [results]

        output_df = df.copy()
        output_df['prediction'] = [r['prediction'] for r in results]
        output_df['prediction_label'] = output_df['prediction'].apply(lambda x: 'EXOPLANET' if x == 1 else 'NOT_EXOPLANET')
        output_df['confidence_percentage'] = [round(r['confidence'] * 100, 2) for r in results]
        output_df['pred_DNN'] = [r['individual_predictions']['DNN'] for r in results]
        output_df['pred_SVM'] = [r['individual_predictions']['SVM'] for r in results]
        output_df['pred_DecisionTree'] = [r['individual_predictions']['Decision_Tree'] for r in results]
        output_df['pred_RandomForest'] = [r['individual_predictions']['Random_Forest'] for r in results]
        output_df['pred_XGBoost'] = [r['individual_predictions']['XGBoost'] for r in results]

        output = BytesIO()
        output_df.to_csv(output, index=False)
        output.seek(0)

        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='exoplanet_predictions.csv')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"CSV Prediction failed: {str(e)}"}), 500

# ============================================================
# 7️⃣ CSV stats route
# ============================================================
@app.route("/predict_csv_stats", methods=["POST"])
def predict_csv_stats():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV: {e}"}), 400

    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

    X = df[FEATURE_NAMES]

    try:
        results = model.predict_with_confidence(X)
        if not isinstance(results, list):
            results = [results]

        positive_count = sum(1 for r in results if r['prediction'] == 1)
        avg_confidence = np.mean([r['confidence'] for r in results]) * 100

        return jsonify({
            "status": "success",
            "message": "Bulk analysis completed successfully",
            "details": f"Processed {len(results)} exoplanet candidates from CSV file",
            "confidence": round(avg_confidence, 1),
            "candidatesFound": positive_count,
            "totalProcessed": len(results)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"CSV stats failed: {str(e)}"}), 500

# ============================================================
# 8️⃣ Run the app
# ============================================================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
