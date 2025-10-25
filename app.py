from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from io import BytesIO
import warnings
import pickle
from model_class import ExoplanetEnsemble  # ✅ Import from separate module

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

FEATURE_NAMES = [
    'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_impact',
    'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag'
]

# Load the ensemble model safely
model_path = "ensemble_model.pkl"
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✅ Ensemble model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading ensemble model: {e}")
    model = None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"healthy", "model_loaded": model is not None})

# ─── Keep your /predict, /predict_csv, /predict_csv_stats routes unchanged ───

if __name__=="__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
