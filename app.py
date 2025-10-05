from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import warnings
from io import BytesIO
warnings.filterwarnings('ignore')

# ============================================================
# 1️⃣ Flask app initialization
# ============================================================
app = Flask(__name__)
CORS(app)

# ============================================================
# 2️⃣ CRITICAL: Define the ExoplanetEnsemble class EXACTLY as in Streamlit
# ============================================================
class ExoplanetEnsemble:
    """
    Ensemble model combining DNN, SVM, Decision Tree, Random Forest, and XGBoost
    """
    
    def __init__(self):
        self.model_dnn = None
        self.scaler_dnn = None
        self.model_svm = None
        self.scaler_svm = None
        self.model_dt = None
        self.model_rf = None
        self.model_xgb = None
        self.feature_names = None
        self.weights = None
    
    def predict_proba(self, X):
        """Predict probabilities using weighted soft voting"""
        if self.feature_names is not None and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        X_scaled_dnn = self.scaler_dnn.transform(X)
        X_scaled_svm = self.scaler_svm.transform(X)
        
        pred_dnn = self.model_dnn.predict(X_scaled_dnn, verbose=0)
        
        if hasattr(self.model_svm, 'predict_proba'):
            pred_svm = self.model_svm.predict_proba(X_scaled_svm)
        else:
            decision = self.model_svm.decision_function(X_scaled_svm)
            if len(decision.shape) == 1:
                proba_positive = 1 / (1 + np.exp(-decision))
                pred_svm = np.column_stack([1 - proba_positive, proba_positive])
            else:
                exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                pred_svm = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        pred_dt = self.model_dt.predict_proba(X)
        pred_rf = self.model_rf.predict_proba(X)
        pred_xgb = self.model_xgb.predict_proba(X)
        
        if pred_dnn.shape[1] == 1:
            pred_dnn = np.hstack([1 - pred_dnn, pred_dnn])
        
        ensemble_proba = (
            self.weights[0] * pred_dnn +
            self.weights[1] * pred_svm +
            self.weights[2] * pred_dt +
            self.weights[3] * pred_rf +
            self.weights[4] * pred_xgb
        )
        
        return ensemble_proba
    
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        if self.feature_names is not None and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return (proba[:, 1] >= threshold).astype(int)
        else:
            return np.argmax(proba, axis=1)
    
    def predict_with_confidence(self, X):
        """Predict with confidence scores and individual model predictions"""
        if self.feature_names is not None and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        X_scaled_dnn = self.scaler_dnn.transform(X)
        X_scaled_svm = self.scaler_svm.transform(X)
        
        pred_dnn = self.model_dnn.predict(X_scaled_dnn, verbose=0)
        
        if hasattr(self.model_svm, 'predict_proba'):
            pred_svm = self.model_svm.predict_proba(X_scaled_svm)
        else:
            decision = self.model_svm.decision_function(X_scaled_svm)
            if len(decision.shape) == 1:
                proba_positive = 1 / (1 + np.exp(-decision))
                pred_svm = np.column_stack([1 - proba_positive, proba_positive])
            else:
                exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                pred_svm = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        pred_dt = self.model_dt.predict_proba(X)
        pred_rf = self.model_rf.predict_proba(X)
        pred_xgb = self.model_xgb.predict_proba(X)
        
        if pred_dnn.shape[1] == 1:
            pred_dnn = np.hstack([1 - pred_dnn, pred_dnn])
        
        ensemble_proba = self.predict_proba(X)
        final_pred = np.argmax(ensemble_proba, axis=1)
        confidence = np.max(ensemble_proba, axis=1)
        
        results = []
        for i in range(len(X)):
            results.append({
                'prediction': int(final_pred[i]),
                'confidence': float(confidence[i]),
                'ensemble_proba': ensemble_proba[i].tolist(),
                'individual_predictions': {
                    'DNN': int(np.argmax(pred_dnn[i])),
                    'SVM': int(np.argmax(pred_svm[i])),
                    'Decision_Tree': int(np.argmax(pred_dt[i])),
                    'Random_Forest': int(np.argmax(pred_rf[i])),
                    'XGBoost': int(np.argmax(pred_xgb[i]))
                },
                'individual_probabilities': {
                    'DNN': pred_dnn[i].tolist(),
                    'SVM': pred_svm[i].tolist(),
                    'Decision_Tree': pred_dt[i].tolist(),
                    'Random_Forest': pred_rf[i].tolist(),
                    'XGBoost': pred_xgb[i].tolist()
                }
            })
        
        return results if len(results) > 1 else results[0]

# ============================================================
# 3️⃣ Feature names (must match the order used in training)
# ============================================================
FEATURE_NAMES = [
    'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_impact',
    'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag'
]

# ============================================================
# 4️⃣ Load the ensemble model
# ============================================================
model_path = "ensemble_model.pkl"

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✅ Ensemble model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading ensemble model: {e}")
    model = None

# ============================================================
# 5️⃣ Health check route
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

# ============================================================
# 6️⃣ Single prediction route
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
        
        # Extract prediction and confidence
        prediction = result['prediction']
        confidence = result['confidence'] * 100  # Convert to percentage
        
        # Binary classification: 1 = Planet, 0 or -1 = Not a Planet
        is_planet = prediction == 1

        if is_planet:
            return jsonify({
                "status": "positive",
                "message": "High probability of exoplanet detection",
                "details": "The analyzed parameters suggest strong exoplanet characteristics",
                "confidence": round(confidence, 1),
                "individual_predictions": result['individual_predictions'],
                "individual_probabilities": result['individual_probabilities']
            })
        else:
            return jsonify({
                "status": "negative",
                "message": "Low probability of exoplanet detection",
                "details": "The analyzed parameters do not indicate clear exoplanet signals",
                "confidence": round(confidence, 1),
                "individual_predictions": result['individual_predictions'],
                "individual_probabilities": result['individual_probabilities']
            })

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ============================================================
# 7️⃣ CSV prediction route - RETURNS CSV FILE
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

    # Check if all required features are present
    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        return jsonify({
            "error": f"Missing required features: {', '.join(missing_features)}"
        }), 400

    # Extract only the required features in the correct order
    X = df[FEATURE_NAMES]

    # Make predictions
    try:
        results = model.predict_with_confidence(X)
        
        # If single row, convert to list
        if not isinstance(results, list):
            results = [results]
        
        # Create output dataframe with original data + predictions
        output_df = df.copy()
        
        # Add prediction results
        output_df['prediction'] = [r['prediction'] for r in results]
        output_df['prediction_label'] = output_df['prediction'].apply(
            lambda x: 'EXOPLANET' if x == 1 else 'NOT_EXOPLANET'
        )
        output_df['confidence_percentage'] = [round(r['confidence'] * 100, 2) for r in results]
        
        # Add individual model predictions
        output_df['pred_DNN'] = [r['individual_predictions']['DNN'] for r in results]
        output_df['pred_SVM'] = [r['individual_predictions']['SVM'] for r in results]
        output_df['pred_DecisionTree'] = [r['individual_predictions']['Decision_Tree'] for r in results]
        output_df['pred_RandomForest'] = [r['individual_predictions']['Random_Forest'] for r in results]
        output_df['pred_XGBoost'] = [r['individual_predictions']['XGBoost'] for r in results]
        
        # Convert to CSV in memory
        output = BytesIO()
        output_df.to_csv(output, index=False)
        output.seek(0)
        
        # Return CSV file
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='exoplanet_predictions.csv'
        )
        
    except Exception as e:
        print(f"CSV Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ============================================================
# 8️⃣ CSV statistics route - RETURNS JSON STATS ONLY
# ============================================================
@app.route("/predict_csv_stats", methods=["POST"])
def predict_csv_stats():
    """Returns only statistics without the CSV file"""
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV: {e}"}), 400

    # Check if all required features are present
    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        return jsonify({
            "error": f"Missing required features: {', '.join(missing_features)}"
        }), 400

    # Extract only the required features in the correct order
    X = df[FEATURE_NAMES]

    # Make predictions
    try:
        results = model.predict_with_confidence(X)
        
        # If single row, convert to list
        if not isinstance(results, list):
            results = [results]
        
        # Count positive predictions
        positive_count = sum(1 for r in results if r['prediction'] == 1)
        
        # Calculate average confidence
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
        print(f"CSV Stats error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Stats calculation failed: {str(e)}"}), 500

# ============================================================
# 9️⃣ Run the app
# ============================================================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)