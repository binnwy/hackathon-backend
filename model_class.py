# model_class.py
import numpy as np
import pandas as pd

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
        if self.feature_names is not None and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        X_scaled_dnn = self.scaler_dnn.transform(X)
        X_scaled_svm = self.scaler_svm.transform(X)
        pred_dnn = self.model_dnn.predict(X_scaled_dnn, verbose=0)
        if hasattr(self.model_svm, 'predict_proba'):
            pred_svm = self.model_svm.predict_proba(X_scaled_svm)
        else:
            decision = self.model_svm.decision_function(X_scaled_svm)
            if len(decision.shape)==1:
                proba_positive = 1 / (1 + np.exp(-decision))
                pred_svm = np.column_stack([1-proba_positive, proba_positive])
            else:
                exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                pred_svm = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        pred_dt = self.model_dt.predict_proba(X)
        pred_rf = self.model_rf.predict_proba(X)
        pred_xgb = self.model_xgb.predict_proba(X)
        if pred_dnn.shape[1]==1:
            pred_dnn = np.hstack([1-pred_dnn, pred_dnn])
        ensemble_proba = (
            self.weights[0]*pred_dnn +
            self.weights[1]*pred_svm +
            self.weights[2]*pred_dt +
            self.weights[3]*pred_rf +
            self.weights[4]*pred_xgb
        )
        return ensemble_proba

    def predict(self, X, threshold=0.5):
        if self.feature_names is not None and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        proba = self.predict_proba(X)
        if proba.shape[1]==2:
            return (proba[:,1]>=threshold).astype(int)
        else:
            return np.argmax(proba, axis=1)

    def predict_with_confidence(self, X):
        if self.feature_names is not None and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        X_scaled_dnn = self.scaler_dnn.transform(X)
        X_scaled_svm = self.scaler_svm.transform(X)
        pred_dnn = self.model_dnn.predict(X_scaled_dnn, verbose=0)
        if hasattr(self.model_svm, 'predict_proba'):
            pred_svm = self.model_svm.predict_proba(X_scaled_svm)
        else:
            decision = self.model_svm.decision_function(X_scaled_svm)
            if len(decision.shape)==1:
                proba_positive = 1 / (1 + np.exp(-decision))
                pred_svm = np.column_stack([1-proba_positive, proba_positive])
            else:
                exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                pred_svm = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        pred_dt = self.model_dt.predict_proba(X)
        pred_rf = self.model_rf.predict_proba(X)
        pred_xgb = self.model_xgb.predict_proba(X)
        if pred_dnn.shape[1]==1:
            pred_dnn = np.hstack([1-pred_dnn, pred_dnn])
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
        return results if len(results)>1 else results[0]
