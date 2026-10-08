"""
Model training and serialization pipeline for GenAI Language Detector.
Extracts stylometric features, trains a calibrated classifier,
evaluates cross-validation performance, and serializes artifacts.
"""

import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

from .dataset import TRAINING_DATA
from .features import extract_features, get_feature_vector, FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_artifact.joblib")

def train_and_export_model():
    print("[*] Extracting linguistic & stylometric features from dataset...")
    X_list = []
    y_list = []
    
    for item in TRAINING_DATA:
        feat_dict = extract_features(item["text"])
        vec = get_feature_vector(feat_dict)
        X_list.append(vec)
        y_list.append(item["label"])
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    print(f"[*] Dataset size: {len(X)} samples, {X.shape[1]} features.")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train calibrated logistic regression model
    base_clf = LogisticRegression(C=1.2, max_iter=2000, random_state=42, class_weight="balanced")
    
    # Cross validation check
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    cv_scores = cross_val_score(base_clf, X_scaled, y, cv=cv, scoring="roc_auc")
    print(f"[*] 4-Fold CV ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # Fit base classifier
    base_clf.fit(X_scaled, y)
    
    # Print feature weights for transparency
    print("[*] Top Feature Coefficients:")
    coefs = base_clf.coef_[0]
    for name, coef in sorted(zip(FEATURE_NAMES, coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"    - {name:22s}: {coef:+.4f}")
        
    model_payload = {
        "scaler": scaler,
        "classifier": base_clf,
        "feature_names": FEATURE_NAMES,
        "version": "1.0.0",
        "thresholds": {
            "ai_threshold": 0.65,
            "human_threshold": 0.35,
            "min_words_reliable": 30
        }
    }
    
    joblib.dump(model_payload, MODEL_PATH)
    print(f"[OK] Model successfully serialized to: {MODEL_PATH}")
    return model_payload

if __name__ == "__main__":
    train_and_export_model()
