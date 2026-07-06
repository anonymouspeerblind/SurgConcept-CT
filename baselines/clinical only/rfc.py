import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

if __name__ == "__main__":
    with open("Path to clinical features list", "r") as js:
        clinical_feat = json.load(js)['clinical_features']
    train_df = pd.read_csv("Path to train clinical features")
    test_df  = pd.read_csv("Path to test clinical features")

    X_train = train_df[clinical_feat]
    y_train = train_df["Target"]
    X_test  = test_df[clinical_feat]
    y_test  = test_df["Target"]

    rf_model = RandomForestClassifier(n_estimators=382, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred_proba_rf              = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, roc_thresholds     = roc_curve(y_test, y_pred_proba_rf)
    auc_value                    = roc_auc_score(y_test, y_pred_proba_rf)
    upper_fpr_idx                = min(i for i, val in enumerate(fpr) if val >= 0.3)
    tar_at_far_103               = tpr[upper_fpr_idx]
    upper_fpr_idx                = min(i for i, val in enumerate(fpr) if val >= 0.2)
    tar_at_far_102               = tpr[upper_fpr_idx]

    print(f"AUC value: {auc_value}")
    print(f"TAR@FAR=0.3: {tar_at_far_103}")
    print(f"TAR@FAR=0.2: {tar_at_far_102}")

    prob_dict           = dict()
    prob_dict['target'] = y_test.tolist()
    prob_dict['pred']   = y_pred_proba_rf.tolist()
    with open("Path to save probabilities jsons/rfc.json", "w") as js:
        json.dump(prob_dict, js, indent = 4)