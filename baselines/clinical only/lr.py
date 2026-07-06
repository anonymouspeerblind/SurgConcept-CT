import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train, y_train)

    y_pred_proba               = lr_model.predict_proba(X_test)[:, 1]
    fpr, tpr, roc_thresholds   = roc_curve(y_test, y_pred_proba)
    auc_value                  = roc_auc_score(y_test, y_pred_proba)
    upper_fpr_idx              = min(i for i, val in enumerate(fpr) if val >= 0.3)
    tar_at_far_103             = tpr[upper_fpr_idx]
    upper_fpr_idx              = min(i for i, val in enumerate(fpr) if val >= 0.2)
    tar_at_far_102             = tpr[upper_fpr_idx]

    print(f"AUC value: {auc_value}")
    print(f"TAR@FAR=0.3: {tar_at_far_103}")
    print(f"TAR@FAR=0.2: {tar_at_far_102}")

    prob_dict           = dict()
    prob_dict['target'] = y_test.tolist()
    prob_dict['pred']   = y_pred_proba.tolist()
    with open("Path to save probabilties jsons/lr.json", "w") as js:
        json.dump(prob_dict, js, indent = 4)