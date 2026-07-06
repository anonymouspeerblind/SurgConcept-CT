import json, random
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import FTTransformerConfig

def preparing_data():
    with open("Path to clinical features list", "r") as js:
        clinical_feat = json.load(js).get("clinical_features", None)
    train_df         = pd.read_csv("Path to train clinical features")
    val_df           = pd.read_csv("Path to validation clinical features")
    test_df          = pd.read_csv("Path to test clinical features")
    target_col       = "Target"
    id_cols          = ["CASE_ID"]
    continuous_cols  = ["Age",
                        "BMI",
                        "FEV1 Predicted",
                        "DLCO Predicted",
                        "Pack-Years Of Cigarette Use"]
    categorical_cols = ["Tumor Size",
                        "Gender",
                        "Prior Cardiothoracic Surgery",
                        "Preoperative Chemo - Current Malignancy",
                        "Preoperative Thoracic Radiation Therapy",
                        "Cigarette Smoking",
                        "ECOG Score",
                        "ASA Classification",
                        "Clinical Staging - Lung Cancer - T",
                        "Clinical Staging - Lung Cancer - N",
                        "Clinical Staging - Lung Cancer - M",
                        "Procedure"]
    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.drop(columns=id_cols, inplace=True, errors="ignore")
        df[target_col] = df[target_col].astype(int)
        for col in categorical_cols:
            df[col] = df[col].astype("int64").astype("category")
        for col in continuous_cols:
            df[col] = df[col].astype("float32")
    return train_df, val_df, test_df

def configure():
    target_col        = "Target"
    id_cols           = ["CASE_ID"]
    continuous_cols   = ["Age",
                        "BMI",
                        "FEV1 Predicted",
                        "DLCO Predicted",
                        "Pack-Years Of Cigarette Use"]
    categorical_cols  = ["Tumor Size",
                        "Gender",
                        "Prior Cardiothoracic Surgery",
                        "Preoperative Chemo - Current Malignancy",
                        "Preoperative Thoracic Radiation Therapy",
                        "Cigarette Smoking",
                        "ECOG Score",
                        "ASA Classification",
                        "Clinical Staging - Lung Cancer - T",
                        "Clinical Staging - Lung Cancer - N",
                        "Clinical Staging - Lung Cancer - M",
                        "Procedure"]
    data_config      = DataConfig(target=[target_col], continuous_cols=continuous_cols, categorical_cols=categorical_cols)
    trainer_config   = TrainerConfig(batch_size=64,
                                    max_epochs=100,
                                    accelerator="gpu",
                                    devices=1,
                                    early_stopping="valid_loss",
                                    early_stopping_patience=10,
                                    checkpoints="valid_loss",
                                    load_best=True)
    optimizer_config = OptimizerConfig(optimizer="AdamW", optimizer_params={"weight_decay": 1e-4}, lr_scheduler=None)
    model_config     = FTTransformerConfig(task="classification",
                                            learning_rate=1e-3,
                                            input_embed_dim=32,
                                            num_heads=4,
                                            num_attn_blocks=3,
                                            attn_dropout=0.1,
                                            head="LinearHead",
                                            head_config={"layers": "64-32", "dropout": 0.1, "activation": "ReLU"})
    return data_config, trainer_config, optimizer_config, model_config

def metrics_calc(all_probs, all_targets):
    metrics                  = dict()
    all_probs                = np.array(all_probs)
    all_targets              = np.array(all_targets)
    fpr, tpr, roc_thresholds = roc_curve(all_targets, all_probs)
    youden_index             = tpr - fpr
    max_index                = np.argmax(youden_index)
    optimal_threshold        = roc_thresholds[max_index]
    auc_value                = roc_auc_score(all_targets, all_probs)
    upper_fpr_idx            = min(i for i, val in enumerate(fpr) if val >= 0.3)
    tar_at_far_103           = tpr[upper_fpr_idx]
    upper_fpr_idx            = min(i for i, val in enumerate(fpr) if val >= 0.2)
    tar_at_far_102           = tpr[upper_fpr_idx]
    metrics["AUC"]           = auc_value
    metrics["TPR@FPR=0.2"]   = tar_at_far_102
    metrics["TPR@FPR=0.3"]   = tar_at_far_103
    metrics["mean_prob"]     = all_probs.mean().item()
    metrics["positive_rate"] = all_targets.mean().item()
    prob_dict                = dict()
    prob_dict['target']      = all_targets.tolist()
    prob_dict['pred']        = all_probs.tolist()
    with open("Path to save probabilities json/fttransformer.json", "w") as js:
        json.dump(prob_dict, js, indent = 4)
    return metrics

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    data_config, trainer_config, optimizer_config, model_config = configure()
    train_df, val_df, test_df                                   = preparing_data()
    tabular_model                                               = TabularModel(data_config=data_config,
                                                                                model_config=model_config,
                                                                                optimizer_config=optimizer_config,
                                                                                trainer_config=trainer_config)
    tabular_model.fit(train=train_df, validation=val_df)
    target_col    = "Target"
    test_features = test_df.drop(columns=[target_col])
    pred_df       = tabular_model.predict(test_features)
    y_true        = test_df[target_col].values.tolist()
    y_prob        = pred_df["Target_1_probability"].values.tolist()
    metrics       = metrics_calc(y_prob, y_true)
    print(metrics)

if __name__ == "__main__":
    main()