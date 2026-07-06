import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import json
import argparse
import plotly.express as px
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
from dataloader import SurgConceptDataset
from model import SurgConceptCPTModel
from extracting_CT_feats import Extract_features

def tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    valid = fpr <= target_fpr
    if valid.sum() == 0:
        return 0.0
    return float(tpr[valid].max())

def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        if len(x) > 0 and isinstance(x[0], str):
            return x
        return [move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    return x

def fwdpass_by_stage(model, batch, stage, extract_ct):
    clinical_data = batch["clinical"]
    patches       = extract_ct.return_pooled_features(batch)
    ct_patch_mask = batch.get("ct_patch_mask", None)
    h_clinical    = model.clinical_encoder(clinical_data)
    if stage == 'full':
        t_ct        = model.ct_projector(patches)
        h_ct, _     = model.ct_attention_layer(h_clinical, t_ct, ct_patch_mask)
    else:
        h_ct        = torch.zeros_like(h_clinical)

    h_multi          = torch.cat([h_clinical, h_ct], dim=-1)
    concept_logits   = model.concept_predict_layer(h_multi)
    concept_probs    = torch.sigmoid(concept_logits)
    h_fusion         = model.fusion_mlp(h_multi)
    concept_logit    = model.concept_linear_layer(concept_probs)
    hidden_logit     = model.hidden_linear_layer(h_fusion)
    risk_logit       = concept_logit + hidden_logit
    risk_prob        = torch.sigmoid(risk_logit)

    return {"risk_logit": risk_logit.squeeze(-1), "concept_probs": concept_probs, "final_concept_logit": concept_logit.squeeze(-1), 
            "concept_pred_logits": concept_logits, "h_fusion": h_fusion, "risk_prob": risk_prob.squeeze(-1)}

def testing(model, test_loader, device, stage, extract_ct):
    model.eval()
    all_probs, all_targets = list(), list()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch    = move_to_device(batch, device)
            targets  = batch["target"].float()
            concepts = batch["concepts"].float()
            outputs  = fwdpass_by_stage(model, batch, stage, extract_ct)
            all_probs.append(outputs["risk_prob"].detach().cpu())
            all_targets.append(targets.detach().cpu())
    metrics                  = dict()
    all_probs                = np.concatenate(all_probs, axis=0)
    all_targets              = np.concatenate(all_targets, axis=0)
    fpr, tpr, roc_thresholds = roc_curve(all_targets, all_probs)
    youden_index             = tpr - fpr
    max_index                = np.argmax(youden_index)
    optimal_threshold        = roc_thresholds[max_index]
    auc_value                = roc_auc_score(all_targets, all_probs)
    tar_at_far_103           = tpr_at_fpr(all_targets, all_probs, 0.3)
    tar_at_far_102           = tpr_at_fpr(all_targets, all_probs, 0.2)
    metrics["AUC"]           = auc_value
    metrics["TPR@FPR=0.2"]   = tar_at_far_102
    metrics["TPR@FPR=0.3"]   = tar_at_far_103
    metrics["mean_prob"]     = all_probs.mean().item()
    metrics["positive_rate"] = all_targets.mean().item()
    return metrics

if __name__ == "__main__":
    ckpt_path    = "Trained checkpoint path"
    device       = torch.device("cuda")
    test_data    = SurgConceptDataset("test", load_ct = True)
    test_loader  = DataLoader(test_data, batch_size = 8, shuffle = False, num_workers = 0)
    model        = SurgConceptCPTModel(256).to(device)
    extract_ct   = Extract_features(device)
    ckpt         = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    test_metrics = testing(model, test_loader, device, "full", extract_ct)
    print(f"AUCROC: {test_metrics['AUC']}")
    print(f"TAR@FAR=0.2: {test_metrics['TPR@FPR=0.2']}")
    print(f"TAR@FAR=0.3: {test_metrics['TPR@FPR=0.3']}")