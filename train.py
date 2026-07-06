import os, random, argparse, json, torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from sklearn.metrics import roc_curve, roc_auc_score
import plotly.express as px
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import SurgConceptDataset
from model import SurgConceptCPTModel
from loss import Stage_wise_loss_fn
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

def set_gradient(module, flag):
    for p in module.parameters():
        p.requires_grad = flag

def defining_stage(model, stage):
    for p in model.parameters():
        p.requires_grad = False
    if stage == "clinical":
        set_gradient(model.clinical_encoder, True)
        set_gradient(model.concept_predict_layer, True)
        set_gradient(model.fusion_mlp, True)
        set_gradient(model.concept_linear_layer, True)
        set_gradient(model.hidden_linear_layer, True)
    elif stage == "full":
        set_gradient(model.clinical_encoder, True)
        set_gradient(model.ct_projector, True)
        set_gradient(model.ct_attention_layer, True)
        set_gradient(model.concept_predict_layer, True)
        set_gradient(model.fusion_mlp, True)
        set_gradient(model.concept_linear_layer, True)
        set_gradient(model.hidden_linear_layer, True)
    print(f"Configured stage: {stage}")

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

def train_per_epoch(model, train_loader, loss_fn, optimizer, device, stage, extract_ct):
    model.train()
    run_iter    = {"loss": 0.0,
                "risk_loss": 0.0,
                "concept_loss": 0.0,
                "fusion_loss": 0.0,
                "calibration_loss": 0.0}
    num_batches = 0
    for batch in tqdm(train_loader, desc=f"Train [{stage}]", leave=False):
        batch    = move_to_device(batch, device)
        targets  = batch["target"].float()
        concepts = batch["concepts"].float()
        optimizer.zero_grad(set_to_none=True)
        outputs  = fwdpass_by_stage(model, batch, stage, extract_ct)
        losses   = loss_fn(outputs, targets, concepts, stage)
        loss     = losses["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        for loss_type in run_iter:
            run_iter[loss_type] += float(losses[loss_type].item())
        num_batches += 1
        tqdm(train_loader, desc=f"Train [{stage}]", leave=False).set_postfix({"loss": run_iter["loss"] / num_batches, "risk": run_iter["risk_loss"] / num_batches, "concept": run_iter["concept_loss"] / num_batches})
    return {k: v / max(num_batches, 1) for k, v in run_iter.items()}

def validate_per_epoch(model, val_loader, loss_fn, device, stage, extract_ct):
    running_state = {"loss": 0.0,
                    "risk_loss": 0.0,
                    "concept_loss": 0.0,
                    "fusion_loss": 0.0,
                    "calibration_loss": 0.0}
    model.eval()
    all_probs, all_targets, num_batches = list(), list(), 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch    = move_to_device(batch, device)
            targets  = batch["target"].float()
            concepts = batch["concepts"].float()
            outputs  = fwdpass_by_stage(model, batch, stage, extract_ct)
            losses   = loss_fn(outputs, targets, concepts, stage)
            for k in running_state:
                running_state[k] += float(losses[k].item())
            all_probs.append(outputs["risk_prob"].detach().cpu())
            all_targets.append(targets.detach().cpu())
            num_batches += 1
    metrics                  = {k: v / max(num_batches, 1) for k, v in running_state.items()}
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

def running_stages(args, model, train_loader, val_loader, device, stage, epochs, lr, weight_decay, output_dir, log_dir, extract_ct):
    stage_log_dir = f"{log_dir}/{stage}"
    os.makedirs(stage_log_dir, exist_ok=True) # for stage-wise tensorboard
    log_writer    = SummaryWriter(stage_log_dir,comment = None)
    defining_stage(model, stage)
    params        = [p for p in model.parameters() if p.requires_grad]
    optimizer     = AdamW(params, lr = lr, weight_decay = weight_decay)
    loss_fn       = Stage_wise_loss_fn().to(device)

    best_val_auc, best_val_tp2, best_val_tp3, best_val_loss, best_path = 0, 0, 0, float("inf"), f"{output_dir}/{stage}"
    os.makedirs(best_path, exist_ok = True)
    for epoch in range(1, epochs + 1):
        print(f"Stage:{stage}: Epoch {epoch}/{epochs}")
        train_metrics = train_per_epoch(model, train_loader, loss_fn, optimizer, device, stage, extract_ct)
        val_metrics   = validate_per_epoch(model, val_loader, loss_fn, device, stage, extract_ct)
        
        print("Train:", train_metrics)
        print("Val:  ", val_metrics)
        log_writer.add_scalar('AUC/epoch',val_metrics["AUC"], epoch)
        log_writer.add_scalar('TAR@FAR0.2/epoch',val_metrics["TPR@FPR=0.2"], epoch)
        log_writer.add_scalar('TAR@FAR0.3/epoch',val_metrics["TPR@FPR=0.3"], epoch)
        
        if val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            torch.save({"stage": stage,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics}, f'{best_path}/auc_{args.remark}_{val_metrics["AUC"]}_{val_metrics["TPR@FPR=0.2"]}_{val_metrics["TPR@FPR=0.3"]}.pt')
            print("Saving best checkpoint for AUC")
        if val_metrics["TPR@FPR=0.2"] > best_val_tp2:
            best_val_tp2 = val_metrics["TPR@FPR=0.2"]
            torch.save({"stage": stage,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics}, f'{best_path}/tp2_{args.remark}_{val_metrics["AUC"]}_{val_metrics["TPR@FPR=0.2"]}_{val_metrics["TPR@FPR=0.3"]}.pt')
            print("Saving best checkpoint for TPR@FPR=0.2")
        if val_metrics["TPR@FPR=0.3"] > best_val_tp3:
            best_val_tp3 = val_metrics["TPR@FPR=0.3"]
            torch.save({"stage": stage,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics}, f'{best_path}/tp3_{args.remark}_{val_metrics["AUC"]}_{val_metrics["TPR@FPR=0.2"]}_{val_metrics["TPR@FPR=0.3"]}.pt')
            print("Saving best checkpoint for TPR@FPR=0.3")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({"stage": stage,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics}, f'{best_path}/loss_{args.remark}_{val_metrics["AUC"]}_{val_metrics["TPR@FPR=0.2"]}_{val_metrics["TPR@FPR=0.3"]}.pt')
            loading_ckpt_path = f'{best_path}/loss_{args.remark}_{val_metrics["AUC"]}_{val_metrics["TPR@FPR=0.2"]}_{val_metrics["TPR@FPR=0.3"]}.pt'
            print("Saving best checkpoint based on loss")
    log_writer.close()
    ckpt = torch.load(loading_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, default=4, help='batch size for training model')
    parser.add_argument("--val_batch_size", type=int, default=2, help='validation batch size')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="", help='directory to store trained checkpoints')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-ce", "--clinical_epochs", type=int, default=20, help='number of epochs for stage 1')
    parser.add_argument("-fulle", "--full_epochs", type=int, default=100, help='number of epochs for stage 2')
    parser.add_argument("-clr", "--clinical_lr", type=float, default=1e-4, help='learning rate for stage 1')
    parser.add_argument("-fulllr", "--full_lr", type=float, default=5e-5, help='learning rate for stage 2')
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--remark", type=str, required=True, help="extra remark for checkpoint's name")
    args = parser.parse_args()

    log_dir = f"Parent directory for tensorboard logs/{args.remark}"
    os.makedirs(log_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = args.output_dir
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset = SurgConceptDataset("train")
    train_loader  = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    val_dataset   = SurgConceptDataset("val")
    val_loader    = DataLoader(val_dataset, batch_size = args.val_batch_size, shuffle = False, num_workers = args.num_workers)
    model         = SurgConceptCPTModel(256).to(device)
    extract_ct    = Extract_features(device)
    model         = running_stages(args, model, train_loader, val_loader, device, "clinical", args.clinical_epochs, args.clinical_lr, args.weight_decay, output_dir, log_dir, extract_ct)
    train_dataset = SurgConceptDataset("train", load_ct = True)
    train_loader  = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    val_dataset   = SurgConceptDataset("val", load_ct = True)
    val_loader    = DataLoader(val_dataset, batch_size = args.val_batch_size, shuffle = False, num_workers = args.num_workers)
    model         = running_stages(args, model, train_loader, val_loader, device, "full", args.full_epochs, args.full_lr, args.weight_decay, output_dir, log_dir, extract_ct)

if __name__ == "__main__":
    main()