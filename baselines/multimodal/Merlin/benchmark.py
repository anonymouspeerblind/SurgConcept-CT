import os, json, argparse, warnings, torch
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from merlin import Merlin
from merlin.data import DataLoader as MerlinDataLoader
from dataloader import SurgConceptDataset
warnings.filterwarnings("ignore")

def tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    valid = fpr <= target_fpr
    if valid.sum() == 0:
        return 0.0
    return float(tpr[valid].max())

def patch_merlin_image_transforms():
    from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
                                Orientationd,
                                Spacingd,
                                ScaleIntensityRanged,
                                SpatialPadd,
                                CenterSpatialCropd,
                                ToTensord,
                                Lambdad)
    import merlin.data.monai_transforms as merlin_transforms
    import merlin.data.dataloaders as merlin_dataloaders
    def force_single_channel(x):
        if x.ndim == 4:
            if x.shape[0] > 1:
                return x[:1]
            return x
        if x.ndim == 3:
            return x[None, ...]
    MerlinSafeImageTransforms          = Compose([LoadImaged(keys=["image"]),
                                            EnsureChannelFirstd(keys=["image"]),
                                            Lambdad(keys=["image"], func=force_single_channel),
                                            Orientationd(keys=["image"], axcodes="RAS"),
                                            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
                                            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
                                            SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
                                            CenterSpatialCropd(keys=["image"], roi_size=[224, 224, 160]),
                                            ToTensord(keys=["image"])])
    merlin_transforms.ImageTransforms  = MerlinSafeImageTransforms
    merlin_dataloaders.ImageTransforms = MerlinSafeImageTransforms

def dataset_to_merlin_datalist(dataset):
    datalist, case_ids, targets = list(), list(), list()
    for i in range(len(dataset)):
        item    = dataset[i]
        case_id = item["case_id"]
        ct_path = item["ct_path"]
        summary = item["summary"]
        target  = float(item["target"].item())
        datalist.append({"image": ct_path, "text": summary, "case_id": case_id})
        case_ids.append(case_id)
        targets.append(target)
    return datalist, np.asarray(case_ids), np.asarray(targets, dtype=np.float32)

@torch.no_grad()
def extract_merlin_features_for_split(split, mode, include_phenotypes, cache_root, batch_size, num_workers, device, recompute_features=False):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_tag = "safe1ch_v1"
    cache_file = cache_root / (f"{split}_{mode}_pheno{int(include_phenotypes)}_{cache_tag}.npz")
    if cache_file.exists() and not recompute_features:
        arr = np.load(cache_file, allow_pickle=True)
        return {"case_ids": arr["case_ids"], "targets": arr["targets"], "features": arr["features"]}
    dataset = SurgConceptDataset(split)
    datalist, case_ids, targets = dataset_to_merlin_datalist(dataset)
    merlin_cache_dir = str(cache_root / f"merlin_preprocess_cache_{split}_{cache_tag}")
    loader = MerlinDataLoader(datalist=datalist, cache_dir=merlin_cache_dir, batchsize=batch_size, shuffle=False, num_workers=num_workers)
    if mode == "image_text":
        model = Merlin().to(device)
    elif mode == "image_only":
        model = Merlin(ImageEmbedding=True).to(device)
    model.eval()
    features, n_seen = list(), 0
    for batch in tqdm(loader, desc=f"Extracting Merlin features: {split}"):
        image = batch["image"]
        image = image.to(device)
        if mode == "image_text":
            outputs   = model(image, batch["text"])
            image_emb = outputs[0]      # [B, 512]
            phenotype = outputs[1]      # [B, 1692]
            text_emb  = outputs[2]       # [B, 512]
            pieces    = [image_emb, text_emb]
            if include_phenotypes:
                pieces.append(phenotype)
            feat = torch.cat(pieces, dim=1)
        else:
            outputs = model(image)
            feat = outputs[0]           # [B, 2048]
        feat_np = feat.detach().cpu().float().numpy()
        features.append(feat_np)
        n_seen += feat_np.shape[0]
    features = np.concatenate(features, axis=0)
    np.savez_compressed(cache_file, case_ids=case_ids, targets=targets, features=features)
    return {"case_ids": case_ids, "targets": targets, "features": features}

class RiskMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, hidden_dim // 2),
                                    nn.LayerNorm(hidden_dim // 2),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_dim // 2, 1))
    def forward(self, x):
        return self.net(x).squeeze(1)

def preprocess_features(X_train, X_val, X_test):
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_train = imputer.fit_transform(X_train)
    X_val   = imputer.transform(X_val)
    X_test  = imputer.transform(X_test)
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    return (X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32), imputer, scaler)

def make_tensor_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())

@torch.no_grad()
def predict_mlp(model, X, device):
    model.eval()
    ds     = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    probs  = list()
    for xb_tuple in loader:
        xb     = xb_tuple[0].to(device)
        logits = model(xb)
        prob   = torch.sigmoid(logits)
        probs.append(prob.detach().cpu().numpy())
    return np.concatenate(probs, axis=0)

def train_mlp_classifier(X_train, y_train, X_val, y_val, X_test, y_test, device, hidden_dim=256, dropout=0.25, lr=1e-4, weight_decay=1e-6, batch_size=128, max_epochs=200, patience=50, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    X_train, X_val, X_test, imputer, scaler = preprocess_features(X_train, X_val, X_test)
    train_loader = make_tensor_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    model        = RiskMLP(in_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    n_pos        = float(np.sum(y_train == 1))
    n_neg        = float(np.sum(y_train == 0))
    pos_weight   = n_neg / n_pos
    loss_fn      = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)
    best_val_auc, best_state = -1.0, None
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = list()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_prob = predict_mlp(model, X_val, device=device)
        val_auc  = roc_auc_score(y_val, val_prob)
        scheduler.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {"epoch": epoch, "model": model.state_dict(), "val_auc": val_auc}
        print(f"Epoch {epoch:03d} | "
            f"loss={np.mean(losses):.5f} | "
            f"val_auc={val_auc:.5f} | "
            f"best_val_auc={best_val_auc:.5f}")
    model.load_state_dict(best_state["model"])
    val_prob  = predict_mlp(model, X_val, device=device)
    test_prob = predict_mlp(model, X_test, device=device)
    results = {"best_epoch": int(best_state["epoch"]),
                "best_val_auc_during_training": float(best_state["val_auc"]),
                "val_auc": float(roc_auc_score(y_val, val_prob)),
                "test_auc": float(roc_auc_score(y_test, test_prob)),
                "test_tpr_at_fpr_0.2": float(tpr_at_fpr(y_test, test_prob, 0.2)),
                "test_tpr_at_fpr_0.3": float(tpr_at_fpr(y_test, test_prob, 0.3)),,
                "pos_weight": float(pos_weight)}
    artifacts = {"model": model,
                "imputer": imputer,
                "scaler": scaler,
                "val_prob": val_prob,
                "test_prob": test_prob,
                "test_target": y_test,
                "results": results}
    return artifacts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image_text", "image_only"], help="image_text uses CT + textual summary embeddings; image_only uses 2048-D CT embeddings.")
    parser.add_argument("--include_phenotypes", action="store_true", help="Only for image_text mode: append Merlin's 1692 phenotype predictions.")
    parser.add_argument("--cache_root", default="", help="Path to merlin's cache folder")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.25)
    parser.add_argument("--mlp_lr", type=float, default=1e-5, help="Learning rate for MLP")
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-6)
    parser.add_argument("--mlp_batch_size", type=int, default=256, help="Batch size for MLP")
    parser.add_argument("--mlp_epochs", type=int, default=200)
    parser.add_argument("--mlp_patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompute_features", action="store_true", help="Ignore saved .npz feature cache and recompute Merlin features.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    patch_merlin_image_transforms()

    train   = extract_merlin_features_for_split(split="train", mode=args.mode,
                                                include_phenotypes=args.include_phenotypes,
                                                cache_root=args.cache_root,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                device=device,
                                                recompute_features=args.recompute_features)
    test    = extract_merlin_features_for_split(split="test", mode=args.mode,
                                            include_phenotypes=args.include_phenotypes,
                                            cache_root=args.cache_root,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            device=device,
                                            recompute_features=args.recompute_features)

    val     = extract_merlin_features_for_split(split="val", mode=args.mode,
                                            include_phenotypes=args.include_phenotypes,
                                            cache_root=args.cache_root,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            device=device,
                                            recompute_features=args.recompute_features)
    X_train = train["features"]
    y_train = train["targets"].astype(int)
    X_test  = test["features"]
    y_test  = test["targets"].astype(int)
    X_val   = val["features"]
    y_val   = val["targets"].astype(int)

    mlp_artifacts = train_mlp_classifier(X_train=X_train, y_train=y_train,
                                        X_val=X_val, y_val=y_val,
                                        X_test=X_test, y_test=y_test,
                                        device=device, hidden_dim=args.mlp_hidden_dim,
                                        dropout=args.mlp_dropout, lr=args.mlp_lr,
                                        weight_decay=args.mlp_weight_decay, batch_size=args.mlp_batch_size,
                                        max_epochs=args.mlp_epochs, patience=args.mlp_patience, seed=args.seed)
    
    prob_dict           = dict()
    prob_dict['target'] = mlp_artifacts['test_target'].tolist()
    prob_dict['pred']   = mlp_artifacts['test_prob'].tolist()
    with open("Path to save probabilities jsons/Merlin(CT+Text).json", "w") as js:
        json.dump(prob_dict, js, indent = 4)
    print(f"Test AUC: {mlp_artifacts['results']['test_auc']}")
    print(f"Test TPR@FPR=0.2: {mlp_artifacts['results']['test_tpr_at_fpr_0.2']}")
    print(f"Test TPR@FPR=0.3: {mlp_artifacts['results']['test_tpr_at_fpr_0.3']}")

if __name__ == "__main__":
    main()