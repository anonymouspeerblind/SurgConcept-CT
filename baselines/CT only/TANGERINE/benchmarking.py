import os, sys, json, argparse, warnings, torch
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from dataloader import SurgConceptDataset
warnings.filterwarnings("ignore")

def tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    valid = fpr <= target_fpr
    if valid.sum() == 0:
        return 0.0
    return float(tpr[valid].max())

def load_ct_as_dhw(ct_path, volume_index=0):
    img = sitk.ReadImage(str(ct_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        if arr.shape[1] > 20 and arr.shape[2] >= 128 and arr.shape[3] >= 128:
            return arr[volume_index]
        else:
            return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3])

def preprocess_ct_for_tangerine(ct_path):
    hu_min = -1000.0
    hu_max = 400.0
    arr    = load_ct_as_dhw(ct_path)
    arr    = np.clip(arr, hu_min, hu_max)
    arr    = (arr - hu_min) / (hu_max - hu_min + 1e-8)
    arr    = arr.astype(np.float32)
    x      = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    x      = F.interpolate(x, size=(256, 256, 256), mode="trilinear", align_corners=False)
    x      = x.squeeze(0)
    return x.contiguous()

class TangerineFeatureDataset(Dataset):
    def __init__(self):
        self.base = SurgConceptDataset(split)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        item    = self.base[idx]
        case_id = item["case_id"]
        ct_path = item["ct_path"]
        target  = float(item["target"].item())
        try:
            image = preprocess_ct_for_tangerine(ct_path)
        except Exception as e:
            raise RuntimeError(f"Failed preprocessing case_id={case_id}, ct_path={ct_path}") from e
        return {"case_id": case_id, "image": image, "target": torch.tensor(target, dtype=torch.float32)}

def tangerine_collate_fn(batch):
    case_ids = [b["case_id"] for b in batch]
    images   = torch.stack([b["image"] for b in batch], dim=0)
    targets  = torch.stack([b["target"] for b in batch], dim=0)
    return {"case_id": case_ids, "image": images, "target": targets}

def strip_prefix_if_present(state_dict, prefix):
    out = dict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out

def get_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model", "state_dict", "model_state_dict", "checkpoint_model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint

def load_tangerine_encoder(repo_root, checkpoint_path, device):
    repo_root = Path(repo_root).resolve()
    sys.path.insert(0, str(repo_root))
    import models_vit
    model       = models_vit.vit_large_patch16_yo(img_size=256, num_classes=1, global_pool=True)
    checkpoint  = torch.load(checkpoint_path, map_location="cpu")
    state_dict  = get_checkpoint_state_dict(checkpoint)
    state_dict  = strip_prefix_if_present(state_dict, "module.")
    state_dict  = strip_prefix_if_present(state_dict, "backbone.")
    remove_keys = list()
    for k in state_dict.keys():
        if k.startswith("head."):
            remove_keys.append(k)
    for k in remove_keys:
        del state_dict[k]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

@torch.no_grad()
def extract_tangerine_embedding(model, image):
    feat = model.forward_features(image)
    if hasattr(model, "fc_norm"):
        feat = model.fc_norm(feat)
    return feat

@torch.no_grad()
def extract_tangerine_features_for_split(split, repo_root, checkpoint_path, cache_root, batch_size, num_workers, device, image_size, hu_min, hu_max, recompute_features=False):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    image_tag  = "x".join(str(x) for x in image_size)
    cache_file = cache_root / f"{split}_tangerine_img{image_tag}_features.npz"
    if cache_file.exists() and not recompute_features:
        arr = np.load(cache_file, allow_pickle=True)
        return {"case_ids": arr["case_ids"], "targets": arr["targets"], "features": arr["features"]}
    dataset = TangerineFeatureDataset()
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=tangerine_collate_fn)
    model   = load_tangerine_encoder(repo_root, checkpoint_path, device)
    all_features, all_targets, all_case_ids, amp_dtype = list(), list(), list(), torch.bfloat16
    for batch in tqdm(loader):
        image = batch["image"].to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            feat = extract_tangerine_embedding(model, image).detach().cpu().float().numpy()
        all_features.append(feat)
        all_targets.extend(batch["target"].cpu().numpy().astype(np.float32).tolist())
        all_case_ids.extend(batch["case_id"])
    features = np.concatenate(all_features, axis=0)
    targets  = np.asarray(all_targets, dtype=np.float32)
    case_ids = np.asarray(all_case_ids)
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
    if torch.cuda.is_available():
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
    parser.add_argument("--tangerine_repo", default="", help="Path of cloned github repo: https://github.com/niccolo246/3D-MAE-MedImaging")
    parser.add_argument("--checkpoint_path", default="", help="Path to downloaded trained tangerine checkpoint: https://zenodo.org/records/18835750/files/tangerine.pth?download=1")
    parser.add_argument("--cache_root", default="", help="Path to cache folder for TANGERINE")
    parser.add_argument("--image_size", type=int, nargs=3, default=[256, 256, 256], help="target size for images before input to TANGERINE")
    parser.add_argument("--hu_min", type=float, default=-1000.0)
    parser.add_argument("--hu_max", type=float, default=400.0)
    parser.add_argument("--extract_batch_size", type=int, default=16, help="Batch size to extract from TANGERINE")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.25)
    parser.add_argument("--mlp_lr", type=float, default=1e-5, help="Learning rate for MLP")
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-6)
    parser.add_argument("--mlp_batch_size", type=int, default=256, help="Batch size for MLP")
    parser.add_argument("--mlp_epochs", type=int, default=200)
    parser.add_argument("--mlp_patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompute_features", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = tuple(args.image_size)

    train = extract_tangerine_features_for_split(split="train", repo_root=args.tangerine_repo, checkpoint_path=args.checkpoint_path,
                                                cache_root=args.cache_root, batch_size=args.extract_batch_size,
                                                num_workers=args.num_workers, device=device,
                                                image_size=image_size, hu_min=args.hu_min,
                                                hu_max=args.hu_max, recompute_features=args.recompute_features)
    val   = extract_tangerine_features_for_split(split="val", repo_root=args.tangerine_repo, checkpoint_path=args.checkpoint_path,
                                                cache_root=args.cache_root, batch_size=args.extract_batch_size,
                                                num_workers=args.num_workers, device=device,
                                                image_size=image_size, hu_min=args.hu_min,
                                                hu_max=args.hu_max, recompute_features=args.recompute_features)
    test  = extract_tangerine_features_for_split(split="test", repo_root=args.tangerine_repo, checkpoint_path=args.checkpoint_path, 
                                                cache_root=args.cache_root, batch_size=args.extract_batch_size,
                                                num_workers=args.num_workers, device=device,
                                                image_size=image_size, hu_min=args.hu_min,
                                                hu_max=args.hu_max, recompute_features=args.recompute_features)
    X_train = train["features"]
    y_train = train["targets"].astype(int)
    X_val   = val["features"]
    y_val   = val["targets"].astype(int)
    X_test  = test["features"]
    y_test  = test["targets"].astype(int)

    mlp_artifacts = train_mlp_classifier(X_train=X_train, y_train=y_train,
                                        X_val=X_val, y_val=y_val,
                                        X_test=X_test, y_test=y_test,
                                        device=device, hidden_dim=args.mlp_hidden_dim,
                                        dropout=args.mlp_dropout, lr=args.mlp_lr,
                                        weight_decay=args.mlp_weight_decay, batch_size=args.mlp_batch_size,
                                        max_epochs=args.mlp_epochs, patience=args.mlp_patience, seed=args.seed)
    
    prob_dict = dict()
    prob_dict['target'] = mlp_artifacts['test_target'].tolist()
    prob_dict['pred']   = mlp_artifacts['test_prob'].tolist()
    with open("Path to save probabilities jsons/tangerine.json", "w") as js:
        json.dump(prob_dict, js, indent = 4)
    print(f"Test AUC: {mlp_artifacts['results']['test_auc']}")
    print(f"Test TPR@FPR=0.2: {mlp_artifacts['results']['test_tpr_at_fpr_0.2']}")
    print(f"Test TPR@FPR=0.3: {mlp_artifacts['results']['test_tpr_at_fpr_0.3']}")

if __name__ == "__main__":
    main()