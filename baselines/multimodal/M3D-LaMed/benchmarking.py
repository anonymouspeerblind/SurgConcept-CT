import os, re, json, argparse, warnings, torch
from pathlib import Path
import nibabel as nib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from dataloader import SurgConceptDataset
warnings.filterwarnings("ignore")

def tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    valid                = fpr <= target_fpr
    if valid.sum() == 0:
        return 0.0
    return float(tpr[valid].max())

def load_nifti_as_dhw(nifti_path, four_d_policy="first"):
    img = nib.load(str(nifti_path))
    arr = img.get_fdata(dtype=np.float32)
    if arr.ndim == 4:
        n_vols = arr.shape[-1]
        if n_vols == 1:
            arr = arr[..., 0]
        elif four_d_policy == "first":
            arr = arr[..., 0]
    arr = np.squeeze(arr)
    arr = np.transpose(arr, (2, 1, 0))
    return arr.astype(np.float32)

def preprocess_ct_for_m3d(nifti_path):
    arr = load_nifti_as_dhw(nifti_path, four_d_policy="first")
    arr = np.clip(arr, -1000.0, 400.0)
    arr = (arr + 1000.0) / (1400 + 1e-8)
    arr = arr.astype(np.float32)
    x   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    x   = F.interpolate(x, size = (32, 256, 256), mode = "trilinear", align_corners = False)
    x   = x.squeeze(0)
    return x.contiguous()

def build_ppc_prompt(summary):
    complications_lst = ["Adult Respiratory Distress Syndrome", "Pneumonia",
                        "Atelectasis Requiring Bronchoscopy", "Bronchopleural Fistula",
                        "Pneumothorax", "Air Leak Greater Than Five Days", "Tracheostomy",
                        "Unexpected Admission To ICU", "Empyema Requiring Treatment", "Initial Vent Support >48 Hours"]
    complication_text = "; ".join(complications_lst)
    prompt = (f"""You are given a preoperative chest CT volume and a textual summary of preoperative clinical variables for a lung cancer surgery patient.\n\n
        Clinical summary:\n{summary}\n\n 
        Task: create an internal representation useful for predicting whether the patient will develop any postoperative pulmonary complication after 
        lung cancer surgery. A positive postoperative pulmonary complication means 
        the patient develops at least one of the following: {complication_text}.\n\n
        Do not make treatment recommendations. Focus on preoperative risk factors and CT imaging findings relevant to postoperative pulmonary complications.""")
    return prompt

def build_m3d_input_text(summary, proj_out_num):
    image_tokens = "<im_patch>" * proj_out_num
    return image_tokens + build_ppc_prompt(summary)

class M3DFeatureDataset(Dataset):
    def __init__(self, split, proj_out_num = 256):
        self.base         = SurgConceptDataset(split)
        self.split        = split
        self.proj_out_num = proj_out_num
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        item    = self.base[idx]
        case_id = item["case_id"]
        summary = item["summary"]
        ct_path = item["ct_path"]
        target  = float(item["target"].item())
        image   = preprocess_ct_for_m3d(ct_path)
        text    = build_m3d_input_text(summary = summary, proj_out_num = self.proj_out_num)
        return {"case_id": case_id, "image": image, "text": text, "target": torch.tensor(target, dtype=torch.float32)}

def m3d_collate_fn(batch):
    case_ids = [b["case_id"] for b in batch]
    images   = torch.stack([b["image"] for b in batch], dim=0)
    texts    = [b["text"] for b in batch]
    targets  = torch.stack([b["target"] for b in batch], dim=0)
    return {"case_id": case_ids, "image": images, "text": texts, "target": targets}

def get_dtype(dtype_name):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32

@torch.no_grad()
def extract_m3d_lamed_features_for_split(split, model_name, tokenizer, model, cache_root, batch_size, num_workers, dtype_name, image_size, hu_min, hu_max, proj_out_num, max_length, recompute_features=False):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    image_tag      = "x".join(str(x) for x in image_size)
    safe_model_tag = re.sub(r"[^a-zA-Z0-9_\-]+", "_", model_name)
    cache_file     = cache_root / (f"{split}_{safe_model_tag}_m3dlamed_" f"img{image_tag}_proj{proj_out_num}_hiddenmean.npz")
    if cache_file.exists() and not recompute_features:
        arr = np.load(cache_file, allow_pickle=True)
        return {"case_ids": arr["case_ids"], "targets": arr["targets"], "features": arr["features"]}
    dataset      = M3DFeatureDataset(split=split, image_size=image_size, hu_min=hu_min, hu_max=hu_max, proj_out_num=proj_out_num)
    loader       = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=m3d_collate_fn)
    model_device = next(model.parameters()).device
    dtype        = get_dtype(dtype_name)
    all_features, all_targets, all_case_ids = list(), list(), list()
    for batch in tqdm(loader):
        images         = batch["image"].to(device=model_device, dtype=dtype)
        tokenized      = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids      = tokenized["input_ids"].to(model_device)
        attention_mask = tokenized["attention_mask"].to(model_device)
        outputs        = model(images=images, input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True, use_cache=False)
        last_hidden    = outputs.hidden_states[-1].float()
        if last_hidden.shape[1] == attention_mask.shape[1]:
            mask = attention_mask.unsqueeze(-1).float()
            feat = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            feat = last_hidden.mean(dim=1)
        feat = feat.detach().cpu().numpy().astype(np.float32)
        all_features.append(feat)
        all_targets.extend(batch["target"].cpu().numpy().astype(np.float32).tolist())
        all_case_ids.extend(batch["case_id"])
    features = np.concatenate(all_features, axis=0)
    targets  = np.asarray(all_targets, dtype=np.float32)
    case_ids = np.asarray(all_case_ids)
    np.savez_compressed(cache_file,
                        case_ids=case_ids,
                        targets=targets,
                        features=features)
    return {"case_ids": case_ids, "targets": targets, "features": features}

class RiskMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim = 256, dropout = 0.25):
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
    scaler = StandardScaler()
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
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

def train_mlp_classifier(X_train, y_train, X_val, y_val, X_test, y_test, device, hidden_dim, dropout, lr, weight_decay, batch_size, max_epochs, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    X_train, X_val, X_test, imputer, scaler = preprocess_features(X_train, X_val, X_test)
    train_loader                            = make_tensor_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    model                                   = RiskMLP(in_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    n_pos                                   = float(np.sum(y_train == 1))
    n_neg                                   = float(np.sum(y_train == 0))
    pos_weight                              = n_neg / n_pos
    loss_fn                                 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer                               = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler                               = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)
    best_val_auc, best_state                = -1.0, None
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
    model.load_state_dict(best_state["model"])
    val_prob    = predict_mlp(model, X_val, device=device)
    test_prob   = predict_mlp(model, X_test, device=device)
    test_target = y_test.tolist()

    results   = {"best_epoch": int(best_state["epoch"]),
                "best_val_auc_during_training": float(best_state["val_auc"]),
                "val_auc": float(roc_auc_score(y_val, val_prob)),
                "test_auc": float(roc_auc_score(y_test, test_prob)),
                "test_tpr_at_fpr_0.2": float(tpr_at_fpr(y_test, test_prob, 0.2)),
                "test_tpr_at_fpr_0.3": float(tpr_at_fpr(y_test, test_prob, 0.3)),
                "pos_weight": float(pos_weight)}
    artifacts = {"model": model, "imputer": imputer, "scaler": scaler, "val_prob": val_prob, "test_prob": test_prob, "test_target": test_target, "results": results}
    return artifacts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="GoodBaiBai88/M3D-LaMed-Llama-2-7B", help="Hugging Face model name/path.")
    parser.add_argument("--cache_root", default="", help="Cache for m3d_lamed")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference for M3D")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device_map_auto", action="store_true", help="Use HF device_map='auto'. Useful for larger models.")
    parser.add_argument("--image_size", type=int, nargs=3, default=[32, 256, 256], help="M3D-LaMed expected size: D H W = 32 256 256.")
    parser.add_argument("--hu_min", type=float, default=-1000.0, help="Max window HU")
    parser.add_argument("--hu_max", type=float, default=400.0, help="Min window HU")
    parser.add_argument("--proj_out_num", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.25)
    parser.add_argument("--mlp_lr", type=float, default=1e-3, help="Learning rate for MLP")
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-6)
    parser.add_argument("--mlp_batch_size", type=int, default=512, help="Batch size for MLP")
    parser.add_argument("--mlp_epochs", type=int, default=200, help="Max number of epochs for training MLP")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompute_features", action="store_true", help="Ignore cached M3D-LaMed features and recompute.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    dtype     = get_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512, padding_side="right", use_fast=False, trust_remote_code=True, cache_dir="")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.device_map_auto:
        m3d_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True, cache_dir="")
    else:
        m3d_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, trust_remote_code=True, cache_dir="")
        m3d_model = m3d_model.to(device)
    m3d_model.eval()
    for p in m3d_model.parameters():
        p.requires_grad = False

    image_size = tuple(args.image_size)
    train      = extract_m3d_lamed_features_for_split(split="train", model_name=args.model_name, tokenizer=tokenizer,
                                                        model=m3d_model, cache_root=args.cache_root, batch_size=args.batch_size,
                                                        num_workers=args.num_workers, dtype_name=args.dtype, image_size=image_size,
                                                        hu_min=args.hu_min, hu_max=args.hu_max, proj_out_num=args.proj_out_num,
                                                        max_length=args.max_length, recompute_features=args.recompute_features)
    val        = extract_m3d_lamed_features_for_split(split="val", model_name=args.model_name, tokenizer=tokenizer, model=m3d_model,
                                                        cache_root=args.cache_root, batch_size=args.batch_size, num_workers=args.num_workers,
                                                        dtype_name=args.dtype, image_size=image_size, hu_min=args.hu_min,
                                                        hu_max=args.hu_max, proj_out_num=args.proj_out_num, max_length=args.max_length, recompute_features=args.recompute_features)
    test       = extract_m3d_lamed_features_for_split(split="test", model_name=args.model_name, tokenizer=tokenizer,
                                                        model=m3d_model, cache_root=args.cache_root, batch_size=args.batch_size,
                                                        num_workers=args.num_workers, dtype_name=args.dtype, image_size=image_size,
                                                        hu_min=args.hu_min, hu_max=args.hu_max, proj_out_num=args.proj_out_num,
                                                        max_length=args.max_length, recompute_features=args.recompute_features)
    
    X_train       = train["features"]
    y_train       = train["targets"].astype(int)
    X_val         = val["features"]
    y_val         = val["targets"].astype(int)
    X_test        = test["features"]
    y_test        = test["targets"].astype(int)
    mlp_artifacts = train_mlp_classifier(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                                        X_test=X_test, y_test=y_test, device=device, hidden_dim=args.mlp_hidden_dim,
                                        dropout=args.mlp_dropout, lr=args.mlp_lr, weight_decay=args.mlp_weight_decay,
                                        batch_size=args.mlp_batch_size, max_epochs=args.mlp_epochs, seed=args.seed)
    
    prob_dict             = dict()
    prob_dict['pred']     = mlp_artifacts['test_prob'].tolist()
    prob_dict['target']   = mlp_artifacts['test_target']
    final_file_model_name = args.model_name.split("/")[-1]
    with open(f"Path to probabilities jsons/m3d_llamed_{final_file_model_name}.json", "w") as js:
        json.dump(prob_dict, js, indent = 4)
    print(f"Test AUC: {mlp_artifacts['results']['test_auc']}")
    print(f"Test TPR@FPR=0.2: {mlp_artifacts['results']['test_tpr_at_fpr_0.2']}")
    print(f"Test TPR@FPR=0.3: {mlp_artifacts['results']['test_tpr_at_fpr_0.3']}")

if __name__ == "__main__":
    main()