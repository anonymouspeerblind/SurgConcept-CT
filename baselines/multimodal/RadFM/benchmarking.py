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
from transformers import LlamaTokenizer
from dataloader import SurgConceptDataset
warnings.filterwarnings("ignore")

def tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    valid = fpr <= target_fpr
    if valid.sum() == 0:
        return 0.0
    return float(tpr[valid].max())

def get_radfm_tokenizer(language_path):
    text_tokenizer       = LlamaTokenizer.from_pretrained(language_path, cache_dir="")
    image_padding_tokens = list()
    special_tokens       = {"additional_special_tokens": ["<image>", "</image>"]}
    for i in range(100):
        image_padding_token = ""
        for j in range(32):
            image_token = f"<image{i}_{j}>"
            image_padding_token += image_token
            special_tokens["additional_special_tokens"].append(image_token)
        image_padding_tokens.append(image_padding_token)
    text_tokenizer.add_special_tokens(special_tokens)
    text_tokenizer.pad_token_id = 0
    text_tokenizer.bos_token_id = 1
    text_tokenizer.eos_token_id = 2
    return text_tokenizer, image_padding_tokens

def build_radfm_prompt(summary):
    complications_lst = ["Adult Respiratory Distress Syndrome",
                        "Pneumonia",
                        "Atelectasis Requiring Bronchoscopy",
                        "Bronchopleural Fistula",
                        "Pneumothorax",
                        "Air Leak Greater Than Five Days",
                        "Tracheostomy",
                        "Unexpected Admission To ICU",
                        "Empyema Requiring Treatment",
                        "Initial Vent Support >48 Hours"]
    complication_text = "; ".join(complications_lst)
    prompt = (f"""You are given a preoperative chest CT volume and a textual summary of clinical variables for a lung cancer surgery patient.\n\n
        Clinical summary:\n{summary}\n\n
        The prediction target is whether the patient develops any postoperative pulmonary complication after lung cancer surgery. A positive event means 
        at least one of the following occurs: {complication_text}.\n\n
        Use the CT imaging information and the clinical summary to form an internal representation useful for postoperative complication risk prediction. 
        Do not provide treatment recommendations.""")
    return prompt

def load_ct_as_dhw(ct_path):
    img = sitk.ReadImage(str(ct_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    def select_from_axis(x, axis):
        n = x.shape[axis]
        return np.take(x, indices=0, axis=axis)
    def looks_like_3d_ct_shape(shape3):
        if len(shape3) != 3:
            return False
        large_axes = sum(int(s >= 64) for s in shape3)
        return large_axes >= 2 and all(s > 1 for s in shape3)
    if arr.ndim == 3:
        arr3d = arr
    elif arr.ndim == 4:
        shape          = arr.shape
        candidate_axes = list()
        for axis, axis_size in enumerate(shape):
            if axis_size <= 16:
                remaining_shape = tuple(shape[i] for i in range(4) if i != axis)
                if looks_like_3d_ct_shape(remaining_shape):
                    candidate_axes.append(axis)
        if 0 in candidate_axes:
            selected_axis = 0
        elif (arr.ndim - 1) in candidate_axes:
            selected_axis = arr.ndim - 1
        else:
            selected_axis = candidate_axes[0]
        arr3d = select_from_axis(arr, axis=selected_axis)
    arr3d = np.squeeze(arr3d)
    return arr3d.astype(np.float32)

def preprocess_ct_for_radfm(ct_path):
    hu_min = -1000.0
    hu_max = 400.0
    arr    = load_ct_as_dhw(ct_path)
    arr    = np.clip(arr, hu_min, hu_max)
    arr    = (arr - hu_min) / (hu_max - hu_min + 1e-8)
    arr    = arr.astype(np.float32)
    x      = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    x      = F.interpolate(x, size=(64, 512, 512), mode="trilinear", align_corners=False)
    x      = x.squeeze(0).squeeze(0)
    x      = x.permute(1, 2, 0).contiguous()
    x      = x.unsqueeze(0).repeat(3, 1, 1, 1)
    return x.contiguous()

class RadFMPPCDataset(Dataset):
    def __init__(self, split, image_padding_tokens):
        self.base                 = SurgConceptDataset(split)
        self.image_padding_tokens = image_padding_tokens
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        item    = self.base[idx]
        case_id = item["case_id"]
        ct_path = item["ct_path"]
        summary = item["summary"]
        target  = float(item["target"].item())
        image   = preprocess_ct_for_radfm(ct_path)
        prompt  = build_radfm_prompt(summary)
        text    = "<image>" + self.image_padding_tokens[0] + "</image>" + prompt
        return {"case_id": case_id, "vision_x": image, "text": text, "target": torch.tensor(target, dtype=torch.float32)}

def radfm_collate_fn(batch):
    case_ids = [b["case_id"] for b in batch]
    images   = torch.stack([b["vision_x"] for b in batch], dim=0)
    images   = images.unsqueeze(1)
    texts    = [b["text"] for b in batch]
    targets  = torch.stack([b["target"] for b in batch], dim=0)
    return {"case_id": case_ids, "vision_x": images, "text": texts, "target": targets}

def load_radfm_model(radfm_repo, language_path, checkpoint_path, device):
    quick_demo_path = Path(radfm_repo).resolve() / "Quick_demo"
    sys.path.insert(0, str(quick_demo_path))
    from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
    model               = MultiLLaMAForCausalLM(lang_model_path=language_path)
    ckpt                = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    model               = model.to(device)
    model               = model.half()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

@torch.no_grad()
def extract_radfm_embedding(model, tokenizer, text_list, vision_x, device):
    tokenized                  = tokenizer(text_list, max_length=2048, truncation=True, padding=True, return_tensors="pt")
    lang_x                     = tokenized["input_ids"].to(device)
    attention_mask             = tokenized["attention_mask"].to(device)
    vision_x                   = vision_x.to(device=device, dtype=torch.float16)
    model.embedding_layer.flag = "Text"
    input_embedding, _         = model.embedding_layer(lang_x, vision_x, key_words_query=None)
    output                     = model.lang_model(inputs_embeds=input_embedding, attention_mask=attention_mask, output_hidden_states=True, return_dict=True, use_cache=False)
    last_hidden                = output.hidden_states[-1].float()
    mask                       = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

@torch.no_grad()
def extract_radfm_features_for_split(split, cache_root, tokenizer, image_padding_tokens, model, batch_size, num_workers, device, target_h, target_w, target_d, recompute_features=False):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / (f"{split}_radfm_cttext_H{target_h}_W{target_w}_D{target_d}_features.npz")
    if cache_file.exists() and not recompute_features:
        arr = np.load(cache_file, allow_pickle=True)
        return {"case_ids": arr["case_ids"], "targets": arr["targets"], "features": arr["features"]}
    dataset = RadFMPPCDataset(split, image_padding_tokens)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=radfm_collate_fn)
    all_features, all_targets, all_case_ids = list(), list(), list()
    for batch in tqdm(loader):
        vision_x = batch["vision_x"]
        feat     = extract_radfm_embedding(model=model, tokenizer=tokenizer, text_list=batch["text"], vision_x=vision_x, device=device)
        feat     = feat.detach().cpu().float().numpy()
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
        xb = xb_tuple[0].to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits)
        probs.append(prob.detach().cpu().numpy())
    return np.concatenate(probs, axis=0)

def train_mlp_classifier(X_train, y_train, X_val, y_val, X_test, y_test, device, hidden_dim=256, dropout=0.25, lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=200, patience=25, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    X_train, X_val, X_test, imputer, scaler = preprocess_features(X_train, X_val, X_test)
    train_loader                            = make_tensor_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    model                                   = RiskMLP(in_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    n_pos, n_neg                            = float(np.sum(y_train == 1)), float(np.sum(y_train == 0))
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
        if epoch == 1 or epoch % 10 == 0:
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
                "test_tpr_at_fpr_0.3": float(tpr_at_fpr(y_test, test_prob, 0.3)),
                "pos_weight": float(pos_weight)}
    artifacts = {"model": model, "imputer": imputer, "scaler": scaler, "val_prob": val_prob, "test_prob": test_prob, "test_target": y_test, "results": results} 
    return artifacts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radfm_repo", default="", help="Path where you clone github repo: https://github.com/chaoyi-wu/radfm")
    parser.add_argument("--language_path", default="", help="Path to language file directory in github repo")
    parser.add_argument("--checkpoint_path", default="", help="Path to trained checkpoint in Quick demo directory")
    parser.add_argument("--cache_root", default="", help="Path to RadFM cache")
    parser.add_argument("--target_h", type=int, default=512)
    parser.add_argument("--target_w", type=int, default=512)
    parser.add_argument("--target_d", type=int, default=64)
    parser.add_argument("--extract_batch_size", type=int, default=16, help="Batch size for RadFM")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.25)
    parser.add_argument("--mlp_lr", type=float, default=1e-5, help="MLP learning rate")
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-6)
    parser.add_argument("--mlp_batch_size", type=int, default=512, help="MLP training batch size")
    parser.add_argument("--mlp_epochs", type=int, default=200, help="Number of epochs for training MLP")
    parser.add_argument("--mlp_patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompute_features", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device                          = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, image_padding_tokens = get_radfm_tokenizer(args.language_path)
    model                           = load_radfm_model(args.radfm_repo, args.language_path, args.checkpoint_path, device)
    train                           = extract_radfm_features_for_split(split="train", cache_root=args.cache_root,
                                                                        tokenizer=tokenizer, image_padding_tokens=image_padding_tokens,
                                                                        model=model, batch_size=args.extract_batch_size,
                                                                        num_workers=args.num_workers, device=device,
                                                                        target_h=args.target_h, target_w=args.target_w,
                                                                        target_d=args.target_d, recompute_features=args.recompute_features)
    val                             = extract_radfm_features_for_split(split="val", cache_root=args.cache_root,
                                                                        tokenizer=tokenizer, image_padding_tokens=image_padding_tokens,
                                                                        model=model, batch_size=args.extract_batch_size,
                                                                        num_workers=args.num_workers, device=device,
                                                                        target_h=args.target_h, target_w=args.target_w,
                                                                        target_d=args.target_d, recompute_features=args.recompute_features)
    test                            = extract_radfm_features_for_split(split="test", cache_root=args.cache_root,
                                                                        tokenizer=tokenizer, image_padding_tokens=image_padding_tokens,
                                                                        model=model, batch_size=args.extract_batch_size,
                                                                        num_workers=args.num_workers, device=device,
                                                                        target_h=args.target_h, target_w=args.target_w,
                                                                        target_d=args.target_d, recompute_features=args.recompute_features)
    X_train       = train["features"]
    y_train       = train["targets"].astype(int)
    X_val         = val["features"]
    y_val         = val["targets"].astype(int)
    X_test        = test["features"]
    y_test        = test["targets"].astype(int)
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
    with open("Path to save probabilities jsons/radfm.json", "w") as js:
        json.dump(prob_dict, js, indent = 4)
    print(f"Test AUC: {mlp_artifacts['results']['test_auc']}")
    print(f"Test TPR@FPR=0.2: {mlp_artifacts['results']['test_tpr_at_fpr_0.2']}")
    print(f"Test TPR@FPR=0.3: {mlp_artifacts['results']['test_tpr_at_fpr_0.3']}")

if __name__ == "__main__":
    main()