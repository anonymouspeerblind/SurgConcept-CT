import argparse, csv, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

class Extract_features():
    def __init__(self, device):
        self.device          = device
        self.tangerine_root  = "root directory where you clone https://github.com/niccolo246/3D-MAE-MedImaging"
        self.checkpoint_path = "Root path where you download Tangerine checkpoint (https://zenodo.org/records/18835750/files/tangerine.pth?download=1)"
        self.save_dtype      = "float16"
        self.model           = self.build_tangerine_mae_encoder()
    def build_tangerine_mae_encoder(self):
        self.tangerine_root = str(Path(self.tangerine_root).resolve())
        if self.tangerine_root not in sys.path:
            sys.path.insert(0, self.tangerine_root)
        import models_mae
        model = models_mae.mae_vit_large_patch16()
        ckpt  = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict):
            for key in ["model", "model_state", "state_dict"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    prev_state = ckpt[key]
                    break
            else:
                prev_state = ckpt
        state = dict()
        for k, v in prev_state.items():
            if not torch.is_tensor(v):
                continue
            k2 = k
            if k2.startswith("module."):
                k2 = k2[len("module."):]
            state[k2] = v
        msg = model.load_state_dict(state, strict=False)
        model.eval()
        model.to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        return model
    @torch.no_grad()
    def encode_tokens_no_mask(self, volume):
        b          = volume.shape[0]
        x          = self.model.patch_embed(volume)
        x          = x + self.model.pos_embed[:, 1:, :].to(dtype=x.dtype, device=x.device)
        cls_token  = self.model.cls_token + self.model.pos_embed[:, :1, :].to(dtype=x.dtype, device=x.device)
        cls_tokens = cls_token.expand(b, -1, -1)
        x          = torch.cat([cls_tokens, x], dim=1)
        for blk in self.model.blocks:
            x = blk(x)
        x      = self.model.norm(x)
        cls    = x[:, 0, :]
        tokens = x[:, 1:, :]
        return cls, tokens
    def pool_tokens_to_64_regions(self, patch_tokens, token_grid_size = (16, 16, 16), region_grid_size = (4, 4, 4)):
        b, l, c          = patch_tokens.shape
        tg_d, tg_h, tg_w = token_grid_size
        rg_d, rg_h, rg_w = region_grid_size
        expected_l       = tg_d * tg_h * tg_w
        block_d          = tg_d // rg_d
        block_h          = tg_h // rg_h
        block_w          = tg_w // rg_w
        x                = patch_tokens.view(b, tg_d, tg_h, tg_w, c)
        x                = x.view(b, rg_d, block_d, rg_h, block_h, rg_w, block_w, c)
        x                = x.mean(dim=(2, 4, 6))
        x                = x.reshape(b, rg_d * rg_h * rg_w, c)
        return x
    def return_pooled_features(self, batch):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        volume                                = self.batch["volume"].to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            cls, tokens = self.encode_tokens_no_mask(volume)
            regional    = self.pool_tokens_to_64_regions(tokens)

        features_cpu = regional.detach().cpu()
        if self.save_dtype == "float16":
            features_cpu = features_cpu.half()
        elif self.save_dtype == "float32":
            features_cpu = features_cpu.float()
        return features_cpu