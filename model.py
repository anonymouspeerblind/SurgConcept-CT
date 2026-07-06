import torch
from torch import nn
import torch.nn.functional as F

class ClinicalFeatureTokenizer(nn.Module):
    def __init__(self, num_features = 17, num_feature_types = 17, num_categories = 45, d_model = 256):
        super().__init__()
        self.name_emb     = nn.Embedding(num_features, d_model) # encoding name
        self.type_emb     = nn.Embedding(num_feature_types, d_model) # encoding type
        self.category_emb = nn.Embedding(num_categories, d_model) # encoding categories
        self.numeric_mlp  = nn.Sequential(nn.Linear(1, d_model // 2), nn.ReLU(inplace=True), nn.Linear(d_model // 2, d_model)) # encoder for numerical values
        self.norm_layer   = nn.LayerNorm(d_model)
        self.dropout      = nn.Dropout(0.3)
    def forward(self, clinical_data):
        numeric_values = clinical_data["numeric_values"].float()
        numeric_mask   = clinical_data.get("numeric_mask", torch.ones_like(numeric_values)).float()
        f_name         = self.name_emb(clinical_data["feature_name_ids"])
        f_type         = self.type_emb(clinical_data["feature_type_ids"])
        cat            = self.category_emb(clinical_data["category_ids"].clamp_min(0))
        numeric_enc    = self.numeric_mlp(numeric_values.unsqueeze(-1)) * numeric_mask.unsqueeze(-1)
        tokens         = self.norm_layer(f_name + f_type + cat + numeric_enc)
        return self.dropout(tokens)

class Clinical_data_encoder(nn.Module):
    def __init__(self, dim_model = 256):
        super().__init__()
        self.feature_tokenizer = ClinicalFeatureTokenizer()
        self.cls               = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.encoder_layer     = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model = dim_model, nhead = 4, dim_feedforward = 512, dropout = 0.3, batch_first = True,
                                                                                    activation = "gelu", norm_first = True), num_layers = 2)
        self.norm_layer        = nn.LayerNorm(dim_model)
        nn.init.trunc_normal_(self.cls, std = 0.02)
    def forward(self, clinical_batch):
        x          = self.feature_tokenizer(clinical_batch)
        cls_token  = self.cls.expand(x.size(0), -1, -1)
        x          = torch.cat([cls_token, x], dim=1)
        x          = self.encoder_layer(x)
        h_clinical = self.norm_layer(x[:, 0])
        return h_clinical

class CT_proj_layer(nn.Module):
    def __init__(self, in_dim = 1024, out_dim = 256, hidden_dim = 256):
        super().__init__()
        self.ct_projector             = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                                        nn.LayerNorm(hidden_dim),
                                                        nn.GELU(),
                                                        nn.Dropout(0.2),
                                                        nn.Linear(hidden_dim, out_dim),
                                                        nn.LayerNorm(out_dim))
        self.patch_position_embedding = nn.Parameter(torch.zeros(1, 64, out_dim))
        nn.init.trunc_normal_(self.patch_position_embedding, std=0.02)
    def forward(self, x):
        x = self.ct_projector(x)
        x = x + self.patch_position_embedding
        return x

class ClinicalGuidedCTAttention(nn.Module):
    def __init__(self, d_model = 256):
        super().__init__()
        self.wq_layer   = nn.Linear(d_model, d_model)
        self.wk_layer   = nn.Linear(d_model, d_model)
        self.wv_layer   = nn.Linear(d_model, d_model)
        self.attn_scale = d_model ** -0.5
        self.norm_layer = nn.LayerNorm(d_model)
    def forward(self, h_clinical, ct_tokens, ct_patch_mask = None):
        Q      = self.wq_layer(h_clinical).unsqueeze(1)
        K      = self.wk_layer(ct_tokens)
        V      = self.wv_layer(ct_tokens)
        scores = torch.bmm(Q, K.transpose(1, 2)).squeeze(1) * self.attn_scale
        if ct_patch_mask is not None:
            scores = scores.masked_fill(ct_patch_mask <= 0, float("-inf"))
        attn   = torch.softmax(scores, dim=-1)
        h_ct   = torch.bmm(attn.unsqueeze(1), V).squeeze(1)
        return self.norm_layer(h_ct), attn

class SurgConceptCPTModel(nn.Module):
    def __init__(self, d = 256):
        super().__init__()
        self.num_of_concepts       = 20
        self.clinical_encoder      = Clinical_data_encoder()
        self.ct_projector          = CT_proj_layer()
        self.ct_attention_layer    = ClinicalGuidedCTAttention()
        self.concept_predict_layer = nn.Sequential(nn.Linear(2 * d, 512),
                                                    nn.LayerNorm(512),
                                                    nn.GELU(),
                                                    nn.Dropout(0.3),
                                                    nn.Linear(512, 256),
                                                    nn.GELU(),
                                                    nn.Dropout(0.4),
                                                    nn.Linear(256, self.num_of_concepts))
        self.fusion_mlp            = nn.Sequential(nn.Linear(2 * d, 512),
                                                    nn.LayerNorm(512),
                                                    nn.GELU(),
                                                    nn.Dropout(0.5),
                                                    nn.Linear(512, d),
                                                    nn.LayerNorm(d),
                                                    nn.GELU())
        self.concept_linear_layer  = nn.Linear(self.num_of_concepts, 1, bias=False)
        self.hidden_linear_layer   = nn.Linear(d, 1)