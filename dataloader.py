import os, json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class SurgConceptDataset(Dataset):
    def __init__(self, split, load_ct = False):
        self.split   = split
        self.load_ct = load_ct
        self.df      = pd.read_csv("Path of clinical data CSV for train/val/test split")
        with open("Path of weak concept list CSV for train/val/test split", "r") as js:
            self.concept_dict = json.load(js)
        with open("data/feature_type_mapping.json", "r") as js:
            self.feature_type_map = json.load(js)
        with open("data/feature_name_mapping.json", "r") as js:
            self.feature_name_map = json.load(js)
        with open("data/global_vocab.json", "r") as js:
            self.global_vocab = json.load(js)
        with open("data/ordinal_value_map.json", "r") as js:
            self.ordinal_values = json.load(js)
    def __len__(self):
        return len(self.df)
    def augment_clinical_mask(self, clinical, no_category_id, p=0.15, never_mask_feature_ids=None):
        clinical = {k: v.clone() for k, v in clinical.items()}
        F        = clinical["feature_name_ids"].shape[0]
        mask     = torch.rand(F) < p
        if never_mask_feature_ids is not None:
            for fid in never_mask_feature_ids:
                mask &= clinical["feature_name_ids"] != fid
        clinical["category_ids"][mask]   = no_category_id
        clinical["numeric_values"][mask] = 0.0
        clinical["aug_mask"]             = mask.float()
        return clinical
    def __getitem__(self, idx):
        row                                                                            = self.df.iloc[idx]
        feature_name_ids, feature_type_ids, category_ids, numeric_values, numeric_mask = list(), list(), list(), list(), list()
        for feature in self.feature_name_map["feature_order"]:
            value = row[feature]
            feature_name_ids.append(self.feature_name_map["feature_name_to_id"][feature])
            feature_type_ids.append(self.feature_type_map["feature_type_to_ID"][self.feature_type_map["feature_to_type"][feature]])

            if feature in ["Age", "BMI", "FEV1 Predicted", "DLCO Predicted", "Pack-Years Of Cigarette Use"]:
                category_ids.append(self.global_vocab["global_vocab"]["NO_CATEGORY"])
            else:
                category_ids.append(self.global_vocab["global_vocab"][f"{feature}::{value}"])
            
            if feature == "Gender":
                numeric_values.append(0)
                numeric_mask.append(0)
            else:
                if feature in ["Age", "BMI", "FEV1 Predicted", "DLCO Predicted", "Pack-Years Of Cigarette Use"]:
                    numeric_values.append(value)
                else:
                    numeric_values.append(self.ordinal_values['Ordinal_values'][feature][value])
                numeric_mask.append(1)

        clinical = {"feature_name_ids": torch.tensor(feature_name_ids, dtype=torch.long),
                    "feature_type_ids": torch.tensor(feature_type_ids, dtype=torch.long),
                    "category_ids": torch.tensor(category_ids, dtype=torch.long),
                    "numeric_values": torch.tensor(numeric_values, dtype=torch.float32),
                    "numeric_mask": torch.tensor(numeric_mask, dtype=torch.float32)}
        if self.split == "train":
            clinical  = self.augment_clinical_mask(clinical, no_category_id=self.global_vocab["global_vocab"]["NO_CATEGORY"])
        if self.load_ct:
            ct_volume = torch.load(f"Path to saved processed volume/{self.split}/{case}.pt", map_location = "cpu")
        else:
            ct_volume = torch.empty(0)
        concept_ls    = torch.tensor(self.concept_dict[row["CASE_ID"]], dtype = torch.float32)
        target        = torch.tensor(float(row["Target"]), dtype=torch.float32)
        return {"case_id": row["CASE_ID"], "clinical": clinical, "volume": ct_volume, "concepts": concept_ls, "target": target}