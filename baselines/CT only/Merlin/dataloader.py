import os, json, torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SurgConceptDataset(Dataset):
    def __init__(self, split):
        self.split            = split
        self.df               = pd.read_csv(f"Parent folder to clinical features/{self.split}_cont_process.csv")
        with open(f"Parent folder to textual summaries of clinical features/{self.split}_summaries.json", "r") as js:
            self.textual_dict = json.load(js)
        with open(f"Parent folder to saved processed CT volume/{self.split}.json", "r") as js:
            self.ct_dict      = json.load(js)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        summary = self.textual_dict[row["CASE_ID"]]
        ct_path = self.ct_dict[row["CASE_ID"]]
        target  = torch.tensor(float(row["Target"]), dtype=torch.float32)
        return {"case_id": row["CASE_ID"], "summary": summary, "ct_path": ct_path, "target": target}