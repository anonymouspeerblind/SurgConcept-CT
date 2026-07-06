import os, json
from tqdm import tqdm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
from sklearn.metrics import roc_curve, roc_auc_score

def tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    valid = fpr <= target_fpr
    if valid.sum() == 0:
        return 0.0
    return float(tpr[valid].max())

def plot_multi_roc(fpr_list, tpr_list, auc_list, labels):
    palette = qualitative.Set1
    n_colors = len(palette)
    line_styles   = ["solid", "dash", "dot", "dashdot", "longdash"]
    marker_symbols = ["circle", "x", "triangle-up", "diamond", "star", "hexagon"]
    fig = go.Figure()
    for i, (fpr, tpr, auc, lbl) in enumerate(zip(fpr_list, tpr_list, auc_list, labels)):
        color     = palette[i % n_colors]
        if str(palette[i % n_colors]) == "rgb(255,255,51)":
            color = "rgb(255,0,255)"
        dash      = line_styles[i % len(line_styles)]
        symbol    = marker_symbols[i % len(marker_symbols)]
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                    mode='lines',
                                    name=f"{lbl} (AUC={auc*100:.2f}%)",
                                    line=dict(color=color, dash=dash, width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
                            mode='lines',
                            line=dict(dash='dash', color='gray'),
                            showlegend=False))
    fig.update_layout(xaxis=dict(title=dict(text="False Positive Rate", font=dict(size=20))),
                        yaxis=dict(title=dict(text="True Positive Rate", font=dict(size=20))),
                        width=1000, height=1000, legend=dict(x=0.995, y=0.005,
                                                            xanchor="right",
                                                            yanchor="bottom",
                                                            bgcolor="rgba(255,255,255,0.5)",
                                                            font=dict(family="sans-serif", size=22),
                                                            tracegroupgap=5), plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black")
    fig.update_yaxes(scaleanchor="y", scaleratio=1)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    fig.write_image("Path to save ROC plot image")

if __name__ == "__main__":
    fpr_list, tpr_list, auc_list, labels   = list(), list(), list(), list()
    parent_dir = "Directory to probability jsons from different models"
    for models in tqdm(os.listdir(parent_dir)):
        with open(f"{parent_dir}/{models}", "r") as js:
            prob_dict = json.load(js)
        fpr, tpr, roc_thresholds = roc_curve(prob_dict['target'], prob_dict['pred'])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(roc_auc_score(prob_dict['target'], prob_dict['pred']))
        labels.append(models.split(".")[0])
    labels[labels.index("LogisticRegression")] = "Logistic Regression"
    labels[labels.index("XGB")]                = "XGBoost"
    labels[labels.index("Tangerine")]          = "TANGERINE"
    plot_multi_roc(fpr_list, tpr_list, auc_list, labels)