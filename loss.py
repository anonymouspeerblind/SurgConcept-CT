import torch
from torch import nn
import torch.nn.functional as F

class RiskPredictionloss(nn.Module):
    def __init__(self, alpha = 0.8, gamma = 4.0, reduction = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        targets   = targets.float()
        logits    = logits.float()
        bce_loss  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pred_prob = torch.sigmoid(logits)
        p_t       = (pred_prob * targets) + ((1.0 - pred_prob) * (1.0 - targets))
        loss      = ((1.0 - p_t) ** (self.gamma)) * bce_loss
        if self.alpha is not None:
            alpha_t = (self.alpha * targets) + ((1.0 - self.alpha) * (1.0 - targets))
            loss    = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

class Weak_concepts_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.all_idx     = torch.arange(20)
        self.binary_idx  = torch.tensor([1, 2, 3, 6, 7, 14])
        self.ordinal_idx = self.all_idx[~torch.isin(self.all_idx, self.binary_idx)]
    def forward(self, concept_logits, concept_targets):
        concept_targets = concept_targets.float()
        loss_binary     = F.binary_cross_entropy_with_logits(concept_logits[:, self.binary_idx], concept_targets[:, self.binary_idx])
        pred_ordinal    = torch.sigmoid(concept_logits[:, self.ordinal_idx])
        loss_ordinal    = F.mse_loss(pred_ordinal, concept_targets[:, self.ordinal_idx])
        concept_loss    = loss_binary + loss_ordinal
        return concept_loss

class Calibration_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_mmce = 1.0
        self.sigma       = 0.4
        self.mode        = "weighted"
        self.include_ce  = True
        self.ce_weight   = 1.0
        self.use_sqrt    = False
        self.eps         = 1e-12
    def _confidence_correctness(self, logits, target):
        if logits.ndim == 1 or (logits.ndim == 2 and logits.size(1) == 1):
            logits_1d = logits.reshape(-1)
            target_1d = target.reshape(-1).float()
            prob_pos  = torch.sigmoid(logits_1d)
            pred      = (prob_pos >= 0.5).long()
            conf      = torch.where(pred.bool(), prob_pos, 1.0 - prob_pos)
            correct   = pred.eq(target_1d.long()).float()
            ce        = F.binary_cross_entropy_with_logits(logits_1d, target_1d, reduction="mean")
            return conf, correct, ce
        target_1d  = target.reshape(-1).long()
        probs      = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct    = pred.eq(target_1d).float()
        ce         = F.cross_entropy(logits, target_1d, reduction="mean")
        return conf, correct, ce
    def _rbf_kernel(self, conf):
        diff = conf[:, None] - conf[None, :]
        return torch.exp(-(diff.pow(2)) / (2.0 * self.sigma ** 2))
    def _biased_mmce2(self, conf, correct):
        kernel   = self._rbf_kernel(conf)
        residual = correct - conf
        return (residual[:, None] * residual[None, :] * kernel).mean()
    def _weighted_mmce2(self, conf, correct):
        kernel         = self._rbf_kernel(conf)
        correct_mask   = correct.bool()
        incorrect_mask = ~correct_mask
        n_correct      = int(correct_mask.sum().item())
        n_incorrect    = int(incorrect_mask.sum().item())
        loss           = conf.new_tensor(0.0)
        if n_incorrect > 0:
            r0   = conf[incorrect_mask]
            k00  = kernel[incorrect_mask][:, incorrect_mask]
            loss = loss + (r0[:, None] * r0[None, :] * k00).sum() / (n_incorrect ** 2)
        if n_correct > 0:
            r1   = conf[correct_mask]
            k11  = kernel[correct_mask][:, correct_mask]
            loss = loss + ((1.0 - r1)[:, None] * (1.0 - r1)[None, :] * k11).sum() / (n_correct ** 2)
        if n_correct > 0 and n_incorrect > 0:
            r1   = conf[correct_mask]
            r0   = conf[incorrect_mask]
            k10  = kernel[correct_mask][:, incorrect_mask]
            loss = loss - 2.0 * (((1.0 - r1)[:, None] * r0[None, :] * k10).sum() / (n_correct * n_incorrect))
        return loss
    def forward(self, logits, target, return_components = False):
        conf, correct, ce = self._confidence_correctness(logits, target)
        if self.mode == "weighted":
            mmce2 = self._weighted_mmce2(conf, correct)
        else:
            mmce2 = self._biased_mmce2(conf, correct)
        mmce2 = torch.clamp(mmce2, min=0.0)
        mmce  = torch.sqrt(mmce2 + self.eps) if self.use_sqrt else mmce2
        total = self.lambda_mmce * mmce
        if self.include_ce:
            total = self.ce_weight * ce + total
        if return_components:
            return total, {"ce": ce.detach(), "mmce": mmce.detach(), "mmce2": mmce2.detach(), "mean_confidence": conf.detach().mean(), "accuracy": correct.detach().mean()}
        return total

class Fusion_loss(nn.Module):
    def __init__(self, temperature = 0.1):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        device            = features.device
        features          = F.normalize(features, dim=-1)
        labels            = labels.view(-1, 1)
        B                 = features.size(0)
        sim               = (features @ features.t()) / self.temperature
        logits_mask       = torch.ones_like(sim, device=device) - torch.eye(B, device=device)
        positive_mask     = (labels == labels.t()).float().to(device) * logits_mask
        sim               = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim           = torch.exp(sim) * logits_mask
        log_prob          = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))
        pos_count         = positive_mask.sum(dim=1)
        valid             = pos_count > 0
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=features.dtype)
        return -mean_log_prob_pos[valid].mean()

class Stage_wise_loss_fn(nn.Module):
    def __init__(self):
        super().__init__()
        self.risk_loss_fn        = RiskPredictionloss()
        self.concept_loss_fn     = Weak_concepts_loss()
        self.fusion_loss_fn      = Fusion_loss(temperature = 0.1)
        self.calibration_loss_fn = Calibration_loss()
    def forward(self, outputs, targets, concept_targets, stage):
        y                = targets.float()
        risk_loss        = self.risk_loss_fn(outputs["risk_logit"], y)
        concept_loss     = self.concept_loss_fn(outputs["concept_pred_logits"], concept_targets.float())
        calibration_loss = self.calibration_loss_fn(outputs["risk_logit"], y)
        fusion_loss      = self.fusion_loss_fn(outputs["h_fusion"], y.long())
        if stage == "clinical":
            self.risk_weight        = 1.0
            self.concept_weight     = 0.5
            self.fusion_weight      = 0.03
            self.calibration_weight = 0.2
        elif stage == "full":
            self.risk_weight        = 1.0
            self.concept_weight     = 0.5
            self.fusion_weight      = 0.05
            self.calibration_weight = 0.2

        global_loss = ((self.risk_weight * risk_loss) + (self.concept_weight * concept_loss) + (self.fusion_weight * fusion_loss) + (self.calibration_weight * calibration_loss))

        return {"loss": global_loss, "risk_loss": risk_loss.detach(), "concept_loss": concept_loss.detach(), "fusion_loss": fusion_loss.detach(), "calibration_loss": calibration_loss.detach()}