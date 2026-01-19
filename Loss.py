import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLoss(nn.Module):
    def __init__(self, ce_weights, alpha=0.5, beta=0.5, gamma=0.5,
                 tversky_alpha=0.7, tversky_beta=0.3, focal_gamma=4 / 3,
                 smooth=1e-6, region_weights=None):
        super().__init__()
        self.ce_weights = ce_weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_gamma = focal_gamma
        self.smooth = smooth
        self.region_weights = region_weights or [1.0, 1.0, 1.0]  # ET, TC, WT
        self.num_classes = ce_weights.shape[0]

        if abs(alpha + beta - 1.0) > 1e-3:
            raise ValueError("alpha + beta = 1.0")

    def dice_loss(self, probs, y_true):

        dice = 0.0
        valid_classes = 0
        present_classes = torch.unique(y_true)


        for cls in range(self.num_classes):
            if cls not in present_classes:
                continue


            target = (y_true == cls).float()
            pred = probs[:, cls]


            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice_cls = (2. * intersection + self.smooth) / (union + self.smooth)
            dice += (1 - dice_cls)
            valid_classes += 1


        if valid_classes == 0:
            return torch.tensor(0.0, device=probs.device)

        return dice / valid_classes

    def focal_tversky_loss(self, probs, y_true):

        et_mask = (y_true == 3).float()
        tc_mask = ((y_true == 1) | (y_true == 3)).float()
        wt_mask = ((y_true == 1) | (y_true == 2) | (y_true == 3)).float()
        regions_mask = [et_mask, tc_mask, wt_mask]

        et_pred = probs[:, 3]
        tc_pred = probs[:, 1] + probs[:, 3]
        wt_pred = probs[:, 1] + probs[:, 2] + probs[:, 3]
        regions_pred = [et_pred, tc_pred, wt_pred]

        losses = []
        present_regions = []


        for i, (pred, mask) in enumerate(zip(regions_pred, regions_mask)):
            if mask.sum() < 1e-3:
                continue
            present_regions.append(i)

            tp = (pred * mask).sum()
            fp = (pred * (1 - mask)).sum()
            fn = ((1 - pred) * mask).sum()
            numerator = tp + self.smooth
            denominator = tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.smooth
            tversky = numerator / denominator

            loss_cls = torch.pow(1 - tversky, 1 / self.focal_gamma)
            losses.append(loss_cls * self.region_weights[i])

        if not losses:
            return torch.tensor(0.0, device=probs.device)


        total_weight = sum(self.region_weights[i] for i in present_regions)
        return sum(losses) / total_weight

    def forward(self, y_pred, y_true):

        y_pred = y_pred - y_pred.max(dim=1, keepdim=True)[0]

        if (y_true < 0).any() or (y_true >= self.num_classes).any():
            raise ValueError(f"label value is out of [0, {self.num_classes-1}]")

        probs = F.softmax(y_pred, dim=1)

        ce_loss = F.cross_entropy(y_pred, y_true, weight=self.ce_weights)

        dice_loss = self.dice_loss(probs, y_true)
        dice_ce_loss = self.gamma * dice_loss + (1 - self.gamma) * ce_loss

        focal_tversky_loss = self.focal_tversky_loss(probs, y_true)

        total_loss = self.alpha * dice_ce_loss + self.beta * focal_tversky_loss
        return total_loss