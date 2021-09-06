import torch
from torch import nn

###### LOSSES #######

class BerHuLoss_test(nn.Module):
    def __init__(self, scale=0.5, eps=1e-5):
        super(BerHuLoss_test, self).__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, pred, gt):
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        img1 = img1[img2 > self.eps]
        img2 = img2[img2 > self.eps]

        diff = torch.abs(img1 - img2)
        threshold = self.scale * torch.max(diff).detach()
        mask = diff > threshold
        diff[mask] = ((img1[mask]-img2[mask])**2 + threshold**2) / (2*threshold + self.eps)
        return diff.sum() / diff.numel()


class LogDepthL1Loss_test(nn.Module):
    def __init__(self, eps=1e-5):
        super(LogDepthL1Loss_test, self).__init__()
        self.eps = eps
    def forward(self, pred, gt):
        mask = gt > self.eps
        diff = torch.abs(torch.log(gt[mask]) - pred[mask])
        return diff.mean()

###### METRICS #######

class DepthL1Loss_test(nn.Module):
    def __init__(self, eps=1e-5):
        super(DepthL1Loss_test, self).__init__()
        self.eps = eps
    def forward(self, pred, gt):
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        # mask = gt > self.eps
        # img1[~mask] = 0.
        # img2[~mask] = 0.
        # non_zero_count = torch.sum((pred>0).int())
        return nn.L1Loss(reduction="mean")(img1, img2), 1
        #return nn.L1Loss(reduction="sum")(img1, img2), pred.numel()

class DepthL2Loss_test(nn.Module):
    def __init__(self, eps=1e-5):
        super(DepthL2Loss_test, self).__init__()
        self.eps = eps
    def forward(self, pred, gt):
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        # mask = gt > self.eps
        # img1[~mask] = 0.
        # img2[~mask] = 0.
        # non_zero_count = torch.sum((pred>0).int())
        return nn.MSELoss(reduction="mean")(img1, img2), 1
        # return nn.MSELoss(reduction="sum")(img1, img2), pred.numel()

class RMSELoss_test(nn.Module):
    def __init__(self, eps=1e-5):
        super(RMSELoss_test, self).__init__()
        self.eps = eps
    def forward(self, pred, gt):
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        # mask = gt > self.eps
        # img1[~mask] = 0.
        # img2[~mask] = 0.
        # non_zero_count = torch.sum((pred>0).int())
        # print(non_zero_count)
        return torch.sqrt(nn.MSELoss(reduction='mean')(img1, img2)), 1 #pred.numel()
        #return torch.sqrt(nn.MSELoss(reduction='sum')(img1, img2)/pred.numel()), 1 #pred.numel()

