import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import torch.nn.functional as F

 
class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class RegionLoss(nn.Module):
    def __init__(self):
        super(RegionLoss, self).__init__()

    def forward(self, pred, target):
        if pred is None:
            return torch.tensor(0.0)
        
        B, N, _, H, W = pred.shape
        
        target_int, target_bound, target_bg = self.build_region_gt(target.squeeze(1))
        target_class = torch.stack([target_bg, target_bound, target_int], dim=1)
        target_class = target_class.unsqueeze(1).expand(-1, N, -1, -1, -1)
        
        pred_fg = pred[:, :, 1:3]
        target_fg = target_class[:, :, 1:3]
        valid_mask = target_fg.sum(dim=2) > 0
        
        kl = F.kl_div(pred_fg, target_fg, reduction="none").sum(dim=2)
        loss = (kl * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        return loss
    
    def build_region_gt(self, gt_mask, boundary_width=2):
        """
        gt_mask: (B,H,W), binary {0,1}
        return:
            gt_int, gt_bnd, gt_bg: (B,H,W)
        """
        gt_mask = gt_mask.float()

        # foreground / background
        gt_fg = gt_mask
        gt_bg = 1.0 - gt_fg

        # boundary: morphological gradient
        kernel = torch.ones(
            (1, 1, boundary_width * 2 + 1, boundary_width * 2 + 1),
            device=gt_mask.device,
        )

        gt_fg_ = gt_fg.unsqueeze(1)

        dilated = F.conv2d(gt_fg_, kernel, padding=boundary_width) > 0
        eroded  = F.conv2d(gt_fg_, kernel, padding=boundary_width) == kernel.numel()

        gt_boundary = (dilated ^ eroded).float().squeeze(1)

        gt_interior = gt_fg * (1.0 - gt_boundary)

        return gt_interior, gt_boundary, gt_bg
