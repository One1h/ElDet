import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.autograd import Variable


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4) # 4
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, pred_tensor, target_tensor):
        return self.neg_loss(pred_tensor, target_tensor)


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, mask, ind, target):
        pred = _transpose_and_gather_feat(pred, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.smooth_l1_loss(pred * mask, target * mask, reduction='sum')
        return loss / (mask.sum() + 1e-8)


class RegL1Loss_ang(nn.Module):
    def __init__(self):
        super(RegL1Loss_ang, self).__init__()

    def forward(self, pred, mask, ind, target, pred_ab):
        pred_ang = _transpose_and_gather_feat(pred, ind)
        mask_ang = mask.unsqueeze(2).expand_as(pred_ang).float()
        loss = F.smooth_l1_loss(pred_ang * mask_ang, target * mask_ang, reduction='none')

        pred_ab = _transpose_and_gather_feat(pred_ab, ind)
        mask_ab = mask.unsqueeze(2).expand_as(pred_ab).float()
        F.relu(pred_ab, inplace=True)

        ab_ratio = ((pred_ab * mask_ab)[:, :, 0] / ((pred_ab * mask_ab)[:, :, 1] + 1e-8)).reshape((-1, 100, 1))
        ab_ratio.clamp_(min=1, max=10)
        ab_ratio = torch.where(ab_ratio < 1.2, 1, 2)

        loss = torch.sum(loss * ab_ratio)
        return loss / (mask.sum() + 1e-8)


def trace(A):
    return A.diagonal(dim1=-2, dim2=-1).sum(-1)


def sqrt_newton_schulz_autograd(A, numIters, dtype):
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A)).cuda()
    I = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(batchSize, 1, 1).type(dtype), requires_grad=False).cuda()
    Z = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(batchSize, 1, 1).type(dtype), requires_grad=False).cuda()

    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def wasserstein_distance_sigma(sigma1, sigma2):
    wasserstein_distance_item2 = torch.matmul(sigma1, sigma1) + torch.matmul(sigma2,
                                                                             sigma2) - 2 * sqrt_newton_schulz_autograd(
        torch.matmul(torch.matmul(sigma1, torch.matmul(sigma2, sigma2)), sigma1), 20, torch.FloatTensor)
    wasserstein_distance_item2 = trace(wasserstein_distance_item2)

    return wasserstein_distance_item2


# @weighted_loss
def gwds_loss(pred, target, weight, eps=1e-6):
    """IoU loss.
    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.
    Args:
        pred (Tensor): Predicted bboxes of format (xc, yc, w, h, a),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    mask = (weight > 0).detach()
    pred = pred[mask]
    target = target[mask]

    x1 = pred[:, 0]
    y1 = pred[:, 1]
    w1 = pred[:, 2]
    h1 = pred[:, 3]
    theta1 = pred[:, 4]

    sigma1_1 = w1 / 2 * torch.cos(theta1) ** 2 + h1 / 2 * torch.sin(theta1) ** 2
    sigma1_2 = w1 / 2 * torch.sin(theta1) * torch.cos(theta1) - h1 / 2 * torch.sin(theta1) * torch.cos(theta1)
    sigma1_3 = w1 / 2 * torch.sin(theta1) * torch.cos(theta1) - h1 / 2 * torch.sin(theta1) * torch.cos(theta1)
    sigma1_4 = w1 / 2 * torch.sin(theta1) ** 2 + h1 / 2 * torch.cos(theta1) ** 2
    sigma1 = torch.reshape(
        torch.cat((sigma1_1.unsqueeze(1), sigma1_2.unsqueeze(1), sigma1_3.unsqueeze(1), sigma1_4.unsqueeze(1)), axis=1),
        (-1, 2, 2))

    x2 = target[:, 0]
    y2 = target[:, 1]
    w2 = target[:, 2]
    h2 = target[:, 3]
    theta2 = target[:, 4]
    sigma2_1 = w2 / 2 * torch.cos(theta2) ** 2 + h2 / 2 * torch.sin(theta2) ** 2
    sigma2_2 = w2 / 2 * torch.sin(theta2) * torch.cos(theta2) - h2 / 2 * torch.sin(theta2) * torch.cos(theta2)
    sigma2_3 = w2 / 2 * torch.sin(theta2) * torch.cos(theta2) - h2 / 2 * torch.sin(theta2) * torch.cos(theta2)
    sigma2_4 = w2 / 2 * torch.sin(theta2) ** 2 + h2 / 2 * torch.cos(theta2) ** 2
    sigma2 = torch.reshape(
        torch.cat((sigma2_1.unsqueeze(1), sigma2_2.unsqueeze(1), sigma2_3.unsqueeze(1), sigma2_4.unsqueeze(1)), axis=1),
        (-1, 2, 2))

    wasserstein_distance_item1 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    wasserstein_distance_item2 = wasserstein_distance_sigma(sigma1, sigma2)
    wasserstein_distance = torch.max(wasserstein_distance_item1 + wasserstein_distance_item2,
                                     Variable(torch.zeros(wasserstein_distance_item1.shape[0]).type(torch.FloatTensor).cuda(),
                                              requires_grad=False))
    wasserstein_distance = torch.max(torch.sqrt(wasserstein_distance + eps),
                                     Variable(torch.zeros(wasserstein_distance_item1.shape[0]).type(torch.FloatTensor).cuda(),
                                              requires_grad=False))
    wasserstein_similarity = 1 / (wasserstein_distance + 2)
    wasserstein_loss = 1 - wasserstein_similarity

    return wasserstein_loss


def xywhr2xyrs(xywhr):
    xywhr = xywhr.reshape(-1, 5)
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7)
    r = torch.deg2rad(xywhr[..., 4])
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    return xy, R, S


def gwd_loss(pred, target, fun='log', tau=1.0, alpha=1.0, normalize=False):
    """
    given any positive-definite symmetrical 2*2 matrix Z:
    Tr(Z^(1/2)) = sqrt(λ_1) + sqrt(λ_2)
    where λ_1 and λ_2 are the eigen values of Z
    meanwhile we have:
    Tr(Z) = λ_1 + λ_2
    det(Z) = λ_1 * λ_2
    combination with following formula:
    (sqrt(λ_1) + sqrt(λ_2))^2 = λ_1 + λ_2 + 2 * sqrt(λ_1 * λ_2)
    yield:
    Tr(Z^(1/2)) = sqrt(Tr(Z) + 2 * sqrt(det(Z)))
    for gwd loss the frustrating coupling part is:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    assuming Z = Σp^(1/2) * Σt * Σp^(1/2) then:
    Tr(Z) = Tr(Σp^(1/2) * Σt * Σp^(1/2))
    = Tr(Σp^(1/2) * Σp^(1/2) * Σt)
    = Tr(Σp * Σt)
    det(Z) = det(Σp^(1/2) * Σt * Σp^(1/2))
    = det(Σp^(1/2)) * det(Σt) * det(Σp^(1/2))
    = det(Σp * Σt)
    and thus we can rewrite the coupling part as:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    = Tr{Z^(1/2)} = sqrt(Tr(Z) + 2 * sqrt(det(Z))
    = sqrt(Tr(Σp * Σt) + 2 * sqrt(det(Σp * Σt)))
    """
    xy_p, R_p, S_p = xywhr2xyrs(pred)
    xy_t, R_t, S_t = xywhr2xyrs(target)

    xy_distance = (xy_p - xy_t).square().sum(dim=-1)

    Sigma_p = R_p.matmul(S_p.square()).matmul(R_p.permute(0, 2, 1))
    Sigma_t = R_t.matmul(S_t.square()).matmul(R_t.permute(0, 2, 1))

    whr_distance = S_p.diagonal(dim1=-2, dim2=-1).square().sum(dim=-1)
    whr_distance = whr_distance + S_t.diagonal(dim1=-2, dim2=-1).square().sum(
        dim=-1)
    _t = Sigma_p.matmul(Sigma_t)

    _t_tr = _t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = S_p.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    _t_det_sqrt = _t_det_sqrt * S_t.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    whr_distance = whr_distance + (-2) * ((_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(0)
    # distance = (xy_distance + alpha * alpha * whr_distance).clamp(0).sqrt()

    if normalize:
        wh_p = pred[..., 2:4].clamp(min=1e-7, max=1e7)
        wh_t = target[..., 2:4].clamp(min=1e-7, max=1e7)
        scale = ((wh_p.log() + wh_t.log()).sum(dim=-1) / 4).exp()
        distance = distance / scale

    if fun == 'log':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance)
    else:
        raise ValueError('Invalid non-linear function {fun} for gwd loss')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance


class GWDLoss(nn.Module):
    def __init__(self):
        super(GWDLoss, self).__init__()

    def forward(self, pred_tensor, target_tensor):
        ind = target_tensor['ind']
        mask = target_tensor['reg_mask']
        pred_ab = _transpose_and_gather_feat(pred_tensor['ab'], ind)
        mask_ab = mask.unsqueeze(2).expand_as(pred_ab).float()
        pred_ang = _transpose_and_gather_feat(pred_tensor['ang'], ind)
        mask_ang = mask.unsqueeze(2).expand_as(pred_ang).float()

        from predict import _topk
        K = 100
        _, inds, _, x, y = _topk(pred_tensor['hm'])
        pred_xy = torch.cat([x.reshape((-1, K, 1)), y.reshape((-1, K, 1))], dim=2)
        pred = torch.cat([pred_xy * mask_ab,
                          pred_ab * 2 * mask_ab,
                          (pred_ang - 90) * mask_ang
                          ],
                         dim=2)

        _, _, _, x, y = _topk(target_tensor['hm'])
        target_xy = torch.cat([x.reshape((-1, K, 1)), y.reshape((-1, K, 1))], dim=2)
        target = torch.cat([target_xy * mask_ab,
                            target_tensor['ab'] * 2 * mask_ab,
                            (target_tensor['ang'] - 90) * mask_ang
                            ],
                           dim=2)

        return torch.sum(gwd_loss(pred, target, fun='log', tau=1.0, alpha=1.0, normalize=False)) / (mask.sum() + 1e-8)


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, pred, target):
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        return loss

class WeightLoss(nn.Module):
    def __init__(self):
        super(WeightLoss, self).__init__()

    def forward(self, pred):
        target = torch.tensor(np.array([0.4, 0.3, 0.2, 0.1]*2, dtype=np.float32)).cuda()
        target = target.reshape((-1,4,1,1))
        loss = F.smooth_l1_loss(pred, target, reduction='sum')
        return loss

class CtdetLoss(torch.nn.Module):
    def __init__(self, loss_weight):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_ab = RegL1Loss()
        # self.crit_ang = RegL1Loss()
        self.crit_ang = RegL1Loss_ang()
        self.crit_iou = GWDLoss()
        self.crit_mask = MaskLoss()
        self.loss_weight = loss_weight

    def forward(self, pred_tensor, target_tensor):
        hm_weight = self.loss_weight['hm_weight']
        ab_weight = self.loss_weight['ab_weight']
        reg_weight = self.loss_weight['reg_weight']
        ang_weight = self.loss_weight['ang_weight']
        iou_weight = self.loss_weight['iou_weight']
        mask_weight = self.loss_weight['mask_weight']

        hm_loss, ab_loss, off_loss, ang_loss, iou_loss, mask_loss = 0, 0, 0, 0, 0, 0

        pred_tensor['hm'] = torch.sigmoid(pred_tensor['hm'])
        hm_loss += self.crit(pred_tensor['hm'], target_tensor['hm'])
        if ang_weight > 0:
            # ang_loss += self.crit_ang(pred_tensor['ang'], target_tensor['reg_mask'],
            #                           target_tensor['ind'], target_tensor['ang'])

            ang_loss += self.crit_ang(pred_tensor['ang'], target_tensor['reg_mask'],
                                      target_tensor['ind'], target_tensor['ang'],
                                      pred_tensor['ab'])

        if ab_weight > 0:
            ab_loss += self.crit_ab(pred_tensor['ab'], target_tensor['reg_mask'],
                                    target_tensor['ind'], target_tensor['ab'])

        if reg_weight > 0:
            off_loss += self.crit_reg(pred_tensor['reg'], target_tensor['reg_mask'],
                                      target_tensor['ind'], target_tensor['reg'])

        if iou_weight > 0:
            iou_loss += self.crit_iou(pred_tensor, target_tensor)

        if mask_weight > 0:
            mask_loss += self.crit_mask(pred_tensor['mask'], target_tensor['mask'])
            
        # self.weight_loss = WeightLoss()
        # weight_loss = self.weight_loss(w)

        return hm_weight * hm_loss + ab_weight * ab_loss + \
               ang_weight * ang_loss + reg_weight * off_loss + \
               iou_weight * iou_loss + mask_weight * mask_loss, \
               [(hm_weight * hm_loss).item(), (ab_weight * ab_loss).item(),
                (ang_weight * ang_loss).item(), (reg_weight * off_loss).item(),
                (iou_weight * iou_loss).item(), (mask_weight * mask_loss).item()]
