#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .dice_loss import DiceLoss




def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        return _iou(pred, target, self.size_average)
#
# class BinaryDiceLoss(nn.Module):
#     """Dice loss of binary class
#     Args:
#         smooth: A float number to smooth loss, and avoid NaN error, default: 1
#         p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
#         predict: A tensor of shape [N, *]
#         target: A tensor of shape same with predict
#         reduction: Reduction method to apply, return mean over batch if 'mean',
#             return sum if 'sum', return a tensor of shape [N,] if 'none'
#     Returns:
#         Loss tensor according to arg reduction
#     Raise:
#         Exception if unexpected reduction
#     """
#     def __init__(self, smooth=1, p=2, reduction='mean'):
#         super(BinaryDiceLoss, self).__init__()
#         self.smooth = smooth
#         self.p = p
#         self.reduction = reduction
#
#     def forward(self, predict, target):
#         assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
#         predict = predict.contiguous().view(predict.shape[0], -1)
#         target = target.contiguous().view(target.shape[0], -1)
#
#         num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
#         den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
#
#         loss = 1 - num / den
#
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         elif self.reduction == 'none':
#             return loss
#         else:
#             raise Exception('Unexpected reduction {}'.format(self.reduction))

#
# class DiceLoss(nn.Module):
#     """Dice loss, need one hot encode input
#     Args:
#         weight: An array of shape [num_classes,]
#         ignore_index: class index to ignore
#         predict: A tensor of shape [N, C, *]
#         target: A tensor of same shape with predict
#         other args pass to BinaryDiceLoss
#     Return:
#         same as BinaryDiceLoss
#     """
#     def __init__(self, weight=None, ignore_index=None, **kwargs):
#         super(DiceLoss, self).__init__()
#         self.kwargs = kwargs
#         self.weight = weight
#         self.ignore_index = ignore_index
#
#     def forward(self, predict, target):
#         assert predict.shape == target.shape, 'predict & target shape do not match'
#         dice = BinaryDiceLoss(**self.kwargs)
#         total_loss = 0
#         predict = F.sigmoid(predict)
#         # predict = F.softmax(predict,dim=1)
#         # print(predict.shape)
#         for i in range(target.shape[1]):
#             if i != self.ignore_index:
#                 dice_loss = dice(predict[:, i], target[:, i])
#                 if self.weight is not None:
#                     assert self.weight.shape[0] == target.shape[1], \
#                         'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
#                     dice_loss *= self.weights[i]
#                 total_loss += dice_loss
#
#         return total_loss/target.shape[1]


class IOU_add_bce_Loss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(IOU_add_bce_Loss, self).__init__()
        self.criterion1 = IOU()
        self.criterion2=nn.BCEWithLogitsLoss()
    def forward(self, predict, target):

        loss1 = self.criterion1(predict,target)

        loss2=self.criterion2(predict,target)
        loss=0.5*loss1+0.5*loss2

        return loss

class IOU_add_bce_Loss2(nn.Module):
    def __init__(self, weight=None, ignore_index=None, boundary_weight=0.1, **kwargs):
        super(IOU_add_bce_Loss2, self).__init__()
        self.criterion1 = IOU()
        self.criterion2 = nn.BCEWithLogitsLoss()
        self.boundary_weight = boundary_weight  # 边界感知损失的权重
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        # 计算 IOU 损失
        loss1 = self.criterion1(predict, target)

        # 计算 BCE 损失
        loss2 = self.criterion2(predict, target)

        # 计算边界感知损失
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            predict = predict[mask]
            target = target[mask]

        # 将预测值通过 sigmoid 转换为概率
        predict_prob = torch.sigmoid(predict)

        # 计算预测和目标的梯度差异
        predict_grad = torch.abs(predict_prob - F.avg_pool2d(predict_prob, kernel_size=3, stride=1, padding=1))
        target_grad = torch.abs(target - F.avg_pool2d(target, kernel_size=3, stride=1, padding=1))
        boundary_loss = torch.mean((predict_grad - target_grad) ** 2)

        # 加权求和
        loss = 0.5 * loss1 + 0.5 * loss2 + self.boundary_weight * boundary_loss

        return loss
import torch
import torch.nn as nn


class IOU_add_bce_Loss_with_Gradient_Penalty(nn.Module):
    def __init__(self, weight=None, **kwargs):
        super(IOU_add_bce_Loss_with_Gradient_Penalty, self).__init__()
        self.criterion1 = IOU()
        self.criterion2 = nn.BCEWithLogitsLoss(weight=weight)


    def forward(self, predict, target):
        loss1 = self.criterion1(predict, target)
        loss2 = self.criterion2(predict, target)

        # 计算梯度惩罚
        grad_penalty = self.compute_gradient_penalty(predict, target)

        # 总损失函数
        total_loss = 0.5 * loss1 + 0.5 * loss2 + 0.2 * grad_penalty
        print("loss1: ", loss1 * 0.5, "  , loss2: ", loss2 * 0.5, "  , grad_penalty: ", grad_penalty * 0.2, "  , total_loss: ", total_loss)
        return total_loss

    def compute_gradient_penalty(self, predict, target, grad_penalty_lambda=1E-5):
        """
        计算梯度惩罚
        """
        # 计算预测值的梯度
        predict.requires_grad_(True)
        loss = self.criterion2(predict, target)
        gradients = torch.autograd.grad(loss, predict, create_graph=True)[0]

        # 计算L2范数
        grad_penalty = torch.norm(gradients, p=2).mean()
        return grad_penalty * grad_penalty_lambda

class IOU_add_bce_Loss_and_dice(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super().__init__()
        self.criterion1 = IOU()
        self.criterion2=nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    def forward(self, predict, target):

        loss1 = self.criterion1(predict,target)

        loss2=self.criterion2(predict,target)

        loss3 = self.dice_loss(predict,target)
        loss=0.4*loss1+0.3*loss2+0.3*loss3

        return loss

# class DiceLoss(nn.Module):
#     def __init__(self, n_classes = 1):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes
#
#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()
#
#     def _dice_loss(self, score, target):
#         target = target.float()
#         smooth = 1e-5
#         intersect = torch.sum(score * target)
#         y_sum = torch.sum(target * target)
#         z_sum = torch.sum(score * score)
#         loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#         loss = 1 - loss
#         return loss
#
#     def forward(self, inputs, target, weight=None, softmax=False):
#         inputs = F.sigmoid(inputs)
#         if softmax:
#             inputs = torch.softmax(inputs, dim=1)
#         target = self._one_hot_encoder(target)
#         if weight is None:
#             weight = [1] * self.n_classes
#         assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
#         class_wise_dice = []
#         loss = 0.0
#         for i in range(0, self.n_classes):
#             dice = self._dice_loss(inputs[:, i], target[:, i])
#             class_wise_dice.append(1.0 - dice.item())
#             loss += dice * weight[i]
#         return loss / self.n_classes

if __name__ == "__main__":

    label = torch.ones((2,1,512,512))
    pre = torch.zeros((2,1,512,512))

    criterion = IOU_add_bce_Loss()
    loss = criterion(pre,label)
    print(loss)