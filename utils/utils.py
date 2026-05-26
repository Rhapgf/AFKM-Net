import torch
import numpy as np
from thop import profile
from thop import clever_format
import argparse

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1

        self.first = 0
        self.second = 0
        self.third = 0
        self.forth = 0
        self.fifth = 0
        self.sixth = 0
        self.seventh = 0
        self.eighth = 0

    def update(self, val, n=1):
        self.val = val
        self.max = val if val>self.max else self.max
        self.min = val if val<self.min else self.min
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if val>=0 and val<=0.3:
            self.first +=1
        elif val>0.3 and val<=0.6:
            self.second +=1
        elif val>0.6 and val<=0.7:
            self.third +=1
        elif val>0.7 and val<=0.8:
            self.forth +=1
        elif val>0.8 and val<=0.85:
            self.fifth +=1
        elif val>0.85 and val<=0.9:
            self.sixth +=1
        elif val>0.9 and val<=0.95:
            self.seventh +=1
        elif val>0.95 and val<=1:
            self.eighth +=1
