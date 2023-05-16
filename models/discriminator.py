# -*- coding: utf-8 -*-
# @Author  : LG

from torch import nn
import functools


def build_D(in_channels, mid_channels, n_layers, norm_type='batch'):
    norm_layer = build_norm_layer(norm_type)
    net = NLayerDiscriminator(in_channels, mid_channels, n_layers, norm_layer=norm_layer)
    return net

def build_norm_layer(norm_type):
    if norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = functools.partial(nn.Identity)
    else:
        raise ValueError('Norm Layer named {} is not supported.'.format(norm_type))
    return norm_layer


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels, mid_channels, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_list= []
        model_list += [nn.Conv2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1)]
        model_list += [nn.LeakyReLU(0.2, True)]

        for i in range(n_layers):
            stride = 1 if i == n_layers-1 else 2
            model_list += [nn.Conv2d(mid_channels, mid_channels*2, kernel_size=4, stride=stride, padding=1, bias=use_bias)]
            model_list += [norm_layer(mid_channels * 2)]
            model_list += [nn.LeakyReLU(0.2, True)]
            mid_channels = mid_channels * 2

        model_list += [nn.Conv2d(mid_channels, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    import torch
    nl = NLayerDiscriminator(3, 64)
    print(nl)
    x = torch.zeros((2, 3, 300, 300))
    y = nl(x)
    print(y.size())