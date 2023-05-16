# -*- coding: utf-8 -*-
# @Author  : LG

from torch import nn
import functools


def build_G(in_channels, mid_channels, out_channels, num_block, norm_type='batch', use_dropout=True):
    norm_layer = build_norm_layer(norm_type)
    net = ResNet(in_channels, mid_channels, out_channels, num_block=num_block, norm_layer=norm_layer, use_dropout=use_dropout)
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

class ResNet(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_block, norm_layer=nn.InstanceNorm2d, padding_type='reflect', use_dropout=True):
        super(ResNet, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_list = []
        model_list += [nn.ReflectionPad2d(3)]
        model_list += [nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=0, bias=use_bias)]
        model_list += [norm_layer(mid_channels)]
        model_list += [nn.ReLU(True)]

        model_list += [nn.Conv2d(mid_channels * 1, mid_channels * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)]
        model_list += [norm_layer(mid_channels * 2)]
        model_list += [nn.ReLU(True)]
        model_list += [nn.Conv2d(mid_channels * 2, mid_channels * 4, kernel_size=3, stride=2, padding=1, bias=use_bias)]
        model_list += [norm_layer(mid_channels * 4)]
        model_list += [nn.ReLU(True)]

        for i in range(num_block):
            model_list += [ResBlock(mid_channels * 4, mid_channels * 4, norm_layer, padding_type, use_bias, use_dropout)]

        model_list += [nn.ConvTranspose2d(mid_channels * 4, mid_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)]
        model_list += [norm_layer(mid_channels * 2)]
        model_list += [nn.ReLU(True)]
        model_list += [nn.ConvTranspose2d(mid_channels * 2, mid_channels * 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)]
        model_list += [norm_layer(mid_channels * 1)]
        model_list += [nn.ReLU(True)]

        model_list += [nn.ReflectionPad2d(3)]
        model_list += [nn.Conv2d(mid_channels, out_channels, kernel_size=7, padding=0)]
        model_list += [nn.Tanh()]

        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer, padding_type, use_bias, use_dropout):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_block(in_channel, out_channel, norm_layer, padding_type, use_bias, use_dropout)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

    def build_block(self, in_channel, out_channel, norm_layer, padding_type, use_bias, use_dropout):

        model_list = []
        if padding_type == 'reflect':
            model_list += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            model_list += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            model_list += [nn.ZeroPad2d(1)]
        else:
            raise ValueError('padding {} is not supported'.format(padding_type))

        model_list += [nn.Conv2d(in_channel, out_channel, kernel_size=3, bias=use_bias)]
        model_list += [norm_layer(out_channel)]
        model_list += [nn.ReLU(True)]
        if use_dropout:
            model_list += [nn.Dropout(0.5)]

        if padding_type == 'reflect':
            model_list += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            model_list += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            model_list += [nn.ZeroPad2d(1)]
        else:
            raise ValueError('padding {} is not supported'.format(padding_type))

        model_list += [nn.Conv2d(out_channel, out_channel, kernel_size=3, bias=use_bias)]
        model_list += [norm_layer(out_channel)]
        return nn.Sequential(*model_list)

if __name__ == '__main__':
    import torch
    model = ResNet(3, 64, 3, 9)
    print(model)

    x = torch.ones(size=(1, 3, 256, 256))
    y = model(x)
    print(y.shape)