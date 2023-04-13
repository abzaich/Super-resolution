import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from numpy import repeat



class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = args.channels
        self.inp_channels = args.inp_channels
        self.out_channels = args.out_channels
        self.channels = channels
        self.angRes = args.angRes
        self.factor = args.scale_factor
        self.ang_window_size = args.ang_window_size
        self.spa_window_size = args.spa_window_size
        self.act_type = args.act_type
        self.exp_ratio = args.exp_ratio
        layer_num = 8

        ##################### Initial Convolution #####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ##############Reshape################
        self.Reshape = nn.Conv2d(1, 64, kernel_size=1)

        ################ LGAN ################
        self.lagnblock = self.make_layer(layer_num=layer_num)

        ####################### UP Sampling ###########################
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels*self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )


    def make_layer(self, layer_num):
        layers1 = []
        layers2 = []
        layers3 = []
        for i in range(layer_num):
            layers1.append(SFEB(self.inp_channels, self.out_channels, self.exp_ratio, self.spa_window_size, self.act_type))
            layers2.append(SFEB(self.inp_channels, self.out_channels, self.exp_ratio, self.spa_window_size * 2, self.act_type))
            layers3.append(SFEB(self.inp_channels, self.out_channels, self.exp_ratio, self.spa_window_size * 3, self.act_type))
        return nn.Sequential(*layers1)

    def forward(self, lr):
        # Bicubic
        lr_upscale = interpolate(lr, self.angRes, scale_factor=self.factor, mode='bicubic')
        # [B(atch), 1, A(ngRes)*h(eight)*S(cale), A(ngRes)*w(idth)*S(cale)]

        # Reshape
        buffer = self.Reshape(lr)

        # LGAN
        buffer = self.lagnblock(buffer) + buffer

        # Up-Sampling

        buffer = self.upsampling(buffer)
        out = buffer + lr_upscale

        return out

class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 4
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0 * g:1 * g, 1, 2] = 1.0
        mask[:, 1 * g:2 * g, 1, 0] = 1.0
        mask[:, 2 * g:3 * g, 2, 1] = 1.0
        mask[:, 3 * g:4 * g, 0, 1] = 1.0
        mask[:, 4 * g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1)
        return y


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 4
        g = inp_channels // self.n_div
        self.weight[0 * g:1 * g, 0, 1, 2] = 1.0  ## left
        self.weight[1 * g:2 * g, 0, 1, 0] = 1.0  ## right
        self.weight[2 * g:3 * g, 0, 2, 1] = 1.0  ## up
        self.weight[3 * g:4 * g, 0, 0, 1] = 1.0  ## down
        self.weight[4 * g:, 0, 1, 1] = 1.0  ## identity

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y)
        return y

class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory':
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        elif conv_type == 'common':
            self.shift_conv = nn.Conv2d(inp_channels, out_channels, kernel_size=1)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y

class FD(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='gelu'):
        super(FD, self).__init__()
        self.exp_ratio = exp_ratio
        self.act_type = act_type

        self.conv0 = ShiftConv2d(inp_channels, out_channels * exp_ratio)
        self.conv1 = ShiftConv2d(out_channels * exp_ratio, out_channels)

        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        elif self.act_type == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)
        return y


class LGAN(nn.Module):
    def __init__(self, channels, window_size=5):
        super(LGAN, self).__init__()
        self.window_size = window_size
        self.split_chns = [channels * 2  // 3 for _ in range(3)]
        self.project_inp = nn.Sequential(nn.Conv2d(channels, 126, kernel_size=1),
                                         nn.BatchNorm2d(126))
        self.project_out = nn.Sequential(nn.Conv2d(63, channels, kernel_size=1))
        self.output = nn.Conv2d(63, 64, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        wsize = self.window_size
        ys = []
        # window attention
        q, v = rearrange(
            xs[0], 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
            qv=2, dh=wsize, dw=wsize
        )
        atn = (q @ q.transpose(-2, -1))

        atn = atn.softmax(dim=-1)
        y_ = (atn @ v)
        y_ = rearrange(
            y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
            h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
        )
        ys.append(y_)
        # shifted window attention
        x_ = torch.roll(xs[1], shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))  #shift
        q, v = rearrange(
            x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
            qv=2, dh=wsize, dw=wsize
        )
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        y_ = (atn @ v)
        y_ = rearrange(
            y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
            h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
        )
        y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))   #unshift
        ys.append(y_)
        # long-range attention
        # for row
        q, v = rearrange(xs[2], 'b (qv c) h w -> qv (b h) w c', qv=2)
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        v = (atn @ v)
        # for column
        q = rearrange(q, '(b h) w c -> (b w) h c', b=b)
        v = rearrange(v, '(b h) w c -> (b w) h c', b=b)
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        y_ = (atn @ v)
        y_ = rearrange(y_, '(b w) h c-> b c h w', b=b)
        ys.append(y_)

        y = torch.cat(ys, dim=1)
        y = self.project_out(y)


        return y


class AFEB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, ang_window_size=8, act_type='gelu'):
        super(AFEB, self).__init__()
        self.exp_ratio = exp_ratio
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.FD = FD(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio, act_type=act_type)
        self.LGAN = LGAN(channels=inp_channels, window_size=ang_window_size)

    def forward(self, x):
        x = self.LGAN(x) + x
        x = self.FD(x) + x
        return x

class SFEB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, spa_window_size=3, act_type='gelu'):
        super(SFEB, self).__init__()
        self.exp_ratio = exp_ratio
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.FD = FD(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio, act_type=act_type)
        self.LGAN = LGAN(channels=inp_channels, window_size=spa_window_size)

    def forward(self, x):
        x = self.LGAN(x) + x
        x = self.FD(x) + x
        return x

def interpolate(x, angRes, scale_factor, mode):
    [B, _, H, W] = x.size()
    h = H // angRes
    w = W // angRes
    x_upscale = x.view(B, 1, angRes, h, angRes, w)
    x_upscale = x_upscale.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * angRes ** 2, 1, h, w)
    x_upscale = F.interpolate(x_upscale, scale_factor=scale_factor, mode=mode, align_corners=False)
    x_upscale = x_upscale.view(B, angRes, angRes, 1, h * scale_factor, w * scale_factor)
    x_upscale = x_upscale.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 1, H * scale_factor, W * scale_factor)
    # [B, 1, A*h*S, A*w*S]

    return x_upscale


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR):
        loss = self.criterion_Loss(SR, HR)

        return loss


def weights_init(m):

    pass

