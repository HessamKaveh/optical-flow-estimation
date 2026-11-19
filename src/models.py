"""Optical-flow networks.

FlowNetS (Dosovitskiy et al., ICCV 2015): both frames stacked as a 6-channel input, a 10-layer encoder
and a refinement decoder that predicts flow at 5 scales (1/64 ... 1/4) with upconvolutions + skip connections.
Implemented here from scratch.

RAFT-small (Teed & Deng, ECCV 2020): all-pairs 4D correlation volume + ConvGRU iterative updates.
The architecture comes from torchvision (randomly initialised and trained here with our own pipeline);
the official pretrained torchvision RAFT weights are only used as a reference in the evaluation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small

MEAN = torch.tensor([0.411, 0.432, 0.45]).view(1, 3, 1, 1) * 255


def conv(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, (k - 1) // 2), nn.LeakyReLU(0.1, True))


def deconv(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.LeakyReLU(0.1, True))


class FlowNetS(nn.Module):
    DIV_FLOW = 20.0   # flow targets are divided by 20 during training, as in the reference code

    def __init__(self):
        super().__init__()
        self.conv1 = conv(6, 64, 7, 2)
        self.conv2 = conv(64, 128, 5, 2)
        self.conv3 = conv(128, 256, 5, 2); self.conv3_1 = conv(256, 256)
        self.conv4 = conv(256, 512, 3, 2); self.conv4_1 = conv(512, 512)
        self.conv5 = conv(512, 512, 3, 2); self.conv5_1 = conv(512, 512)
        self.conv6 = conv(512, 1024, 3, 2); self.conv6_1 = conv(1024, 1024)
        self.deconv5, self.deconv4 = deconv(1024, 512), deconv(1026, 256)
        self.deconv3, self.deconv2 = deconv(770, 128), deconv(386, 64)
        self.pred6, self.pred5 = nn.Conv2d(1024, 2, 3, 1, 1), nn.Conv2d(1026, 2, 3, 1, 1)
        self.pred4, self.pred3 = nn.Conv2d(770, 2, 3, 1, 1), nn.Conv2d(386, 2, 3, 1, 1)
        self.pred2 = nn.Conv2d(194, 2, 3, 1, 1)
        self.up = nn.ModuleList([nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False) for _ in range(4)])
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, im1, im2):
        x = torch.cat([im1, im2], 1)
        x = (x - MEAN.to(x).repeat(1, 2, 1, 1)) / 255.0
        c2 = self.conv2(self.conv1(x))
        c3 = self.conv3_1(self.conv3(c2))
        c4 = self.conv4_1(self.conv4(c3))
        c5 = self.conv5_1(self.conv5(c4))
        c6 = self.conv6_1(self.conv6(c5))
        f6 = self.pred6(c6)
        x5 = torch.cat([c5, self.deconv5(c6), self.up[0](f6)], 1); f5 = self.pred5(x5)
        x4 = torch.cat([c4, self.deconv4(x5), self.up[1](f5)], 1); f4 = self.pred4(x4)
        x3 = torch.cat([c3, self.deconv3(x4), self.up[2](f4)], 1); f3 = self.pred3(x3)
        x2 = torch.cat([c2, self.deconv2(x3), self.up[3](f3)], 1); f2 = self.pred2(x2)
        return [f2, f3, f4, f5, f6]          # finest first, in units of flow / DIV_FLOW

    @torch.no_grad()
    def predict(self, im1, im2):
        f2 = self.forward(im1, im2)[0]
        return F.interpolate(f2, scale_factor=4, mode="bilinear", align_corners=False) * self.DIV_FLOW


class RAFTSmall(nn.Module):
    def __init__(self, weights=None, iters=12):
        super().__init__()
        self.net, self.iters = raft_small(weights=weights), iters

    @staticmethod
    def norm(x):
        return x / 127.5 - 1.0

    def forward(self, im1, im2, iters=None):
        return self.net(self.norm(im1), self.norm(im2), num_flow_updates=iters or self.iters)

    @torch.no_grad()
    def predict(self, im1, im2, iters=None):
        return self.forward(im1, im2, iters)[-1]


def build(name):
    return FlowNetS() if name == "flownets" else RAFTSmall()
