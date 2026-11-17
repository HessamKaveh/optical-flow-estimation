import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class OpticalFlowNet(nn.Module):
    """Simple network for optical flow estimation"""
    def __init__(self):
        super().__init__()
        
        # Encoder: 2 frames (6 channels) -> features
        self.conv1 = conv_block(6, 32, stride=2)  # H/2
        self.conv2 = conv_block(32, 64, stride=2)  # H/4
        self.conv3 = conv_block(64, 128, stride=2)  # H/8
        self.conv4 = conv_block(128, 256, stride=2)  # H/16
        
        # Decoder: features → optical flow
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv4_up = conv_block(128, 128)
        
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3_up = conv_block(64, 64)
        
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv2_up = conv_block(32, 32)
        
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv1_up = conv_block(16, 16)
        
        # Output: flow (dx, dy)
        self.flow_out = nn.Conv2d(16, 2, kernel_size=1)
    
    def forward(self, frames):
        # frames: (B, 6, H, W) - two RGB frames
        
        # Encoder
        x1 = self.conv1(frames)  # (B, 32, H/2, W/2)
        x2 = self.conv2(x1)      # (B, 64, H/4, W/4)
        x3 = self.conv3(x2)      # (B, 128, H/8, W/8)
        x4 = self.conv4(x3)      # (B, 256, H/16, W/16)
        
        # Decoder
        d4 = self.deconv4(x4)    # (B, 128, H/8, W/8)
        d4 = d4 + x3
        d4 = self.conv4_up(d4)
        
        d3 = self.deconv3(d4)    # (B, 64, H/4, W/4)
        d3 = d3 + x2
        d3 = self.conv3_up(d3)
        
        d2 = self.deconv2(d3)    # (B, 32, H/2, W/2)
        d2 = d2 + x1
        d2 = self.conv2_up(d2)
        
        d1 = self.deconv1(d2)    # (B, 16, H, W)
        d1 = self.conv1_up(d1)
        
        # Optical flow output
        flow = self.flow_out(d1)  # (B, 2, H, W)
        
        return flow
