import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


class SyntheticOpticalFlowDataset(Dataset):
    """دیتاست مصنوعی: فریم‌های متوالی با motion معروف"""
    def __init__(self, num_samples=200, img_size=256):
        self.num_samples = num_samples
        self.img_size = img_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # فریم اول: مربع‌های متحرک
        frame1 = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 200
        
        np.random.seed(idx)
        
        # مربع اول
        x1_start = 50 + idx % 100
        y1_start = 50
        cv2.rectangle(frame1, (x1_start, y1_start), (x1_start + 50, y1_start + 50), (0, 255, 0), -1)
        
        # مربع دوم
        x2_start = 200 - (idx % 100)
        y2_start = 150
        cv2.rectangle(frame1, (x2_start, y2_start), (x2_start + 50, y2_start + 50), (255, 0, 0), -1)
        
        # فریم دوم: با جابجایی کوچک
        frame2 = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 200
        
        dx1, dy1 = 2, 1  # جابجایی مربع اول
        x1_end = x1_start + dx1
        cv2.rectangle(frame2, (x1_end, y1_start + dy1), (x1_end + 50, y1_start + 50 + dy1), (0, 255, 0), -1)
        
        dx2, dy2 = -2, -1  # جابجایی مربع دوم
        x2_end = x2_start + dx2
        cv2.rectangle(frame2, (x2_end, y2_start + dy2), (x2_end + 50, y2_start + 50 + dy2), (255, 0, 0), -1)
        
        # تخمین optical flow (ground truth)
        flow = np.zeros((self.img_size, self.img_size, 2), dtype=np.float32)
        
        # منطقه مربع اول: dx1, dy1
        flow[y1_start:y1_start+50, x1_start:x1_start+50, 0] = dx1
        flow[y1_start:y1_start+50, x1_start:x1_start+50, 1] = dy1
        
        # منطقه مربع دوم: dx2, dy2
        flow[y2_start:y2_start+50, x2_start:x2_start+50, 0] = dx2
        flow[y2_start:y2_start+50, x2_start:x2_start+50, 1] = dy2
        
        # تبدیل به tensor
        frame1_tensor = torch.from_numpy(frame1.transpose(2, 0, 1)).float() / 255.0
        frame2_tensor = torch.from_numpy(frame2.transpose(2, 0, 1)).float() / 255.0
        flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1)).float()
        
        # Concatenate frames
        frames = torch.cat([frame1_tensor, frame2_tensor], dim=0)  # (6, H, W)
        
        return frames, flow_tensor


def get_dataloaders(batch_size=16):
    full_dataset = SyntheticOpticalFlowDataset(num_samples=200)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader
