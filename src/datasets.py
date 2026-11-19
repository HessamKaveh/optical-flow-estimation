"""FlyingChairs (Dosovitskiy et al. 2015) and KITTI-2015 flow datasets.

FlyingChairs: 22,872 synthetic 384x512 pairs; the official FlyingChairs_train_val.txt split gives
22,232 training and 640 validation pairs. Dense ground truth (.flo).
KITTI-2015 training: 200 real driving pairs (~375x1242) with sparse LiDAR-based ground truth
(flow_occ); used here only for cross-dataset evaluation (no training on KITTI).
"""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DATA = os.environ.get("FLOW_ROOT", os.path.expanduser("~/datasets/portfolio"))


def read_flo(path):
    with open(path, "rb") as f:
        assert np.fromfile(f, np.float32, 1)[0] == 202021.25, "bad .flo file"
        w, h = np.fromfile(f, np.int32, 2)
        return np.fromfile(f, np.float32, 2 * w * h).reshape(h, w, 2)


def read_kitti_flow(path):
    f = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)
    flow, valid = (f[:, :, :2] - 2 ** 15) / 64.0, f[:, :, 2]
    return flow, valid


def rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def to_t(img):
    return torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1).copy())


class FlyingChairs(Dataset):
    def __init__(self, split="train", crop=(368, 496), aug=True):
        root = os.path.join(DATA, "flyingchairs", "FlyingChairs_release", "data")
        labels = np.loadtxt(os.path.join(DATA, "flyingchairs", "FlyingChairs_train_val.txt"), dtype=np.int32)
        ids = np.where(labels == (1 if split == "train" else 2))[0] + 1
        self.items = [tuple(os.path.join(root, f"{i:05d}_{s}") for s in ("img1.ppm", "img2.ppm", "flow.flo"))
                      for i in ids]
        self.train, self.crop, self.aug = split == "train", crop, aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p1, p2, pf = self.items[i]
        im1, im2, flow = rgb(p1), rgb(p2), read_flo(pf)
        if self.train and self.aug:
            # photometric: same random colour jitter on both frames (+ small per-frame brightness change)
            g = np.random.uniform(0.7, 1.3, 3).astype(np.float32)
            im1 = np.clip(im1 * g * np.random.uniform(0.9, 1.1), 0, 255)
            im2 = np.clip(im2 * g * np.random.uniform(0.9, 1.1), 0, 255)
            # spatial: random rescale, flips and crop
            s = 2 ** np.random.uniform(-0.1, 0.6)
            if np.random.rand() < 0.8:
                h, w = im1.shape[:2]
                nh, nw = max(int(h * s + 0.5), self.crop[0] + 8), max(int(w * s + 0.5), self.crop[1] + 8)
                sx, sy = nw / w, nh / h
                im1 = cv2.resize(im1, (nw, nh), interpolation=cv2.INTER_LINEAR)
                im2 = cv2.resize(im2, (nw, nh), interpolation=cv2.INTER_LINEAR)
                flow = cv2.resize(flow, (nw, nh), interpolation=cv2.INTER_LINEAR) * [sx, sy]
            if random.random() < 0.5:
                im1, im2, flow = im1[:, ::-1], im2[:, ::-1], flow[:, ::-1] * [-1, 1]
            if random.random() < 0.1:
                im1, im2, flow = im1[::-1], im2[::-1], flow[::-1] * [1, -1]
            h, w = im1.shape[:2]
            y, x = random.randint(0, h - self.crop[0]), random.randint(0, w - self.crop[1])
            sl = (slice(y, y + self.crop[0]), slice(x, x + self.crop[1]))
            im1, im2, flow = im1[sl], im2[sl], flow[sl]
        valid = np.ones(flow.shape[:2], np.float32)
        return to_t(im1), to_t(im2), to_t(flow), torch.from_numpy(valid)


class Kitti15(Dataset):
    def __init__(self):
        root = os.path.join(DATA, "kitti", "training")
        self.items = [(os.path.join(root, "image_2", f"{i:06d}_10.png"), os.path.join(root, "image_2", f"{i:06d}_11.png"),
                       os.path.join(root, "flow_occ", f"{i:06d}_10.png")) for i in range(200)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p1, p2, pf = self.items[i]
        flow, valid = read_kitti_flow(pf)
        return to_t(rgb(p1)), to_t(rgb(p2)), to_t(flow), torch.from_numpy(valid)
