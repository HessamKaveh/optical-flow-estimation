"""Evaluate on FlyingChairs val (640 pairs, EPE) and KITTI-2015 train (200 pairs, EPE and Fl-all).

Fl-all = % of valid pixels whose end-point error is > 3 px and > 5 % of the ground-truth magnitude.
Models trained here (from scratch, FlyingChairs only) are compared with torchvision's official RAFT
weights (trained on FlyingChairs + FlyingThings3D), which serve only as an external reference.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large
from torchvision.utils import flow_to_image

from datasets import FlyingChairs, Kitti15
from models import RAFTSmall, build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


class RaftLargeRef(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = raft_large(weights=Raft_Large_Weights.C_T_V2)

    @torch.no_grad()
    def predict(self, a, b, iters=None):
        return self.net(a / 127.5 - 1, b / 127.5 - 1, num_flow_updates=iters or 12)[-1]


def pad8(x):
    h, w = x.shape[-2:]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    return F.pad(x, [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2], mode="replicate"), (ph, pw)


def unpad(x, p):
    ph, pw = p
    h, w = x.shape[-2:]
    return x[..., ph // 2:h - (ph - ph // 2), pw // 2:w - (pw - pw // 2)]


def pad64(x):
    h, w = x.shape[-2:]
    H, W = (h + 63) // 64 * 64, (w + 63) // 64 * 64
    return F.interpolate(x, (H, W), mode="bilinear", align_corners=False), (h, w)


@torch.no_grad()
def infer(model, im1, im2, is_flownets):
    if is_flownets:   # FlowNetS needs sizes divisible by 64: resize, predict, resize flow back
        (a, hw), (b, _) = pad64(im1), pad64(im2)
        f = model.predict(a, b)
        f = F.interpolate(f, hw, mode="bilinear", align_corners=False)
        return f * torch.tensor([hw[1] / a.shape[-1], hw[0] / a.shape[-2]], device=f.device).view(1, 2, 1, 1)
    (a, p), (b, _) = pad8(im1), pad8(im2)
    return unpad(model.predict(a, b, iters=24), p)   # RAFT: 24 refinement iterations at test time


@torch.no_grad()
def evaluate(model, ds, dev, is_flownets, kitti=False):
    epes, outs = [], []
    for i in range(len(ds)):
        a, b, flow, valid = ds[i]
        a, b, flow, valid = a[None].to(dev), b[None].to(dev), flow[None].to(dev), valid[None].to(dev)
        pred = infer(model, a, b, is_flownets)
        epe = torch.norm(pred - flow, dim=1)[valid >= 0.5]
        if kitti:
            mag = torch.norm(flow, dim=1)[valid >= 0.5]
            outs.append(((epe > 3) & (epe / mag.clamp(min=1e-6) > 0.05)).float().cpu())
        epes.append(epe.cpu())
    epe = torch.cat(epes)
    r = {"EPE": epe.mean().item()}
    if kitti:
        r["EPE"] = float(np.mean([e.mean().item() for e in epes]))   # KITTI EPE: mean over images
        r["Fl-all"] = 100 * torch.cat(outs).mean().item()
    return r


def main():
    dev = torch.device("cuda")
    chairs, kitti = FlyingChairs("val"), Kitti15()
    models = {}
    for name in ("flownets", "raft_small"):
        p = os.path.join(ROOT, "checkpoints", f"{name}.pt")
        if os.path.exists(p):
            m = build(name).to(dev); m.load_state_dict(torch.load(p, map_location=dev)); m.eval()
            models[{"flownets": "FlowNetS (ours, Chairs)", "raft_small": "RAFT-small (ours, Chairs)"}[name]] = (m, name == "flownets")
    ref_s = RAFTSmall(weights=Raft_Small_Weights.C_T_V2).to(dev).eval()
    models["RAFT-small, torchvision C+T weights (reference)"] = (ref_s, False)
    models["RAFT-large, torchvision C+T weights (reference)"] = (RaftLargeRef().to(dev).eval(), False)

    table = {}
    for k, (m, fn) in models.items():
        table[k] = {"Chairs val EPE": evaluate(m, chairs, dev, fn)["EPE"]}
        kk = evaluate(m, kitti, dev, fn, kitti=True)
        table[k]["KITTI-15 EPE"], table[k]["KITTI-15 Fl-all (%)"] = kk["EPE"], kk["Fl-all"]
        print(k, {a: round(b, 3) for a, b in table[k].items()}, flush=True)
    os.makedirs(RES, exist_ok=True)
    json.dump(table, open(os.path.join(RES, "metrics.json"), "w"), indent=2)

    # qualitative: Chairs val and KITTI examples
    show = [k for k in table if "ours" in k]
    rows = [("chairs", chairs, 7), ("chairs", chairs, 311), ("kitti", kitti, 15), ("kitti", kitti, 120)]
    fig, axs = plt.subplots(len(rows), 2 + len(show), figsize=(3.6 * (2 + len(show)), 2.6 * len(rows)))
    for r, (nm, ds, i) in enumerate(rows):
        a, b, flow, valid = ds[i]
        axs[r, 0].imshow(a.permute(1, 2, 0).byte()); axs[r, 0].set_title(f"{nm} frame 1" if r in (0, 2) else "")
        gt = flow * valid[None] if nm == "kitti" else flow
        axs[r, 1].imshow(flow_to_image(gt).permute(1, 2, 0)); axs[r, 1].set_title("ground truth" if r in (0, 2) else "")
        for c, k in enumerate(show):
            m, fn = models[k]
            pr = infer(m, a[None].to(dev), b[None].to(dev), fn)[0].cpu()
            axs[r, 2 + c].imshow(flow_to_image(pr).permute(1, 2, 0)); axs[r, 2 + c].set_title(k.split(" (")[0] if r in (0, 2) else "")
        for ax in axs[r]:
            ax.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "flow_examples.png"), dpi=95); plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for name, lab in (("flownets", "FlowNetS"), ("raft_small", "RAFT-small")):
        p = os.path.join(RES, f"history_{name}.json")
        if os.path.exists(p):
            h = json.load(open(p)); ax.plot(h["step"], h["epe"], label=lab)
    ax.set(xlabel="training step", ylabel="train EPE (px)", title="FlyingChairs training", yscale="log"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RES, "training_curves.png"), dpi=110)


if __name__ == "__main__":
    main()
