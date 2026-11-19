"""Train FlowNetS or RAFT-small from scratch on FlyingChairs (same data, augmentation and step budget).

    python src/train.py --model raft_small --steps 60000 --batch 6
    python src/train.py --model flownets   --steps 60000 --batch 8

Losses: FlowNetS uses the multi-scale EPE of the original paper (weights 0.32/0.08/0.02/0.01/0.005,
targets downsampled and divided by 20); RAFT uses the exponentially weighted sequence loss (gamma 0.8).
"""
import argparse
import copy
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import FlyingChairs
from models import FlowNetS, build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def flownets_loss(preds, flow):
    w = [0.32, 0.08, 0.02, 0.01, 0.005]
    tgt = flow / FlowNetS.DIV_FLOW
    loss = 0
    for p, wi in zip(preds, w):
        t = F.adaptive_avg_pool2d(tgt, p.shape[-2:])
        loss = loss + wi * torch.norm(p - t, dim=1).mean()
    return loss


def raft_loss(preds, flow, valid, gamma=0.8, max_flow=400):
    mag = torch.norm(flow, dim=1)
    v = (valid >= 0.5) & (mag < max_flow)
    n = len(preds)
    loss = 0
    for i, p in enumerate(preds):
        loss = loss + gamma ** (n - i - 1) * ((p - flow).abs().sum(1) * v).sum() / v.sum().clamp(min=1)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["raft_small", "flownets"], default="raft_small")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--hours", type=float, default=None, help="time budget; overrides --steps")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    lr = args.lr or (4e-4 if args.model == "raft_small" else 1e-4)

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    dl = DataLoader(FlyingChairs("train"), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    model = build(args.model).to(dev)
    if args.hours and not args.resume:
        # time budget: measure the real throughput (data loading included), then pick the step count
        init = copy.deepcopy(model.state_dict())
        probe = torch.optim.SGD(model.parameters(), lr=0.0)
        it_dl = iter(dl)
        for k in range(50):
            if k == 10:
                torch.cuda.synchronize(); t_start = time.time()
            im1, im2, flow, valid = next(it_dl)
            if args.model == "flownets":
                im1, im2 = im1[..., :320, :448], im2[..., :320, :448]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = model(im1.to(dev), im2.to(dev))
            l = sum(p.float().abs().mean() for p in preds)
            probe.zero_grad(); l.backward(); probe.step()
        torch.cuda.synchronize()
        sec_it = (time.time() - t_start) / 40
        model.load_state_dict(init); del probe, it_dl
        args.steps = int(args.hours * 3600 / sec_it) // 500 * 500
        print(f"[{args.model}] {sec_it:.3f} s/it -> {args.steps} steps in {args.hours} h", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, args.steps + 100, pct_start=0.05,
                                                cycle_momentum=False, anneal_strategy="linear")
    scaler = torch.amp.GradScaler()
    ck = os.path.join(ROOT, "checkpoints", f"{args.model}.pt")
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    step, hist = 0, {"step": [], "loss": [], "epe": []}
    if args.resume and os.path.exists(ck + ".state"):
        st = torch.load(ck + ".state", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, hist = st["step"], st["hist"]

    t0, run_l, run_e, n = time.time(), 0.0, 0.0, 0
    while step < args.steps:
        for im1, im2, flow, valid in dl:
            if args.model == "flownets":  # FlowNetS needs H, W divisible by 64: 368x496 random crop -> 320x448
                im1, im2, flow, valid = (t[..., :320, :448] for t in (im1, im2, flow, valid))
            im1, im2 = im1.to(dev, non_blocking=True), im2.to(dev, non_blocking=True)
            flow, valid = flow.to(dev, non_blocking=True), valid.to(dev, non_blocking=True)
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = model(im1, im2)
            preds = [p.float() for p in preds]
            if args.model == "flownets":
                loss = flownets_loss(preds, flow)
                full = F.interpolate(preds[0], scale_factor=4, mode="bilinear") * FlowNetS.DIV_FLOW
            else:
                loss = raft_loss(preds, flow, valid)
                full = preds[-1]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            run_l += loss.item(); run_e += torch.norm(full.detach() - flow, dim=1).mean().item(); n += 1
            if step % 500 == 0:
                hist["step"].append(step); hist["loss"].append(run_l / n); hist["epe"].append(run_e / n)
                print(f"[{args.model}] step {step}/{args.steps} loss {run_l / n:.3f} train-EPE {run_e / n:.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f} min", flush=True)
                run_l, run_e, n = 0.0, 0.0, 0
            if step % 5000 == 0 or step >= args.steps:
                torch.save(model.state_dict(), ck)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "hist": hist}, ck + ".state")
            if step >= args.steps:
                break
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(hist, open(os.path.join(ROOT, "results", f"history_{args.model}.json"), "w"))


if __name__ == "__main__":
    main()
