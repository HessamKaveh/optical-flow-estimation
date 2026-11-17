import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset import get_dataloaders
from model import OpticalFlowNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_opticalflow.pt")


def flow_to_color(flow):
    """Convert optical flow to a colour image for visualisation"""
    h, w = flow.shape[:2]
    flow_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    flow_image[..., 0] = angle * 180 / np.pi / 2
    flow_image[..., 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_image[..., 2] = 255
    
    flow_image = cv2.cvtColor(flow_image, cv2.COLOR_HSV2BGR)
    return flow_image


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_dataloaders(batch_size=4)

    model = OpticalFlowNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    frames, flow_gt = next(iter(val_loader))
    frames = frames.to(device)

    with torch.no_grad():
        flow_pred = model(frames)

    n_samples = min(3, frames.size(0))
    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3 * n_samples))

    for i in range(n_samples):
        frame1 = frames[i, :3].cpu().permute(1, 2, 0).numpy()
        frame2 = frames[i, 3:].cpu().permute(1, 2, 0).numpy()
        
        flow_gt_np = flow_gt[i].cpu().numpy().transpose(1, 2, 0)
        flow_pred_np = flow_pred[i].cpu().detach().numpy().transpose(1, 2, 0)

        axes[i, 0].imshow(frame1)
        axes[i, 0].set_title("Frame 1")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(frame2)
        axes[i, 1].set_title("Frame 2")
        axes[i, 1].axis("off")

        # Magnitude visualization
        mag_gt = np.linalg.norm(flow_gt_np, axis=2)
        mag_pred = np.linalg.norm(flow_pred_np, axis=2)

        axes[i, 2].imshow(mag_gt, cmap="hot")
        axes[i, 2].set_title("GT Flow Magnitude")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(mag_pred, cmap="hot")
        axes[i, 3].set_title("Pred Flow Magnitude")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "opticalflow_results.png"), dpi=150)
    print(f"Saved optical flow results to {os.path.join(RESULTS_DIR, 'opticalflow_results.png')}")


if __name__ == "__main__":
    import cv2
    evaluate()
