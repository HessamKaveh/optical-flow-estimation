import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import OpticalFlowNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(batch_size=16)

    model = OpticalFlowNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    num_epochs = 15
    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for frames, flow_gt in train_loader:
            frames, flow_gt = frames.to(device), flow_gt.to(device)

            optimizer.zero_grad()
            flow_pred = model(frames)
            loss = criterion(flow_pred, flow_gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * frames.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, flow_gt in val_loader:
                frames, flow_gt = frames.to(device), flow_gt.to(device)
                flow_pred = model(frames)
                loss = criterion(flow_pred, flow_gt)
                val_loss += loss.item() * frames.size(0)

        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_opticalflow.pt"))

    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Optical Flow Estimation Training")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"))
    print(f"\nBest validation loss: {best_loss:.6f}")


if __name__ == "__main__":
    train()
