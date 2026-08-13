# Optical Flow Estimation

Convolutional neural network for estimating motion (optical flow) between
consecutive video frames. Applications: motion detection, video compression,
autonomous driving.

## Architecture
Encoder-decoder network: down-sample to extract features, up-sample to
predict per-pixel (dx, dy) flow vectors.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/train.py
python src/evaluate.py
```

## Results
- `results/training_curves.png` — MSE loss
- `results/opticalflow_results.png` — flow visualization

## Author
Hessam Kaveh — Research Fellow, Italian Institute of Technology
