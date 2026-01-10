# YOLOv8 Object Detection

This repository contains a minimal setup for training a YOLOv8 model for custom object detection using the Ultralytics framework.

## Overview
- Model architecture: YOLOv8 (nano variant)
- Training framework: Ultralytics YOLO
- Hardware: GPU-based training with mixed precision
- Dataset format: YOLO-style dataset with `data.yaml`

## Requirements
The project uses the following core dependencies:
- Python
- Ultralytics
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- PyYAML
- tqdm

All dependencies are listed in the `requirements.txt` file.

## Dataset
- Dataset is organized in YOLO format
- Configuration is defined in `data.yaml`
- Images and labels are split into training and validation sets

## Training
- Training uses a pretrained YOLOv8 checkpoint
- GPU acceleration is enabled
- Mixed precision (AMP) is used for better performance
- Training results are saved automatically

## Outputs
- `best.pt`: best-performing model based on validation metrics
- `last.pt`: model from the final training epoch
- Logs and metrics are stored under the `runs/` directory

