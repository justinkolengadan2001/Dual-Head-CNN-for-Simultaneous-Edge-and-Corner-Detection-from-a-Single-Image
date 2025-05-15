# Dual-Head CNN for Simultaneous Edge and Corner Detection from a Single Image

This project implements a deep learning-based dual-head convolutional neural network (CNN) architecture for **simultaneous edge and corner detection** from a single high-resolution image. The method utilized formulates edge and corner detection as a **joint pixel-wise binary classification task** using a shared encoder with two decoder heads.

### Problem Statement
- Detect edge and corner features from 512×512 image crops extracted from a single 5616×3744 image.
- Ground truth labels are auto-generated using classical detectors (Canny and Harris).
- Tackles **sparse supervision**, **class imbalance**, and **generalization challenges** with only one image as input.

### Model Architectures
- **UNetDualHead**: Lightweight U-Net variant with skip connections.
- **ResNetLiteDualHeadSkip**: Residual-style encoder with symmetric upsampling path.
- Dual heads output binary edge and corner maps simultaneously.

### Dataset and Preprocessing
- Generated 512×512 overlapping patches with stride = 256.
- Filtered out background-heavy crops (<5% non-zero pixels).
- Retained 104 structure-rich image crops.
- Canny and Harris detectors used for ground truth label generation.

### Ablation Studies
- **Loss functions**: BCE, Weighted BCE, Focal Loss.
- **Optimizers**: Adam, AdamW, SGD with momentum.
- **Activations**: ReLU, Leaky ReLU, Mish.
- **Metrics**: IoU and F1-Score per task (edge and corner).
- **Experiments**: Multiple learning rates and training configurations tested.

### Results
- UNetDualHead + AdamW: Best for edge detection.
- ResNetDualHeadSkip + Mish: Best for corner detection.
- Plots and evaluation metrics provided in `/results`.

### Project Structure
├── model/ # Model definitions for UNet and ResNet variants
├── data/ # Preprocessed training patches
├── utils/ # Loss functions, metrics, augmentation scripts
├── results/ # Evaluation plots, predictions
├── train.py # Training loop
├── evaluate.py # Evaluation script
├── requirements.txt # Dependencies
└── README.md # Project description

### Key Takeaways
- Demonstrates learning from **sparse and synthetic labels**.
- Effectively handles class imbalance with loss weighting and focal loss.
- Highlights ethical considerations of deploying edge/corner detectors in sensitive systems.

### For full model visualizations, evaluation results, and plots, please explore the repository.