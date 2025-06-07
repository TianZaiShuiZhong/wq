# Tennis Ball GAN Generator

This project implements a conditional GAN to generate synthetic tennis ball images under various conditions (sunny/shadow/deformed/occluded/dirty).

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- opencv-python
- tqdm

## Installation
```bash
pip install torch torchvision numpy matplotlib opencv-python tqdm
```

## Training
To train the model:
```bash
python train.py
```

Training progress will be saved in `results/` directory.

## Generation
To generate images with specific conditions:
```bash
python generate.py --conditions [1 0 0 0 0] --num 10 --output my_samples
```

### Condition Parameters
The condition vector should have 5 values (1 or 0) representing:
1. Sunny
2. Shadow 
3. Deformed
4. Occluded
5. Dirty

Example: `[1 0 0 0 0]` generates sunny condition images.

## Model Checkpoints
Trained models are saved in `models/` directory every 10 epochs.

## Dataset
The training dataset should be placed in `labelimg_data/images/train/` with JPG format.
