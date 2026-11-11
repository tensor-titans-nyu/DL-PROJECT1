# Efficient ResNet with Knowledge Distillation for CIFAR-10

**NYU Tandon School of Engineering | Deep Learning — Fall 2026 | Project 1**

**Team:** Tensor Titans  
**GitHub Organization:** [tensor-titans-nyu](https://github.com/tensor-titans-nyu)

## Project Overview

This project explores the design and training of efficient Residual Neural Networks (ResNets) for CIFAR-10 image classification under a strict budget of **5 million trainable parameters**.

We experimented with different ResNet architectures, data augmentation techniques, training schedules, and knowledge distillation to improve classification accuracy while satisfying the competition's parameter constraint.

All models were trained from scratch using the competition-provided dataset, without external pretrained weights.

## Final Results

Our selected model is a **pre-activation Wide ResNet-22-4 (WRN-22-4)** trained using knowledge distillation from a larger WRN-28-10 teacher.

| Metric | Result |
|---|---|
| Final student architecture | Pre-activation WRN-22-4 |
| Trainable parameters | 4,298,970 |
| Teacher architecture | WRN-28-10 |
| Public leaderboard accuracy | **88.8%** |
| Private leaderboard accuracy | **86.1%** |
| Parameter limit | 5,000,000 |

The teacher model was trained from scratch and used only during student training. The submitted student model satisfies the parameter budget.

## Architecture Exploration

We evaluated multiple architecture and training configurations, including:

- Standard and wider ResNets
- Three-stage and four-stage ResNets
- Pre-activation residual blocks
- Squeeze-and-Excitation (SE) blocks
- MixUp, CutMix, Cutout, and random erasing
- OneCycle learning-rate scheduling
- Exponential Moving Average (EMA)
- Horizontal-flip Test-Time Augmentation (TTA)
- Single-teacher and two-teacher knowledge distillation

### Selected Experimental Results

| Experiment | Public accuracy |
|---|---:|
| One-teacher KD, WRN-22-4 (selected) | **88.8%** |
| Two-teacher KD, WRN-22-4 | 88.4% |
| ResNet + TTA | 88.3% |
| ResNet architecture experiments (best) | 88.2% |
| Four-stage ResNet + OneCycleLR | 88.0% |
| Pre-activation ResNet + Cutout + EMA | 87.8% |
| SE-ResNet + augmentation + EMA | 87.2% |
| Three-stage ResNet + MixUp + EMA | 86.2% |

These values are public leaderboard accuracies. They should not be interpreted as validation accuracy.

## Repository Contents

### ResNet Architecture Experiments — Gurjeet Kaur

| Notebook | Description | Public accuracy |
|---|---|---:|
| `notebook8dbb702849.ipynb` | ResNet with TTA | 88.3% |
| `notebook6200c2438c.ipynb` | Multiple ResNet architecture experiments (V2–V9) | Best: 88.2% |
| `four_stage_resnet_onecycle.ipynb` | Four-stage ResNet with OneCycleLR | 88.0% |
| `preactivation_resnet_cutout_ema.ipynb` | Pre-activation Wide ResNet with Cutout and EMA | 87.8% |
| `resnet_100_epoch_experiment.ipynb` | ResNet training experiments with different configurations | Best: 86.8% |
| `resnet_cutout_ema.ipynb` | Four-stage ResNet with Cutout and EMA | 86.4% |
| `notebooke88d98cc73.ipynb` | SE-ResNet with RandAugment, MixUp/CutMix, EMA, and TTA | 87.2% |
| `notebook4c6e54ec7a.ipynb` | Three-stage ResNet with MixUp, EMA, and TTA | 86.2% |

### Additional Team Experiments

| File | Description |
|---|---|
| `project1_saakshi1.ipynb` | ResNet experiment, public score 87.2% |
| `project1_saakshi2.ipynb` | ResNet experiment, public score 88.4% |
| `project1_saakshi3.ipynb` | ResNet experiment, public score 87.6% |
| `project1_saakshi4.ipynb` | ResNet experiment, public score 87.7% |
| `cifar_kd_kaggle.ipynb` | Teacher training and single-teacher knowledge distillation |
| `cifar_2teachers_kaggle.ipynb` | Two-teacher knowledge distillation |
| `train.py` | Model training, distillation, and prediction pipeline |

*The tables describe the project files; verify that each filename matches the uploaded repository before finalizing this README.*

## Reproducing the Experiments

### 1. Clone the repository

```bash
git clone https://github.com/tensor-titans-nyu/DL-PROJECT1.git
cd DL-PROJECT1
```

### 2. Set up the environment

Python and PyTorch with GPU support are recommended for model training.

Install the dependencies specified by the individual notebooks or the repository's `requirements.txt`, once available.

### 3. Obtain the dataset

The project uses the dataset supplied through the **NYU Tandon Deep Learning F26** Kaggle competition.

The dataset includes 50,000 labeled training images and 2,000 unlabeled test images.

Download the dataset using authorized competition access and configure the input paths in the relevant notebook or training script.

The dataset is not redistributed in this repository.

### 4. Run the experiments

Open the relevant Jupyter notebook in Kaggle, Google Colab, or another compatible Python environment.

For each experiment:

1. Configure the dataset paths and GPU runtime.
2. Run the model definition and preprocessing cells.
3. Train the model using the notebook's specified configuration.
4. Generate predictions using the available inference code.
5. Save predictions in the format required by the Kaggle competition.

Some notebooks contain multiple experimental versions. Refer to their configuration cells for the corresponding architectures and hyperparameters.

### 5. Knowledge distillation

The team's training script supports teacher and student training. Example commands from the project report:

**Teacher training**

```bash
python train.py --data DATA --out runs/t0 --depth 28 --width 10 \
    --epochs 150 --max_params 0
```

**Student training**

```bash
python train.py --data DATA --out runs/s_kd --depth 22 --width 4 \
    --epochs 250 --teachers runs/t0/final.pt
```

Replace `DATA` with the local dataset path. These commands should be checked against the uploaded `train.py` and the required checkpoint files.

## Key Findings

- Architecture changes alone did not consistently improve leaderboard performance.
- Data augmentation, training schedules, EMA, and TTA influenced model performance.
- Longer training did not always guarantee higher accuracy.
- Knowledge distillation achieved the team's highest public leaderboard accuracy.
- Public and private leaderboard scores differed, highlighting the importance of careful model evaluation.
- Controlled ablation studies and multiple random seeds could help distinguish meaningful improvements from experimental variability.

## Team Members

- Gurjeet Kaur
- Saakshi Narayan
- Krinal *(confirm full name)*

## References

- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- He et al., *Identity Mappings in Deep Residual Networks*, ECCV 2016.
- Zagoruyko and Komodakis, *Wide Residual Networks*, BMVC 2016.
- Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015.
- Yun et al., *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*, ICCV 2019.

---

**Course:** Deep Learning, Fall 2026  
**Institution:** New York University, Tandon School of Engineering  
**Project:** Efficient ResNet Architecture Challenge — CIFAR-10
