# Efficient ResNet with Knowledge Distillation for CIFAR-10

**NYU Tandon School of Engineering | Deep Learning — Fall 2026 | Project 1**

**Team:** Tensor Titans  
**Repository:** [tensor-titans-nyu/DL-PROJECT1](https://github.com/tensor-titans-nyu/DL-PROJECT1)

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

- Standard, wider, three-stage, and four-stage ResNets
- Pre-activation residual blocks and Squeeze-and-Excitation (SE) blocks
- MixUp, CutMix, Cutout, RandAugment, and random erasing
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

These values represent public leaderboard accuracies, not validation accuracies.

## Repository Contents

### ResNet Architecture Experiments — Gurjeet Kaur

| Notebook | Description | Public accuracy |
|---|---|---:|
| [notebook8dbb702849.ipynb](notebook8dbb702849.ipynb) | ResNet with TTA | **88.3%** |
| [notebook6200c2438c.ipynb](notebook6200c2438c.ipynb) | Multiple ResNet architecture experiments (V2–V9) | 88.2% |
| [four-stage-resnet-onecycle-experiment.ipynb](four-stage-resnet-onecycle-experiment.ipynb) | Four-stage ResNet with OneCycleLR | 88.0% |
| [preactivation-wide-resnet-cutout-ema.ipynb](preactivation-wide-resnet-cutout-ema.ipynb) | Pre-activation Wide ResNet with Cutout and EMA | 87.8% |
| [resnet-100-epoch-experiment.ipynb](resnet-100-epoch-experiment.ipynb) | ResNet training experiments with different configurations | 86.8% |
| [resnet-full-data-cutout-em.ipynb](resnet-full-data-cutout-em.ipynb) | Four-stage ResNet with Cutout and EMA | 86.4% |
| [notebooke88d98cc73.ipynb](notebooke88d98cc73.ipynb) | SE-ResNet with RandAugment, MixUp/CutMix, EMA, and TTA | 87.2% |
| [notebook4c6e54ec7a.ipynb](notebook4c6e54ec7a.ipynb) | Three-stage ResNet with MixUp, EMA, and TTA | 86.2% |

### Additional ResNet Experiments — Saakshi Narayan

| Notebook | Public accuracy |
|---|---:|
| [project1_saakshi1.ipynb](project1_saakshi1.ipynb) | 87.2% |
| [project1_saakshi2.ipynb](project1_saakshi2.ipynb) | 88.4% |
| [project1_saakshi3.ipynb](project1_saakshi3.ipynb) | 87.6% |
| [project1_saakshi4.ipynb](project1_saakshi4.ipynb) | 87.7% |

### Knowledge Distillation Experiments

The project report describes the following components:

| File | Purpose |
|---|---|
| `train.py` | Model training, distillation, and prediction |
| `cifar_kd_kaggle.ipynb` | Teacher training and single-teacher knowledge distillation |
| `cifar_2teachers_kaggle.ipynb` | Two-teacher knowledge distillation |

**Note:** These knowledge-distillation files still need to be added or confirmed in the repository.

## Reproducing the Experiments

### 1. Clone the repository

```bash
git clone https://github.com/tensor-titans-nyu/DL-PROJECT1.git
cd DL-PROJECT1
```

### 2. Set up the environment

The experiments were developed using Python, PyTorch, and GPU-enabled Kaggle or Google Colab environments.

Install the dependencies required by the selected notebook. A shared `requirements.txt` can be added once the team confirms the package versions.

### 3. Obtain the dataset

The experiments use the competition-provided CIFAR-10 dataset from the **NYU Tandon Deep Learning F26** Kaggle competition.

The dataset contains:

- 50,000 labeled training images
- 2,000 unlabeled test images
- 10 image classes
- RGB images of size 32 × 32

Download the dataset through authorized Kaggle competition access. The dataset is not redistributed in this repository.

### 4. Run the ResNet experiments

1. Open the relevant notebook in Kaggle, Google Colab, or a compatible Jupyter environment.
2. Enable GPU acceleration.
3. Configure the dataset path according to the notebook.
4. Run the preprocessing and model definition cells.
5. Train the model using the specified hyperparameters.
6. Run inference and generate the prediction CSV for Kaggle submission.

Some notebooks contain multiple architecture versions or experiments. Refer to their configuration cells for the corresponding settings.

### 5. Knowledge distillation

The project report includes the following example commands for the knowledge-distillation pipeline.

**Teacher training:**

```bash
python train.py --data DATA --out runs/t0 --depth 28 --width 10 \
    --epochs 150 --max_params 0
```

**Student training:**

```bash
python train.py --data DATA --out runs/s_kd --depth 22 --width 4 \
    --epochs 250 --teachers runs/t0/final.pt
```

Replace `DATA` with the dataset path. The commands and checkpoint names should be verified against the final uploaded training script.

## Key Findings

- Architecture changes alone did not consistently improve leaderboard accuracy.
- Training strategies such as augmentation, EMA, and TTA played an important role in model performance.
- Increasing model complexity or training for more epochs did not always lead to better results.
- Knowledge distillation achieved the highest public leaderboard accuracy among the team's reported submissions.
- The difference between public and private leaderboard scores highlights the importance of robust evaluation.
- Further controlled ablation studies and multiple random seeds could help evaluate the individual contributions of different training techniques.

## Limitations and Future Work

- No systematic hyperparameter sweep for knowledge distillation or augmentation strength.
- Limited seed-averaged experiments.
- No controlled ablation study of MixUp, CutMix, EMA, and TTA under identical training conditions.
- Future work could explore stochastic depth, SAM optimization, more diverse teacher architectures, and checkpoint averaging.

## Team Members

- Gurjeet Kaur
- Saakshi Narayan
- Krinal *(full name to be confirmed)*

## References

1. He et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
2. He et al. (2016). *Identity Mappings in Deep Residual Networks*. ECCV.
3. Zagoruyko and Komodakis (2016). *Wide Residual Networks*. BMVC.
4. Hinton et al. (2015). *Distilling the Knowledge in a Neural Network*.
5. Yun et al. (2019). *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*. ICCV.

---

**Course:** Deep Learning, Fall 2026  
**Institution:** New York University, Tandon School of Engineering  
**Project:** Efficient ResNet Architecture Challenge — CIFAR-10
