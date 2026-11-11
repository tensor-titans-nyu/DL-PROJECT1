# Reproduce the selected Wide ResNet

The [selected Kaggle run](https://www.kaggle.com/code/krinal9571/cifer?scriptVersionId=366323895) uses a single pre-activation WRN-22-4 student with 4,298,970 trainable parameters. Kaggle reported 88.8% public accuracy and 86.1% private accuracy. The selected notebook contains the training logs, final parameter count, and leaderboard plot.

## Setup

Use a GPU environment with Python, PyTorch, torchvision, NumPy, pandas, and Pillow. Set `DATA` to the official competition data directory containing `train_images/`, `test_images/`, `train_labels.csv`, `test.csv`, and `sample_submission.csv`. Obtain the data through the course Kaggle competition.

## Selected predictions

Run the following commands from a clean working directory containing `train.py`:

```bash
python train.py --data "$DATA" --out runs/teacher --depth 28 --width 10 \
  --epochs 100 --max_params 0 --seed 0 --workers 2
python train.py --data "$DATA" --out runs/kd_s1 --depth 22 --width 4 \
  --epochs 200 --seed 1 --workers 2 --teachers runs/teacher/final.pt
cp runs/kd_s1/submission_ema_tta.csv submission.csv
```

The teacher trains from scratch on the official training data. The student uses SGD with Nesterov momentum 0.9, a batch size of 256, a peak learning rate of 0.2, five epochs of warmup and cosine decay, and distillation with temperature 2 and loss weight 0.9. At inference, the student's EMA weights produce probabilities for each image and its horizontal flip. The script writes 2,000 class predictions in sample-submission order. Kaggle supplies the test accuracy because evaluation labels are hidden.

## Other WRN runs

`cifar_baseline_kaggle.ipynb` contains the early 30-epoch baseline. `cifar_2teachers_kaggle.ipynb` records the two-teacher experiment. Its Kaggle inputs reference checkpoints produced by the team. To rebuild the additional teacher locally, use the first command above with `--epochs 150 --out runs/teacher_hpc`. Then use both teacher checkpoints after `--teachers`, with `--epochs 150` and seed 21 or 22 as logged in the notebook.

## References

- He et al., [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027).
- Zagoruyko and Komodakis, [Wide Residual Networks](https://arxiv.org/abs/1605.07146).
- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).
- Müller and Hutter, [TrivialAugment](https://arxiv.org/abs/2103.10158).
- Yun et al., [CutMix](https://arxiv.org/abs/1905.04899).
- Zhong et al., [Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896).
- Loshchilov and Hutter, [SGDR](https://arxiv.org/abs/1608.03983).
