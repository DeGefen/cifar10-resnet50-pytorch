# CIFAR-10 Image Classification in PyTorch

Final project for a Deep Learning course. The goal was to train and optimize a neural network on CIFAR-10, working step by step from a simple CNN to a fine-tuned, pre-trained ResNet50 and using bias and variance to choose each next experiment.

**Final result: 92.0% accuracy on the held-out test set**, up from 65.1% validation accuracy for the first baseline CNN.

## Dataset

CIFAR-10, 32x32 color images in 10 classes. The original training set is split 80/20 to add a validation set:

| Split | Images |
|---|---|
| Train | 40,000 |
| Validation | 10,000 |
| Test | 10,000 |

For ResNet50, images are resized to 224x224 and normalized with CIFAR-10 channel statistics. The training set also uses random horizontal flips.

## Experiments

| Attempt | Change | Train acc. | Val acc. |
|---|---|---|---|
| 1 | Baseline CNN, 2 conv layers | 74.19% | 65.09% |
| 2 | Added batch normalization (best of 4 architectures tried) | 77.62% | 69.32% |
| 3 | Random learning-rate search, 100 trials on a GPU cluster | 78.47% | 69.79% |
| 4 | 150 epochs | 90.90% | 69.85% |
| 5 | Dropout and weight decay | 63.93% | 60.06% |
| 6 | Up to 1000 epochs with early stopping | 70.36% | 60.80% |
| 7 | Pre-trained ResNet50 with a new 3-layer classifier head | 98.74% | 90.66% |
| 8.1 | CIFAR-10 normalization and horizontal flips | 97.16% | 91.09% |
| 8.2 | Dropout 0.2 | 97.28% | 91.65% |
| 8.3 | Label smoothing 0.1 | 97.60% | 91.63% |
| 8.4 | Dropout and label smoothing combined | 97.14% | 91.23% |
| 8.5 | Batch size 64 | 96.17% | 90.05% |
| 8.6 | Batch size 256 | 97.72% | 91.96% |
| 8.7 | Up to 100 epochs with early stopping | 98.06% | 92.31% |
| **8.8** | **All layers explicitly trainable (final model)** | **98.80%** | **92.61%** |

## Final model

- Pre-trained ResNet50, all parameters trainable
- Classifier head: dropout 0.2, then linear layers 2048 → 1024 → 512 → 10 with ReLU
- Adam, learning rate 0.001, no weight decay
- Cross-entropy loss with label smoothing 0.1
- Batch size 256, up to 100 epochs, early stopping with patience 5

## What I learned

- Attempts 1 to 6 used a Softmax layer together with `CrossEntropyLoss`, which already applies log-softmax. I caught this later and removed it from Attempt 7 onward.
- Adding depth to a small CNN did not help on its own. Batch normalization did, and the large jump came from transfer learning.
- Regularizing the small CNN reduced variance but raised bias, which pointed to the architecture as the limit.

## Running the notebook

```bash
pip install -r requirements.txt
jupyter notebook cifar10_resnet50.ipynb
```

CIFAR-10 downloads automatically. A GPU is strongly recommended. Longer experiments were run on a GPU cluster, and the notebook reads their saved logs from `results/` and the final weights from `trained/`. Those files are not included in this repository, so the saved cell outputs show the original results.
