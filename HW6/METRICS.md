# DATA 266 HW6 Metrics

All numbers below are from the Google Colab run (Tesla T4) of HW6_STL10_SSL.ipynb, the same run whose console output is in RUN_LOG.txt and results/run_log_raw.txt. The full set of per-epoch values is in results/hw6_metrics.json.

## Personal Parameters

| Field | Value | HW6 use |
| --- | ---: | --- |
| SID4 | 9486 | Reported |
| SEED | 9486 | random, NumPy, and torch (CPU and CUDA) seeds, the labeled subset, the SimCLR subset, DataLoader generators and worker seeds |
| SLICE | 486 | Query images are the first test images at index 486 or later that match each rule |
| HP_ID | 0 | Reported only |
| CLS_A | 6 | Query 1 class (horse) |
| CLS_B | 0 | Query 2 class (airplane) |

## Main Results

| Model | Encoder training | Labeled training | Test accuracy | Test loss |
| --- | --- | --- | ---: | ---: |
| A: Supervised | end to end with the labels | 15 epochs, all parameters | **34.80%** | 2.1619 |
| B: Rotation SSL + linear | rotation prediction, 15 epochs, 100,000 unlabeled images | 20 epochs, linear head only | **42.58%** | 1.6311 |
| C: SimCLR + linear | NT-Xent, 20 epochs, 20,000 unlabeled images | 20 epochs, linear head only | **50.45%** | 1.4060 |

Test accuracy is measured on all 8,000 STL-10 test images after the last epoch. There is no validation set, so no epoch was picked by looking at test accuracy. For reference, the supervised model's test accuracy was highest at epoch 12 (40.74%) and ended at 34.80%.

## Data and Subsets

| Item | Value |
| --- | --- |
| Dataset | STL-10 via torchvision (train 5,000, test 8,000, unlabeled 100,000), 96×96 RGB |
| Labeled subset (Parts A, B, C) | 500 images, 50 per class, chosen with np.random.default_rng(SEED); fingerprint 2a1412ac754f, the same in all three parts |
| Rotation pretraining set | all 100,000 unlabeled images |
| SimCLR pretraining set | 20,000 unlabeled images, chosen without replacement with np.random.default_rng(SEED) |
| Test set | all 8,000 test images (accuracy, curves, and nearest neighbors) |
| Input | native 96×96, normalized with mean (0.4467, 0.4398, 0.4066) and std (0.2603, 0.2566, 0.2713) computed from the labeled train split |

## Encoder and Heads

| Item | Value |
| --- | --- |
| Encoder (all three) | torchvision resnet18(weights=None), fc replaced with Identity, 512-d output, 11,176,512 parameters |
| Part A head | Linear 512 → 10, trained together with the encoder (11,181,642 trainable parameters) |
| Part B pretraining head | Linear 512 → 4 (rotation classes), removed after pretraining |
| Part C projection head | Linear 512 → 512, ReLU, Linear 512 → 128, removed after pretraining |
| Linear evaluation (B and C) | Linear 512 → 10, 5,130 trainable parameters; encoder frozen (requires_grad=False and kept in eval mode) |

## Hyperparameters

| Setting | Part A | Part B pretraining | Part C pretraining | Linear evaluation (B and C) |
| --- | --- | --- | --- | --- |
| Epochs | 15 | 15 | 20 | 20 |
| Batch size | 32 | 256 | 256 (drop_last) | 32 |
| Optimizer | Adam | Adam | Adam | Adam |
| Learning rate | 1e-3 | 3e-4 | 3e-4 | 1e-3 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| Steps per epoch | 16 | 391 | 78 | 16 |
| Augmentation | RandomCrop(96, padding=12), HorizontalFlip | RandomCrop(96, padding=12), HorizontalFlip, then a random 0/90/180/270° rotation | RandomResizedCrop(96, scale 0.6–1.0), HorizontalFlip(p=0.5), ColorJitter(0.4, 0.4, 0.4, 0.1), RandomGrayscale(p=0.2), two views per image | RandomCrop(96, padding=12), HorizontalFlip |
| Other | | | NT-Xent with cosine similarity, tau = 0.2 | |

## Training Progress

| Part | Start | End |
| --- | --- | --- |
| A supervised: train acc / test acc | 20.40% / 18.36% (epoch 1) | 60.80% / 34.80% (epoch 15) |
| B rotation pretraining: loss / rotation acc | 1.1396 / 45.23% (epoch 1) | 0.7292 / 62.58% (epoch 15) |
| B linear evaluation: train acc / test acc | 15.00% / 15.01% (epoch 1) | 45.00% / 42.58% (epoch 20) |
| C SimCLR pretraining: NT-Xent loss | 4.2217 (epoch 1) | 2.1757 (epoch 20) |
| C linear evaluation: train acc / test acc | 20.60% / 36.75% (epoch 1) | 67.60% / 50.45% (epoch 20) |

Curves for every part are in figures/. Rotation labels over all of training: 374,790 / 375,061 / 375,330 / 374,819 for 0°/90°/180°/270°.

## Verification Checks (all passed in the reported run)

| Check | Result |
| --- | --- |
| Image shapes and labels | every split is 3×96×96; train and test labels are 0–9; unlabeled labels are all -1 |
| Labeled subset | exactly 500 images, 50 per class, same fingerprint 2a1412ac754f in Parts A, B, and C |
| SimCLR subset | exactly 20,000 unique indices |
| No pretrained weights | weights=None; two encoders built with different seeds have different conv1 weights |
| Rotation labels | in range 0–3, all four used about equally |
| Frozen encoders | Part B and Part C: 120 encoder tensors (weights and BatchNorm buffers) compared before and after linear evaluation, 0 changed, 0 parameters with requires_grad=True |
| Cosine similarity | NT-Xent similarity matrix matches F.cosine_similarity (max difference 6.6e-7); test embeddings have norm 1.0000; top neighbor checked against F.cosine_similarity |
| Same queries | QUERY_INDICES = [498, 491, 486] used for all three encoders |
| Saved outputs | 8 figures, results/hw6_metrics.json, results/run_log_raw.txt |

## Part D: Nearest Neighbors (top 5 by cosine similarity)

| Query | Supervised | Rotation SSL | SimCLR |
| --- | --- | --- | --- |
| 498, horse | 3/5 (monkey, bird) | 3/5 (dog, deer) | 5/5 |
| 491, airplane | 5/5 | 5/5 | 5/5 |
| 486, ship | 2/5 (car, car, truck) | 0/5 (car, car, truck, truck, truck) | 3/5 (car, airplane) |
| Total same-class | 10/15 | 8/15 | 13/15 |

Wrong neighbors are listed in parentheses. Rotation SSL's top-5 cosine scores were 0.98–1.00 for every query, while SimCLR's were 0.88–0.92 for the horse and ship queries. With only three queries, these counts are examples, not a measured retrieval score.

## Hardware and Run Time

| Item | Value |
| --- | --- |
| Hardware | Google Colab, Tesla T4 GPU |
| Software | torch 2.11.0+cu130, torchvision 0.26.0+cu130 |
| Precision | fp32, cuDNN deterministic mode on |
| STL-10 download and load | 148.3 s (2.64 GB at about 30.8 MB/s) |
| Part A supervised | 95.9 s |
| Part B rotation pretraining | 1,476.5 s (about 98 s per epoch) |
| Part B linear evaluation | 123.5 s |
| Part C SimCLR pretraining | 1,694.8 s (about 85 s per epoch) |
| Part C linear evaluation | 124.0 s |
| Whole run | 05:48:15 to 06:50:19 UTC, about 62 minutes |
