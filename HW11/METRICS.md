# DATA 266 HW11 Metrics

All numbers come from the full run of `HW11_CLIP_Similarity_ZeroShot.ipynb` on my Apple M1 (see `results/hw11_metrics.json`, `results/part1_similarity_scores.csv`, and `results/part2_predictions.csv`). Similarity = cosine similarity of the L2-normalized CLIP image and text embeddings. The CLIP logit is 100 × cosine.

## Personal Parameters

| Field | Value | How it is calculated | HW11 use |
| --- | ---: | --- | --- |
| SID4 | 9486 | given | Reported |
| SEED | 9486 | SID4 | Python / NumPy / PyTorch seeds (only used for the jitter in the strip plot) |
| SLICE | 486 | SID4 % 1000 | Part 1: first 100 test images per class at index ≥ 486. Part 2: misclassified examples shown from index 486 |
| HP_ID | 0 | SID4 % 6 | Reported only |
| CLS_A | 6 | SID4 % 10 | Part 1 caption class: Shirt → "a photo of a shirt." |
| CLS_B | 0 | (CLS_A + 1 + (SID4 // 10) % 9) % 10 | Part 1 comparison class: T-shirt/top (AUC Shirt vs. T-shirt/top) |

## Setup

| Item | Value |
| --- | --- |
| Model | `openai/clip-vit-base-patch32`: 151,277,313 parameters (image side 87,849,216, text side 63,428,096), 512-d shared space, learned logit scale 100.00 |
| Preprocessing | `CLIPProcessor`: bicubic resize to 224, center crop 224×224, CLIP mean/std. FashionMNIST converted from 1-channel gray to RGB |
| Part 1 data | FashionMNIST test, 100 images per class (indices 486-1594), 1,000 images |
| Part 2 data | CIFAR-10 test, all 10,000 images (no training images used) |
| Prompts | main `"a photo of a {}."`, demo `"an image of a {}"`, ensemble of 8 templates (averaged normalized text embeddings) |
| Hardware | Apple M1, 8 GB, MPS. Python 3.13.9, torch 2.14.0, torchvision 0.29.0, transformers 5.21.0, scikit-learn 1.7.2 |

## Runtime

| Stage | Time |
| --- | ---: |
| Load CLIP | 3.3 s |
| Encode 1,000 FashionMNIST images | 14.8 s |
| Encode 10,000 CIFAR-10 images | 152.3 s (66 images/s) |
| Whole notebook (data already downloaded) | 3.0 min |

## Part 1: Similarity with "a photo of a shirt." (100 images per class)

| Class | Mean | Std | Min | Median | Max | In top 100 | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T-shirt/top | 0.2552 | 0.0202 | 0.1993 | 0.2611 | 0.2859 | 49 | 180.0 |
| **Shirt** | **0.2525** | 0.0171 | 0.2018 | 0.2547 | 0.2830 | **37** | 194.6 |
| Pullover | 0.2457 | 0.0134 | 0.1883 | 0.2479 | 0.2718 | 9 | 246.2 |
| Coat | 0.2389 | 0.0132 | 0.2105 | 0.2397 | 0.2693 | 2 | 314.2 |
| Dress | 0.2268 | 0.0138 | 0.1955 | 0.2258 | 0.2739 | 3 | 449.2 |
| Trouser | 0.2235 | 0.0078 | 0.1941 | 0.2247 | 0.2386 | 0 | 481.8 |
| Bag | 0.2126 | 0.0087 | 0.1941 | 0.2119 | 0.2359 | 0 | 621.0 |
| Sandal | 0.1986 | 0.0092 | 0.1727 | 0.2000 | 0.2160 | 0 | 786.2 |
| Sneaker | 0.1929 | 0.0083 | 0.1702 | 0.1924 | 0.2173 | 0 | 853.8 |
| Ankle boot | 0.1905 | 0.0086 | 0.1715 | 0.1908 | 0.2113 | 0 | 877.9 |

| Summary | Value |
| --- | ---: |
| Score range (all 1,000 images) | 0.1702 – 0.2859 |
| ROC AUC, Shirt vs. all other classes | 0.840 |
| ROC AUC, Shirt vs. T-shirt/top | 0.429 |
| Precision of the top 100 (share that is Shirt) | 0.37 |
| Most similar image | test #1049, T-shirt/top, 0.2859 |

### Same test with a caption for every class

| Caption | AUC (own class vs. rest) | Own class in top 100 |
| --- | ---: | ---: |
| a photo of a ankle boot. | 0.992 | 0.89 |
| a photo of a trouser. | 0.975 | 0.83 |
| a photo of a sneaker. | 0.960 | 0.65 |
| a photo of a bag. | 0.939 | 0.70 |
| a photo of a coat. | 0.918 | 0.48 |
| a photo of a pullover. | 0.906 | 0.51 |
| a photo of a dress. | 0.897 | 0.50 |
| a photo of a t-shirt/top. | 0.873 | 0.55 |
| **a photo of a shirt.** | **0.840** | **0.37** |
| a photo of a sandal. | 0.734 | 0.30 |

## Part 2: Zero-Shot CIFAR-10 (10,000 test images)

| Prompt set | Accuracy | Correct |
| --- | ---: | ---: |
| main: "a photo of a {}." | 88.33% | 8,833 |
| demo: "an image of a {}" | 89.06% | 8,906 |
| ensemble of 8 templates | 89.17% | 8,917 |
| (reference) teacher CNN from Demo_11_Knowledge_Distillation, supervised | 73.06% | – |

### Per-class accuracy (%)

| Class | main | demo | ensemble | Precision (main) |
| --- | ---: | ---: | ---: | ---: |
| horse | 97.1 | 96.8 | 97.6 | 83.1 |
| ship | 95.9 | 95.7 | 96.4 | 92.7 |
| truck | 95.0 | 96.8 | 94.8 | 92.4 |
| automobile | 93.3 | 93.7 | 93.3 | 93.7 |
| bird | 89.8 | 87.5 | 87.7 | 78.0 |
| dog | 89.1 | 89.0 | 88.6 | 81.7 |
| airplane | 89.0 | 89.5 | 90.8 | 95.4 |
| cat | 85.7 | 87.0 | 85.8 | 82.0 |
| deer | 76.9 | 78.9 | 79.8 | 93.3 |
| frog | 71.5 | 75.7 | 76.9 | 98.1 |

### Confusion matrix (main prompt, counts; rows = true, columns = predicted)

| true \ pred | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| airplane | **890** | 7 | 60 | 0 | 0 | 1 | 0 | 0 | 31 | 11 |
| automobile | 2 | **933** | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 61 |
| bird | 10 | 1 | **898** | 19 | 22 | 16 | 3 | 30 | 1 | 0 |
| cat | 2 | 4 | 24 | **857** | 7 | 83 | 6 | 10 | 6 | 1 |
| deer | 2 | 1 | 36 | 17 | **769** | 31 | 4 | 137 | 3 | 0 |
| dog | 0 | 1 | 22 | 63 | 7 | **891** | 1 | 12 | 3 | 0 |
| frog | 1 | 3 | 104 | 85 | 14 | 57 | **715** | 8 | 13 | 0 |
| horse | 2 | 1 | 5 | 4 | 5 | 9 | 0 | **971** | 3 | 0 |
| ship | 24 | 8 | 2 | 0 | 0 | 2 | 0 | 0 | **959** | 5 |
| truck | 0 | 37 | 0 | 0 | 0 | 0 | 0 | 0 | 13 | **950** |

### Most common mistakes

| True | Predicted as | Images | % of true class |
| --- | --- | ---: | ---: |
| deer | horse | 137 | 13.7 |
| frog | bird | 104 | 10.4 |
| frog | cat | 85 | 8.5 |
| cat | dog | 83 | 8.3 |
| dog | cat | 63 | 6.3 |
| automobile | truck | 61 | 6.1 |
| airplane | bird | 60 | 6.0 |
| frog | dog | 57 | 5.7 |

Mean top-1 softmax probability: 0.875 for correct predictions, 0.534 for wrong ones.
