# DATA 266 HW5

## Overview

Homework 5 fine-tunes google/flan-t5-small for dialogue summarization on DialogSum (neil-code/dialogsum-test) using LoRA through the Hugging Face PEFT library. The notebook follows the flow of the professor's Demo_5_LoRA.ipynb, adapted from causal language modeling to sequence-to-sequence summarization. It covers:

1. Data preparation: dialogue as the encoder input and summary as the decoder labels, a 1,000-example training subset shuffled with SEED, and two held-out test dialogues for evaluation.
2. Baseline inference with the untouched base model, using greedy decoding.
3. LoRA adapters on the T5 q and v projections (r=4, lora_alpha=16, lora_dropout=0.1).
4. Fine-tuning with Seq2SeqTrainer and DataCollatorForSeq2Seq, then saving the adapter.
5. Reloading the saved adapter and running inference on the same two dialogues.
6. A before and after comparison table.
7. A rank experiment comparing r=4 against r=16 (lora_alpha=32) under the same training budget.

The submitted results come from a single Run all on Google Colab with a Tesla T4 GPU, in a fresh runtime.

## Personal Parameters

| Field | Value | HW5 use |
| --- | ---: | --- |
| SID4 | 9486 | Reported |
| SEED | 9486 | Python, NumPy, PyTorch, transformers, Trainer seed and data_seed, training subset shuffle |
| SLICE | 486 | Selects the first evaluation dialogue (test index 486); the second is the next test row with a different dialogue (489) |
| HP_ID | 0 | Reported only |
| CLS_A | 6 | Reported only |
| CLS_B | 0 | Reported only |

## Folder Structure

- `HW5_LoRA_Fine_Tuning.ipynb`: the main submission notebook, with outputs from the Colab GPU run
- `Demo_5_LoRA.ipynb`: the professor's reference demo, not modified
- `METRICS.md`: parameter counts, training budget, training losses, and a summary of the outputs
- `RUN_LOG.txt`: step-by-step run history, the full training logs, and every generated summary
- `findings.pdf`: short findings write-up
- `AI_USE.md`: AI use disclosure
- `README.md`: this file

The notebook saves the adapters to `outputs/dialogsum_lora_r4` and `outputs/dialogsum_lora_r16` next to wherever it runs. They were written to Colab's disk during the run and are not included in this folder.

## Run Instructions

Open `HW5_LoRA_Fine_Tuning.ipynb` in Colab with a GPU runtime and run all cells. No Hugging Face login is needed. The install cell uninstalls Colab's preinstalled torchao, because the current peft release rejects versions below 0.16. If the notebook was already partly run in the same session before that cell, restart the runtime first.
