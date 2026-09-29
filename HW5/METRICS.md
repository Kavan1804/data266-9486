# DATA 266 HW5 Metrics

The numbers below are from the final Google Colab GPU run (Tesla T4) of HW5_LoRA_Fine_Tuning.ipynb, a single Run all in a fresh runtime (execution counts 1 through 27). See RUN_LOG.txt for the full console output, and for how this run compares to the earlier Colab runs and the local CPU runs I used while building the notebook.

## Personal Parameters

| Field | Value | HW5 use |
| --- | ---: | --- |
| SID4 | 9486 | Reported |
| SEED | 9486 | Used for random, NumPy, PyTorch (CPU and CUDA), and transformers.set_seed; also passed to the Trainer as seed and data_seed, and used to shuffle the training split |
| SLICE | 486 | Picks the first evaluation dialogue (test index 486); the second is the next test row with a different dialogue (index 489) |
| HP_ID | 0 | Reported only, HW5 has no HP_ID arm |
| CLS_A | 6 | Reported only, not used in HW5 |
| CLS_B | 0 | Reported only, not used in HW5 |

## Environment

| Setting | Value |
| --- | --- |
| Runtime | Google Colab, Tesla T4 GPU, fresh runtime |
| torch | 2.11.0+cu128 |
| transformers | 5.16.1 |
| Base model | google/flan-t5-small, 76,961,152 parameters, fp32, no quantization |
| Hugging Face login | not needed, model and dataset are public |

The install cell uninstalls Colab's preinstalled torchao (0.10), because the current peft release raises an ImportError for torchao versions below 0.16. The notebook does not print the peft version, so it is not recorded here.

## Task 1: Data Preparation

| Setting | Value |
| --- | --- |
| Dataset | neil-code/dialogsum-test |
| Splits | train 1,999, validation 499, test 499 (columns: id, dialogue, summary, topic) |
| Input / target | dialogue wrapped in "Summarize the following conversation. ... Summary:" / summary |
| Training subset | 1,000 examples, from the train split shuffled with SEED |
| Evaluation dialogues | test[486] (test_162_1) and test[489] (test_163_1) |
| Leakage check | neither evaluation dialogue appears in the training subset (assertion passed) |
| Max source / target length | 512 / 128 tokens, truncation only, dynamic padding in the collator |
| Processed samples shown | train_1804 (163 input tokens, 27 label tokens) and train_918 (386 input tokens, 64 label tokens) |

## Task 3: LoRA Parameters

LoRA was applied to the T5 query and value projections (q and v). The notebook confirmed both names exist, each found in 24 attention layers.

| Config | r | lora_alpha | lora_dropout | alpha / r | Total parameters | Trainable parameters | Trainable % | Adapter file size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main run | 4 | 16 | 0.1 | 4 | 77,133,184 | 172,032 | 0.2230% | 684.8 KB |
| Rank experiment | 16 | 32 | 0.1 | 2 | 77,649,280 | 688,128 | 0.8862% | 2,701.1 KB |

Total parameters include the LoRA weights, which is why the total goes up with rank. r=16 has exactly four times the trainable parameters of r=4.

## Task 4: Training Budget

Both configurations used the same budget.

| Setting | Value |
| --- | --- |
| Training subset size | 1,000 examples |
| Epochs | 2 |
| Batch size | 8 |
| Optimizer steps | 250 |
| Learning rate | 1e-3, 10 warmup steps |
| Weight decay | 0.0 |
| Precision | fp32 (fp16 off, since T5 is unstable in fp16) |
| Seed | 9486 (seed and data_seed) |
| Trainer / collator | Seq2SeqTrainer / DataCollatorForSeq2Seq (label_pad_token_id=-100) |

### Training loss

| Step | Epoch | r=4 loss | r=16 loss |
| ---: | ---: | ---: | ---: |
| 25 | 0.20 | 2.1567 | 2.0876 |
| 50 | 0.40 | 1.6888 | 1.6578 |
| 75 | 0.60 | 1.5910 | 1.5660 |
| 100 | 0.80 | 1.5677 | 1.5505 |
| 125 | 1.00 | 1.6214 | 1.6044 |
| 150 | 1.20 | 1.5611 | 1.5348 |
| 175 | 1.40 | 1.5971 | 1.5654 |
| 200 | 1.60 | 1.5550 | 1.5265 |
| 225 | 1.80 | 1.4628 | 1.4365 |
| 250 | 2.00 | 1.5171 | 1.4858 |
| **Final training loss (average over the run)** | | **1.6319** | **1.6015** |
| Training runtime | | 66.8 s | 64.6 s |

r=16 had a lower loss than r=4 at every logged step.

### Saved adapters

| Path | r | lora_alpha | target_modules | adapter_model.safetensors |
| --- | ---: | ---: | --- | ---: |
| outputs/dialogsum_lora_r4 | 4 | 16 | q, v | 684.8 KB |
| outputs/dialogsum_lora_r16 | 16 | 32 | q, v | 2,701.1 KB |

Both adapters were reloaded from disk onto a fresh copy of flan-t5-small with PeftModel.from_pretrained and merged before inference. They were saved to the Colab instance's disk and are not in this repository.

## Tasks 2, 5, 6, and 7: Generated Summaries

All generation used greedy decoding: do_sample=False, num_beams=1, max_new_tokens=100. The full text of every output is in RUN_LOG.txt and findings.pdf.

| Example | Baseline | LoRA r=4 | LoRA r=16 |
| --- | --- | --- | --- |
| 1 (band, test_162_1) | A question copied from the dialogue, not a summary | Summary style and keeps the Vanilla Ice detail, but calls both speakers "Ernie" | Summary style, but repeats "Ernie and Ernie are excited" twice and drops Vanilla Ice |
| 2 (New Orleans, test_163_1) | A question copied from the dialogue, not a summary | Uses #Person# tags and mentions the theater, but says they want to go to New Orleans and misses the decision | Mentions both speakers, but says #Person2# wants the jazz club and riverboat tour (both rejected), no theater |

No numerical quality metric (such as ROUGE) was computed, so the comparison above is qualitative only. On these two examples, fine-tuning clearly changed the output from a copied question into a summary-style sentence. r=16's lower training loss did not lead to better summaries: r=4 was similar or slightly better on both examples.
