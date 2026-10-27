# Checkpoints

`HW4a_BERT_vLLM.ipynb` saves the best-epoch BERT classifier (lowest validation loss) to
`checkpoints/<RUN_MODE>/hw4a_bert_best/` with `save_pretrained`. The folder also holds the tokenizer
and a `training_config.json` with the full configuration, the best epoch, and how it was run
(`HW4a_BERT_vLLM.ipynb` run top to bottom; `RUN_MODE` is `gpu` automatically when CUDA is visible).

The weight file (`model.safetensors`, about 440 MB) is git-ignored because it is too large for GitHub.
The config and tokenizer files are small and are kept. `checkpoints/smoke/` only holds the smoke-test
model trained on 48 rows, which is not a real result.

HW4.5 doesn't save checkpoints. Its runs are throughput and memory measurements, and their accuracies
are recorded in `results/gpu/part_b_techniques.csv`.
