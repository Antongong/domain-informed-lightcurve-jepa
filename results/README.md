# Results

This directory contains lightweight artifacts from the paper experiments. Large model checkpoints, frozen StarEmbed embedding matrices, and generated benchmark work directories are excluded from git.

## EXP21

`exp21/` corresponds to the no-overlap mean-pooling model:

```text
EXP21_lejepa_mean_pooling_no_group_no_starembed_overlap
```

Included files:

- `pretraining/`: final config, training metrics, and model/optimization details.
- `starembed/logistic_knn/`: logistic-regression and k-NN classification reports.
- `starembed/random_forest/`: random-forest summary and seed-42 report.
- `starembed/mlp/`: MLP sweep summary and selected run.
- `starembed/fewshot/`: bootstrap linear-probe reports for 1, 5, 20, and 100 examples per class.
- `figures/`: representative UMAP, similarity-search, and parameter-recovery figures.

## Synthetic Recovery

`synthetic_recovery/` contains CSV summaries for sinusoid injection/recovery experiments.
