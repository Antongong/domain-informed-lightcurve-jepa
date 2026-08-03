# Bootstrap evaluation pipeline

Two stages, run in order:

1. `get_predictions.py` — for each of the 9 train%/test% combos and each of the 4
   classifiers (logistic, knn, rf, mlp), regenerates one real set of test-set
   predictions (refitting sklearn models once with their already-known
   hyperparameters; reloading the MLP checkpoint with no retraining) and saves
   `(y_true, y_pred, text_labels)` to `predictions/train{X}_test{Y}_{clf}.npz`.
   Run once; nothing here is bootstrapped yet.

2. `bootstrap_metrics.py` — reads those small prediction files and, for each one,
   resamples the test set with replacement 2000 times, recomputing accuracy and
   macro precision/recall/F1 each time. Writes one consolidated table,
   `bootstrap_results.csv`, with one row per (classifier, train_pct, test_pct)
   and `_mean`/`_std` columns for each metric (in percent).

Run with the project's venv:

```bash
cd "/Users/anton.gong/Documents/SJTU/PYTHON/Yicheng proj/domain-informed-lightcurve-jepa/bootstrap_eval"
../jepa-env/bin/python get_predictions.py
../jepa-env/bin/python bootstrap_metrics.py
```

`manifest.py` is the single place that maps each (train_pct, test_pct) combo to
its features directory and benchmark output directory — update it there if a
run ever gets redone or relocated, rather than editing paths in the other two
scripts.
