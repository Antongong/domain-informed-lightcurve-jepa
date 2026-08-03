#!/usr/bin/env python3

import os
import sys
import glob
import json
import pickle

import numpy as np
import torch

import manifest as M

sys.path.insert(0, M.MY_STAR_EMBED_DIR)
from label_utils import build_label_remap, labels_from_npz, remap_y  # noqa: E402
from mlp import LitMLP  # noqa: E402

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def load_split(features_dir, split, scenario=M.SCENARIO):
    """take the data from the features file and clean it from
    infinite values and return x and y in matrices"""
    path = os.path.join(features_dir, f"handcrafted_features_{split}.npz")
    with np.load(path, allow_pickle=True) as z:
        key = "x" if scenario == "concat" else "x_avg"
        X = np.asarray(z[key], dtype=np.float64)
        y = labels_from_npz(z)
    finite = np.isfinite(X).all(axis=1)
    return X[finite], y[finite]


def load_combo_data(train_pct, test_pct):
    """Load the data from trainin, validation etc, do remapping
    from label name to number, concatinate training and validation."""
    train_dir = M.PCT_TO_FEATURES_DIR[train_pct]
    test_dir = M.PCT_TO_FEATURES_DIR[test_pct]

    X_train, y_train = load_split(train_dir, "train")
    X_val, y_val = load_split(train_dir, "validation")
    X_test, y_test = load_split(test_dir, "test")

    label_to_idx, text_labels = build_label_remap(np.concatenate([y_train, y_val], axis=0))
    y_train_i = remap_y(y_train, label_to_idx)
    y_val_i = remap_y(y_val, label_to_idx)
    y_test_i = remap_y(y_test, label_to_idx)

    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train_i, y_val_i], axis=0)

    return X_train_full, y_train_full, X_test, y_test_i, text_labels

"""Following taken from my_star_embed with some modifications 
from saved checkpoints """

def predict_logistic(X_train, y_train, X_test):
    """Standardize data, define logistic regression then train it
    and then predict """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=M.SEED, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    return clf.predict(X_test_s)


def predict_knn(X_train, y_train, X_test, k=5):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train_s, y_train)
    return clf.predict(X_test_s)


RF_SEEDS = [42, 100, 158 + 42]  # matches rf.py: seed, seed+58, seed+158 with seed=42 -> [42, 100, 200]
# RF_SEEDS = [42] #use if only one seed

def predict_rf(X_train, y_train, X_test, out_base):
    """find file with best params, extract best params, then fit and predict
    once per original seed (42, 100, 200) -- rf.py's own numbers are the MEAN
    across these three seeds, so we reproduce all three rather than just one."""
    pattern = os.path.join(out_base, "rf", "*", "best_params.json")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No best_params.json found under {pattern}")
    with open(matches[0]) as f:
        info = json.load(f)
    best_params = info["best_params"]

    preds_by_seed = {}
    for seed in RF_SEEDS: #If only run 42 no loop
        clf = RandomForestClassifier(random_state=seed, n_jobs=-1, **best_params)
        clf.fit(X_train, y_train)
        preds_by_seed[seed] = clf.predict(X_test)
    return preds_by_seed 


def predict_mlp(X_test, out_base):
    """finds the scaler used and scales data x, finds best run
    finds checkpoint,  load checkpoint litMLP, return prediction
    when ran through"""
    mlp_dir = os.path.join(out_base, "mlp")

    scaler_path = os.path.join(mlp_dir, f"standard_scaler_seed{M.SEED}_{M.SCENARIO}.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No scaler found at {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_test_s = scaler.transform(X_test)

    best_run_path = os.path.join(mlp_dir, "best_run.txt")
    with open(best_run_path) as f:
        lines = f.readlines()
    log_dir = lines[1].strip().split("=", 1)[1]

    ckpt_matches = glob.glob(os.path.join(log_dir, "checkpoints", "*.ckpt"))
    if not ckpt_matches:
        raise FileNotFoundError(f"No checkpoint found under {log_dir}")
    ckpt_path = ckpt_matches[0]

    model = LitMLP.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_s).float())
        preds = logits.argmax(1).numpy()
    return preds


def main():
    """Check for prediction repository, prepare the data,
    does everything and saves 36 files, 4 classification heads x
    3 train x 3 test"""
    os.makedirs(M.PRED_DIR, exist_ok=True)

    for train_pct, test_pct in M.COMBOS:
        print(f"\n=== train{train_pct}/test{test_pct} ===")
        out_base = M.COMBO_OUT_BASE[(train_pct, test_pct)]

        X_train, y_train, X_test, y_test, text_labels = load_combo_data(train_pct, test_pct)

        single_preds = {
            "logistic": predict_logistic(X_train, y_train, X_test),
            "knn": predict_knn(X_train, y_train, X_test),
            "mlp": predict_mlp(X_test, out_base),
        }

        for clf_name, y_pred in single_preds.items():
            out_path = os.path.join(M.PRED_DIR, f"train{train_pct}_test{test_pct}_{clf_name}.npz")
            np.savez_compressed(
                out_path,
                y_true=y_test,
                y_pred=y_pred,
                text_labels=np.array(text_labels, dtype=object),
            )
            acc = float((y_pred == y_test).mean())
            print(f"  {clf_name:10s} -> {os.path.basename(out_path)}  (quick acc check: {acc*100:.1f}%)")

        # rf: save one prediction array per seed, since the original numbers are a 3-seed mean
        rf_preds_by_seed = predict_rf(X_train, y_train, X_test, out_base)
        out_path = os.path.join(M.PRED_DIR, f"train{train_pct}_test{test_pct}_rf.npz")
        save_kwargs = {"y_true": y_test, "text_labels": np.array(text_labels, dtype=object)}
        for seed, y_pred in rf_preds_by_seed.items():
            save_kwargs[f"y_pred_seed{seed}"] = y_pred
        np.savez_compressed(out_path, **save_kwargs)
        accs = [float((y_pred == y_test).mean()) for y_pred in rf_preds_by_seed.values()]
        print(f"  {'rf':10s} -> {os.path.basename(out_path)}  (quick acc check per seed: {[f'{a*100:.1f}%' for a in accs]})")


if __name__ == "__main__":
    main()
