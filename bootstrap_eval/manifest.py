"""Just configs cuh"""
import os

REPO_ROOT = "/Users/anton.gong/Documents/SJTU/PYTHON/Yicheng proj/domain-informed-lightcurve-jepa"
RUNS_DIR = os.path.join(REPO_ROOT, "runs")

# percent-truncation -> features directory holding handcrafted_features_{train,validation,test,anom}.npz
PCT_TO_FEATURES_DIR = {
    100: os.path.join(RUNS_DIR, "handcrafted_features"),
    50:  os.path.join(RUNS_DIR, "handcrafted_features_pct50_fixed"),
    25:  os.path.join(RUNS_DIR, "handcrafted_features_pct25_fixed"),
}

# (train_pct, test_pct) -> benchmark output dir containing x/rf, x/logistic_knn, x/mlp
COMBO_OUT_BASE = {
    (100, 100): os.path.join(RUNS_DIR, "handcrafted_features", "benchmark", "x"),
    (100, 50):  os.path.join(RUNS_DIR, "handcrafted_features_train100_test50_fixed", "benchmark", "x"),
    (100, 25):  os.path.join(RUNS_DIR, "handcrafted_features_train100_test25_fixed", "benchmark", "x"),
    (50, 100):  os.path.join(RUNS_DIR, "handcrafted_features_train50_test100_fixed", "benchmark", "x"),
    (50, 50):   os.path.join(RUNS_DIR, "handcrafted_features_pct50_fixed", "benchmark", "x"),
    (50, 25):   os.path.join(RUNS_DIR, "handcrafted_features_train50_test25_fixed", "benchmark", "x"),
    (25, 100):  os.path.join(RUNS_DIR, "handcrafted_features_train25_test100_fixed", "benchmark", "x"),
    (25, 50):   os.path.join(RUNS_DIR, "handcrafted_features_train25_test50_fixed", "benchmark", "x"),
    (25, 25):   os.path.join(RUNS_DIR, "handcrafted_features_pct25_fixed", "benchmark", "x"),
}

TRAIN_LEVELS = [100, 50, 25]
TEST_LEVELS = [100, 50, 25]
COMBOS = [(tr, te) for tr in TRAIN_LEVELS for te in TEST_LEVELS]

CLASSIFIERS = ["logistic", "knn", "rf", "mlp"]

SEED = 42
SCENARIO = "concat"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(THIS_DIR, "predictions")
RESULTS_CSV = os.path.join(THIS_DIR, "bootstrap_results.csv")

MY_STAR_EMBED_DIR = "/Users/anton.gong/Documents/SJTU/PYTHON/Yicheng proj/my_star_embed"
