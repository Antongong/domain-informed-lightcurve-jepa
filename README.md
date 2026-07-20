# Domain-Informed Light-Curve JEPA

Code and experimental artifacts for **Domain-Informed Multi-View Self-Distillation for Astronomical Light-Curve Representation Learning with JEPA**.

The model learns global representations of irregular astronomical light curves using three domain-informed views:

- raw light curve,
- Generalized Lomb-Scargle periodogram,
- phase-folded light curve.

Each view is encoded by a numeric Transformer with continuous RoPE for irregular times and error-aware numeric embeddings for measurement uncertainty. The encoders are trained with multi-view LeJEPA self-distillation. For downstream use, the view embeddings are concatenated.

## Repository Contents

- `model_numeric_only.py`: multi-view numeric Transformer backbone, C-RoPE, error-aware numeric embeddings, pooling, and view heads.
- `losses.py`: LeJEPA, SIGReg, and optional CLIP-style multi-view losses.
- `train_ddp_numeric.py`: distributed pretraining entry point.
- `extract_starembed_embeddings.py`: export frozen embeddings for StarEmbed-style evaluation.
- `run_starembed_benchmarks.py`: wrapper for StarEmbed downstream benchmarks.
- `benchmark_pyrregular.py`, `pretrain_pyrregular.py`, `pyrregular_utils.py`: PYRREGULAR adaptation/evaluation utilities.
- `configs/exp21_lejepa_mean_pooling_no_group.yaml`: main no-overlap EXP21 configuration used for the released checkpoint.
- `results/`: lightweight paper-linked summaries, reports, and figures.

Large checkpoints and embedding matrices are intentionally not tracked by git.

## Released Checkpoint

The EXP21 no-overlap mean-pooling checkpoint is released on Hugging Face:

```text
https://huggingface.co/ruiyicheng/domain-informed-lightcurve-jepa-exp21
```

Expected file:

```text
ckpt_final.pt
```

This checkpoint corresponds to:

```text
runs/EXP21_lejepa_mean_pooling_no_group_no_starembed_overlap/ckpt_final.pt
configs/exp21_lejepa_mean_pooling_no_group.yaml
```

## Installation

```bash
git clone https://github.com/ruiyicheng/domain-informed-lightcurve-jepa.git
cd domain-informed-lightcurve-jepa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments were run in a PyTorch/CUDA environment. StarEmbed and LEAVES data are external and are not redistributed in this repository.

## Pretraining

Edit `configs/exp21_lejepa_mean_pooling_no_group.yaml` so that:

- `data.precomputed.out_dir` points to the precomputed aligned LEAVES directory,
- `data.precomputed.manifest` points to the LEAVES manifest with StarEmbed duplicates removed,
- `training.output_dir` points to your desired run directory.

Then run:

```bash
torchrun --standalone --nproc_per_node=8 \
  train_ddp_numeric.py \
  --config configs/exp21_lejepa_mean_pooling_no_group.yaml
```

The released run used 10,000 optimization steps, batch size 128, three views, mean pooling, LeJEPA with `lambda=0.02`, and no group branch.

## Export StarEmbed Embeddings

```bash
python extract_starembed_embeddings.py \
  --config configs/exp21_lejepa_mean_pooling_no_group.yaml \
  --ckpt ckpt_final.pt \
  --out_dir runs/EXP21_starembed_features \
  --out_info embeddings \
  --repr_mode concat \
  --batch_size 256 \
  --num_workers 8 \
  --num_gpus 1 \
  --save_y_str
```

## Run StarEmbed Benchmarks

```bash
python run_starembed_benchmarks.py \
  --features_dir runs/EXP21_starembed_features \
  --out_dir runs/EXP21_starembed_features/benchmark \
  --mlp_accelerator auto \
  --mlp_devices 1 \
  --mlp_num_workers 8
```

## Key Results

From the paper:

- StarEmbed: the full model outperforms hand-crafted features on 15 of 16 classification metrics.
- Few-shot linear probing macro-F1: `42.56 +/- 7.21` with one sample per class and `63.58 +/- 1.20` with 100 samples per class.
- Photometric zero-point drift detection F1 for the full model: `0.8292`, `0.9660`, `0.9936`, and `0.9964` for drift ranges `(0.05, 0.1)`, `(0.1, 0.3)`, `(0.3, 0.5)`, and `(0.5, 1.0)`.
- PYRREGULAR adaptation: the adapted variant matches or exceeds previous state of the art on 5 of 12 irregular time-series datasets.

Local EXP21 result files are summarized under `results/exp21/`.

## Notes On Data Leakage Control

For the no-overlap LEAVES pretraining run, LEAVES targets cross-matched to StarEmbed sources were removed before pretraining. The paper describes the cross-match procedure using Gaia DR3 source identifiers and a positional match within 2 arcsec.

## Citation

```bibtex
@misc{rui2026domaininformedjepa,
  title  = {Domain-Informed Multi-View Self-Distillation for Astronomical Light-Curve Representation Learning with JEPA},
  author = {Rui, Yicheng},
  year   = {2026}
}
```
