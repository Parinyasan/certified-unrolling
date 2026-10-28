# Certified adaptive-depth unrolling for sparse inverse problems

Code for the article *Certified adaptive-depth unrolling for sparse inverse problems* by Parinya Sanguansat and
Supatta Viriyavisuthisakul (submitted to *Neurocomputing*).

A learned unrolled network (the *head*, LISTA-type) is used only as a warm start for a certified fixed-point *tail*,
the proximal-gradient iteration of an elastic-net problem. The tail is a contraction whose constant follows from two
singular values of the operator, so the Banach bound gives a per-input certificate and an observable stopping rule:
every returned point is within a prescribed relative distance of the elastic-net solution, whatever the head
produced. Duality-gap certificates, an accelerated (FISTA) tail and a learned monotone-operator tail are included.

This repository contains the code that produces every result of the article. The trained heads of seed 0 are
attached to the release as `certified-unrolling-heads-seed0.zip`.

## Requirements

* CPU only (no GPU needed). Tested on Windows 11 with Python 3.13; Linux and macOS should work.
* About 1 GB of disk for data and results. A full reproduction takes about 24 hours on a 10-core desktop CPU.

## Installation

With conda:

```bash
git clone https://github.com/Parinyasan/certified-unrolling.git
cd certified-unrolling
conda env create -f environment.yml
conda activate certified-unrolling
```

or with pip in an existing Python 3.10+ environment:

```bash
pip install -e ".[dev,ecg]"
```

All commands below are run from the repository root with

```bash
export PYTHONPATH=src CU_CACHE_DIR=cache          # Windows PowerShell: $env:PYTHONPATH="src"; $env:CU_CACHE_DIR="cache"
```

## Data

The synthetic data are generated from the seeds. The real data are public and are downloaded once:

```bash
python scripts/download_data.py
```

This fetches the MIT-BIH Arrhythmia and MIT-BIH Noise Stress Test databases from PhysioNet and two files of the Case
Western Reserve University Bearing Data Center, stores them under `cache/data/`, and checks them against the SHA-256
checksums in `configs/data_checksums.json`. If a download fails, run the script again; it fetches only what is
missing. The data are not redistributed here; please cite their sources when you use them.

## Check the installation

```bash
python -m pytest tests          # 30 tests, about 3 minutes
```

## Reproduce the results

Everything, in dependency order:

```bash
bash scripts/reproduce.sh       # set PYTHON=/path/to/python to choose the interpreter
```

Training steps whose output already exists are skipped, so the script can be restarted after an interruption. The
steps can also be run one by one (`python -m cu.eval.experiments.<module> ...`):

| step | module(s) | output | time |
|---|---|---|---|
| fixed-depth baselines | `m2_baselines --config configs/synth_base.yaml --seed 0` | `results/m2/` | 0.5 h |
| synthetic heads (seeds 0-4) | `m3_variant_a --config configs/synth_base.yaml --seed s`, then `m3_report` | `results/m3/` | 5 x 25 min |
| ideal warm starts, alternative heads | `m3_diagnose --seed 0` | `results/m3/diag/` | 20 min |
| certified tails and certificates (seeds 0-4) | `m3r_revision`, `m3r_rhos_stats`, `m3r_round2`, `m3r_rhos_shared` with `--seed s`, then `m3r_report` | `results/m3r/` | 5 x 45 min |
| learned monotone-operator tail | `m4_variant_b` (see `scripts/reproduce.sh` for the settings), `m4_report` | `results/m4*/` | 3 h |
| real signals (ECG, bearings) | `m5r_e6`, then `m5r_e7` (operator shift), `m5r_report`, `python scripts/replot_fig8.py` | `results/m5r/` | 4 h |
| bearing heads (dense vs. shift-invariant) | `m7_bearing_heads --files cwru_or007 cwru_ir007 --seeds 0 1 2` | `results/m7/` | 3 h |
| accuracy over (lambda, eps) | `m8_frontier --datasets synthetic ecg_compressive` | `results/m8/` | 1 h |
| floating-point bounds | `m9_numerics --parts spectral float32 float64 gap reference margins` | `results/m9/` | 30 min |
| head statistics | `m10_head_stats` | `results/m10/` | 30 min |
| heads at a validated operating point | `m11_operating_point --seeds 0 1 2 --steps 1600` (and `--steps 400 --out results/m11_400`) | `results/m11*/` | 1.5 h |
| end-to-end cost (time) | `m6_endtoend --datasets ...` (one thread, see below) | `results/m6/` | 1.7 h |
| end-to-end figure | `python scripts/fig_endtoend.py` | `results/figs/` | 1 s |

Every module writes JSON files with all the numbers it computes; the figures are written next to them.

### Timing

The end-to-end times are measured on one thread:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CU_THREADS=1 \
  python -m cu.eval.experiments.m6_endtoend --datasets synthetic --seeds 0 1 2 3 4 --repeats 5
```

Times depend on the machine and its load; run them with nothing else on the machine. Iteration counts and outputs
do not depend on the timing; the module checks every output against the reference solution.

## Using the pre-trained heads

Download `certified-unrolling-heads-seed0.zip` from the release and unpack it in the repository root; the heads
land under `results/`. They can then be evaluated without training, for example:

```bash
OMP_NUM_THREADS=1 python -m cu.eval.experiments.m6_endtoend --datasets synthetic --seeds 0 --repeats 1
```

The dense LISTA-CP heads of the bearing data (38 MB each) are not included; the scripts retrain them.

## Layout

```
src/cu/algos/        certified tails, certificates, accelerated tail, floating-point bounds
src/cu/models/       unrolled heads (LISTA-CP, ALISTA, ConvLISTA) and the monotone-operator tail
src/cu/ops/          linear operators (dense, convolution, subsampling)
src/cu/data/         synthetic, ECG and bearing data, downloader
src/cu/train/        head training
src/cu/eval/         metrics and the experiment modules (experiments/m*.py)
configs/             problem configurations and data checksums
scripts/             data download, full reproduction, figures
tests/               test suite
```

## Licence

Code: MIT (see `LICENSE`). The trained heads in the release asset: CC BY 4.0.

## Citation

If you use this code, please cite the article (the journal reference will be added on publication):

> P. Sanguansat and S. Viriyavisuthisakul, Certified adaptive-depth unrolling for sparse inverse problems,
> submitted to Neurocomputing.
