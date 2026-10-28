#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
export PYTHONPATH=src CU_CACHE_DIR="${CU_CACHE_DIR:-cache}"
run() { echo "== $*"; "$PY" -u -m "$@"; }
have() { [ -e "$1" ]; }
T1="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CU_THREADS=1"

"$PY" scripts/download_data.py
mkdir -p results/tests
"$PY" -m pytest -q tests 2>&1 | tee results/tests/pytest_full_final.log | tail -3
ls results/m2/*/summary.json >/dev/null 2>&1 || run cu.eval.experiments.m2_baselines --config configs/synth_base.yaml --seed 0
for s in 0 1 2 3 4; do
  have results/m3/seed$s/summary.json || run cu.eval.experiments.m3_variant_a --config configs/synth_base.yaml --seed $s
done
run cu.eval.experiments.m3_report
have results/m3/diag/.done || { run cu.eval.experiments.m3_diagnose --seed 0 && touch results/m3/diag/.done; }
for s in 0 1 2 3 4; do
  have results/m3r/seed$s/summary.json || run cu.eval.experiments.m3r_revision --seed $s
  have results/m3r/seed$s/rhoS_stats.json || run cu.eval.experiments.m3r_rhos_stats --seed $s
  have results/m3r/seed$s/.round2_done || { run cu.eval.experiments.m3r_round2 --seed $s && touch results/m3r/seed$s/.round2_done; }
  have results/m3r/seed$s/.rhos_shared_done || { run cu.eval.experiments.m3r_rhos_shared --seed $s && touch results/m3r/seed$s/.rhos_shared_done; }
done
run cu.eval.experiments.m3r_report
for s in 0 1 2 3 4; do
  have results/m4/seed$s/mondeq.pt || run cu.eval.experiments.m4_variant_b --seed $s --steps 3000
  run cu.eval.experiments.m4_variant_b --seed $s --m 0.5 --L-max 1.5 --reuse-from results/m4/seed$s --rel-tau --out results/m4r/m4
done
have results/m4_m03/seed0/mondeq.pt || run cu.eval.experiments.m4_variant_b --seed 0 --steps 10000 --m 0.3 --L-max 2.0 --out results/m4_m03
have results/m4_m02/seed0/mondeq.pt || run cu.eval.experiments.m4_variant_b --seed 0 --steps 10000 --m 0.2 --L-max 3.0 --out results/m4_m02
run cu.eval.experiments.m4_variant_b --seed 0 --m 0.3 --L-max 2.0 --reuse-from results/m4_m03/seed0 --rel-tau --out results/m4r/m4_m03
run cu.eval.experiments.m4_variant_b --seed 0 --m 0.2 --L-max 3.0 --reuse-from results/m4_m02/seed0 --rel-tau --out results/m4r/m4_m02
for t in m4 m4_m03 m4_m02; do run cu.eval.experiments.m4_report --root results/m4r/$t; done
run cu.eval.experiments.m5r_e6 --datasets ecg_compressive cwru_or007 cwru_ir007 --seeds 0 1 2
run cu.eval.experiments.m5r_e6 --datasets ecg_denoise_overcomplete --seeds 0 --heads lista_cp
run cu.eval.experiments.m5r_e7 --seeds 0 1 2 3 4
run cu.eval.experiments.m5r_report
"$PY" scripts/replot_fig8.py
run cu.eval.experiments.m7_bearing_heads --files cwru_or007 cwru_ir007 --seeds 0 1 2
run cu.eval.experiments.m8_frontier --datasets synthetic ecg_compressive
run cu.eval.experiments.m9_numerics --parts spectral float32 float64 gap reference margins
run cu.eval.experiments.m10_head_stats
have results/m11/seed2/summary.json || run cu.eval.experiments.m11_operating_point --seeds 0 1 2 --steps 1600
have results/m11_400/seed2/summary.json || run cu.eval.experiments.m11_operating_point --seeds 0 1 2 --steps 400 --out results/m11_400
env $T1 "$PY" -u -m cu.eval.experiments.m6_endtoend --datasets synthetic --seeds 0 1 2 3 4 --repeats 5
env $T1 "$PY" -u -m cu.eval.experiments.m6_endtoend --datasets synthetic_acc --seeds 0 1 2 --repeats 5
env $T1 "$PY" -u -m cu.eval.experiments.m6_endtoend --datasets ecg_compressive --seeds 0 1 2 --repeats 5
env $T1 "$PY" -u -m cu.eval.experiments.m6_endtoend --datasets ecg_denoise_overcomplete --seeds 0 --repeats 3
env $T1 "$PY" -u -m cu.eval.experiments.m6_endtoend --datasets cwru_or007 cwru_ir007 --seeds 0 1 2 --repeats 3
"$PY" scripts/fig_endtoend.py
