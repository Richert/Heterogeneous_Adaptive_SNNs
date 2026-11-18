#!/usr/bin/env bash
# Two-ensemble (core/shell) SCN model: D0 scan (directed model) + symmetric model at the best D0 + figure.
# Run on the server from the repository root, e.g.
#   (activate the environment first) nohup bash circadian_data/run_coreshell_scan.sh > coreshell_scan.log 2>&1 &
# Estimated wall time with 64 processes (2026-11-18 run: ~17 min per D0 with 12 parameters): ~16 min per D0 value
# (dir, 11 parameters, SCN1-5, 32 starts) x 4 + ~11 min sym ≈ 75 min.
# Inputs (cached on the shared drive): coreshell/scn<i>_coreshell_target.npz, scn<i>_ens*_floor_mixture.npy,
# scn<i>_amplitude_q.npy, targets/scn<i>_target.npz. Outputs go to the same coreshell/ folder.
set -euo pipefail
export HASNN_MPMF="${HASNN_MPMF:-/mnt/kennedy_labdata/richard_turbulence/data/mpmf_simulations}"
PROCS="${PROCS:-64}"
PY="${PY:-python}"                        # the active environment (needs numpy, scipy, matplotlib)
cd "$(dirname "$0")"
"$PY" scn_coreshell_model.py scan --d0s 2.5e-4,5e-4,1e-3,2e-3 --slices 1,2,3,4,5 --starts 32 --procs "$PROCS"
