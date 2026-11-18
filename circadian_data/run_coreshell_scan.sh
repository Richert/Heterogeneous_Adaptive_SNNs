#!/usr/bin/env bash
# Two-ensemble (core/shell) SCN model, directed model MODEL (default dirg) only: one fit per D0 value (+ figure).
# Run on the server from the repository root (activate the Python environment first), e.g.
#   HASNN_MPMF=/media/storage/DATA/richard_turbulence/data/mpmf_simulations PROCS=64 \
#     nohup bash circadian_data/run_coreshell_scan.sh > coreshell_scan.log 2>&1 &
# D0 values: D0S="2.5e-4 5e-4" (default), MODEL=dirg (default; dir = per-ensemble width factors, shared K_g).
# Estimated wall time with 64 processes: ~15 min per D0 (10 parameters, SCN1-5, 32 starts).
# Inputs (cached on the shared drive): coreshell/scn<i>_coreshell_target.npz, scn<i>_ens*_floor_mixture.npy,
# scn<i>_amplitude_q.npy, targets/scn<i>_target.npz. Outputs: coreshell/scn<i>_<MODEL>_d<D0>_fit.npy,
# scn_coreshell_fits_d<D0>.png, coreshell_summary_d<D0>.json.
set -euo pipefail
export HASNN_MPMF="${HASNN_MPMF:-/mnt/kennedy_labdata/richard_turbulence/data/mpmf_simulations}"
PROCS="${PROCS:-64}"
PY="${PY:-python}"                        # the active environment (needs numpy, scipy, matplotlib)
D0S="${D0S:-2.5e-4 5e-4}"
MODEL="${MODEL:-dirg}"
cd "$(dirname "$0")"
for d0 in $D0S; do
    "$PY" scn_coreshell_model.py fit figure --models "$MODEL" --d0 "$d0" --slices 1,2,3,4,5 --starts 32 --procs "$PROCS"
done
