# LMMF — Lorentzian-mixture mean field

Scripts behind the manuscript in `~/OneDrive/manuscripts/PRL_2026/LMMF`
(*"A multi-ensemble mean-field reduction method for networks of globally coupled
phase oscillators with arbitrary parameter distributions"*, `manuscript_v0.pdf`).

The shared LMMF fitter itself lives in [`../shared/lorentzian_mixture.py`](../shared/lorentzian_mixture.py);
the PRL figure style in [`../shared/prl_style.py`](../shared/prl_style.py); output
locations in [`../shared/data_paths.py`](../shared/data_paths.py) (override the root
with `$HASNN_DATA`).

## Figure pipelines

Run each column left to right. Unless noted, use the `pycobi` conda env; the Allen
scripts (`allen_*`, `pv_*_rate`, `pyramidal_fig_rate`) need `allen` (allensdk, numba).

| Manuscript figure | Data generation | Figure assembly |
|---|---|---|
| **Fig. 1** — fit quality vs. `M_max`, `λ` | `kmo_lorentzian_fit_sweep.py` | `kmo_lorentzian_fit_figure.py` |
| **Fig. 2** — Skardal rational-distribution benchmark | `skardal_benchmark_sweep.py` → `skardal_benchmark_lmmf.py` | `skardal_benchmark_sweep_figure.py` |
| **Fig. 3** — global heterogeneity knobs | `kmo_heterogeneity_sim.py fit` → `kmo_heterogeneity_bifurcation.py {lumped,twoknob}` → `kmo_heterogeneity_sim.py` | `kmo_heterogeneity_figure.py` |
| **Fig. A1** — PV+ interneurons | `allen_lorentzian_fit.py` → `pv_figure2_bifurcation.py` + `pv_figure2_rate.py` | `pv_figure2.py` |
| **Fig. A2** — pyramidal cells | `allen_lorentzian_fit.py` → `pyramidal_fig_bifurcation.py` + `pyramidal_fig_rate.py` | `pyramidal_fig.py` |
| **Fig. S1** — seed-to-seed finite-size fidelity | `skardal_benchmark_simulate.py` | `skardal_benchmark_supplement.py` |

The Allen scripts take `"<cell class>" "<layer>"` as CLI arguments, e.g.
`python pv_figure2_bifurcation.py "PV+ Interneuron" "L5/6"`, and tag their output
`<name>_<pv|pyramidal>_<L23|L56>.npz`.

## Shared building blocks in this directory

These are imported by the scripts above rather than run on their own:

- `allen_qif_bifurcation.py` — `build_circuit` / `build_equations` (with the `hD`/`hC`
  heterogeneity knobs), `load_fit`, `_tag`. Reused by every Allen bifurcation script.
  Also runnable standalone for the plain `I`/`J` continuation.
- `allen_qif_meanfield.py` — `build_micro` / `build_mf`, `_state_slice`, `_param`.
  Reused by `pv_figure2_rate.py` and `pyramidal_fig_rate.py`. Standalone it compares
  the spiking QIF net against the threshold-heterogeneity mean field.
- `kmo_lorentzian_fit_sweep.py` — `simulate_micro` / `simulate_ensemble` (PyRates +
  `solve_ivp`). Reused by all the Skardal scripts and by `kmo_heterogeneity_sim.py`.
- `skardal_benchmark_simulate.py` — `gn_density`, `sample_gn`, `simulate_skardal`.

## Supporting / secondary

- `allen_spike_thresholds.py` — plots the raw Allen excitability distributions
  (exploration that motivated the fits). Downloads into `cell_types/`.
- `gaussian_mixture_fit.py` — demo of the Lorentzian-mixture fitter on Gaussian-mixture samples.
- `skardal_benchmark_figure.py` — earlier single-exponent version of Fig. 2.
- `skardal_finite_size_check.py` — validation of *why* the LMMF beats the Skardal
  reduction at large `n`.
- `pv_figure2_loci_fig.py` — Hopf loci supplement for the PV+ regime.
- `pv_heterogeneity_principle.py` — schematic of the heterogeneity re-parameterization.

## `exploratory/`

Precursors and dead ends of the same pipelines, kept for reference: the earlier
Allen summary figure and heterogeneity scans (`allen_qif_figure.py`,
`allen_qif_heterogeneity*.py`, `allen_qif_hopf_scan.py`, `pv_heta_check.py`,
`tmp_pyramidal_ih.py`), the structured-coupling extension
(`kmo_random_coupling_*.py`, `qif_structured_coupling_lmmf.py`,
`qif_conductance_meanfield.py`) and the plastic E/I network (`allen_ei_stdp_lmmf.py`).
