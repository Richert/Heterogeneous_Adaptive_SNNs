# Heterogeneous_Adaptive_SNNs

Simulations and numerical analysis of heterogeneous oscillator and spiking neural
networks with synaptic plasticity.

## Layout

Manuscript pipelines are grouped per paper; each has a `README.md` mapping scripts to
manuscript figures.

| Directory | Contents |
|---|---|
| [`shared/`](shared/) | Manuscript-agnostic library: the Lorentzian-mixture fitter, the PRL figure style, output paths |
| [`LMMF/`](LMMF/) | PRL_2026 **LMMF** paper — multi-ensemble mean-field reduction for phase oscillators |
| [`weight_variance/`](weight_variance/) | PRL_2026 **adaptive_coupling_statistics** paper — coupling-weight statistics of adaptive networks |
| `kuramoto/`, `theory/`, `bifurcation_analysis/`, `grid_search/`, `qif_simulations/` | Earlier / ongoing work not tied to those two papers, including the PRL_2026 `stdp` draft (`kuramoto/kuramoto_ensemble_fitting.py` and the adaptive-ensemble bifurcation scripts) |
| `config/` | PyRates YAML equation templates, Auto-07p constants, Fortran right-hand sides |
| `rnn/`, `waves/`, `sender_receiver/`, `oja_weight_distributions/`, `striatal_plasticity/`, `memory_formation/`, `reservoir_computing/`, `fre_training/` | Older project directories, untouched |

## `shared/`

- `lorentzian_mixture.py` — fit an arbitrary 1-D parameter distribution by a weighted
  Lorentzian mixture (greedy penalized Cramér–von Mises order selection). Imported as
  `LM` throughout.
- `prl_style.py` — `set_prl_style(preset, **overrides)` and `panel_label(...)`. Presets:
  `"prl"` (canonical single/two-column), `"prl_wide"`, `"diagnostic"`.
- `data_paths.py` — output directories. `mpmf_simulations` lives on the shared lab drive,
  which is mounted differently per machine: the first existing entry of
  `SHARED_MPMF_CANDIDATES` is used (workstation `/mnt/kennedy_labdata/...`, compute server
  `/media/storage/DATA/...`; add a line for a new machine). The other directories live
  under the data root, which defaults to `~/data`. Both are overridable:

      HASNN_MPMF=/some/dir python LMMF/kmo_lorentzian_fit_sweep.py          # mpmf_simulations only
      HASNN_DATA=/scratch/$USER/data python weight_variance/weight_variance_rule_micro.py

Scripts in `LMMF/` and `weight_variance/` reach these through a small bootstrap block
at the top of each file, which puts the script's own directory and `shared/` on
`sys.path`. Run scripts from their own directory (or anywhere — the bootstrap resolves
paths relative to the file, not the cwd).

## Conda environments

| Env | Used for |
|---|---|
| `pycobi` | PyCoBi/Auto-07p continuations, PyRates + `solve_ivp`, the Kuramoto sweeps |
| `allen` | anything touching `allensdk`, `numba` or `seaborn` (the Allen QIF scripts) |

Never run two PyCoBi/Auto sessions in parallel — concurrent Auto sessions produce
spurious `MX` failures.
