# weight_variance — adaptive coupling weight statistics

Scripts behind the manuscript in `~/OneDrive/manuscripts/PRL_2026/adaptive_coupling_statistics`
(*"On the Non-linear Relationship Between Phase Coherence and Edge Variability in
Adaptively Coupled Phase Oscillator Networks"*, `manuscript_v0.pdf`).

Model: Kuramoto oscillators (Eqs. 1–2) with adaptive coupling
`dA_ij/dt = μ G(θ_j − θ_i) + (1 − A_ij) γ` (Eq. 4), `G ∈ {cos, sin, |sin|}`.
The closed mean field tracks the average weight `Ā` (Eq. 10), the weight variance
`V_A` (Eq. 30) and the weight–drive covariance `C_A` (Eq. 31).

PRL figure style comes from [`../shared/prl_style.py`](../shared/prl_style.py);
output locations from [`../shared/data_paths.py`](../shared/data_paths.py) (override
the root with `$HASNN_DATA`). Everything here runs in the `pycobi` conda env.

## Figure pipelines

Run each row left to right.

| Manuscript figure | Data generation | Figure assembly |
|---|---|---|
| `weight_variance.svg` — `V_A/Ā²` vs. coherence `R` and vs. mean-field error | `weight_variance_coherence_sweep.py` + `weight_variance_coherence_meanfield.py` | `weight_variance_coherence_figure.py` |
| `weight_variance_rmse.svg` — RMSE of `R`, `Ā`, `V_A` vs. `Δ` | `weight_variance_rule_micro_oainit.py` | `weight_variance_VA_sweep_figure.py` |
| `cosine.svg` / `sine.svg` — `Δ`-bifurcation diagrams per rule | `weight_variance_bifurcation_micro.py` | `weight_variance_bifurcation_rules.py` |
| `weight_variance_connectivity.svg` — coupling-matrix structure vs. `Δ` | `weight_variance_connectivity_sweep.py` | `weight_variance_connectivity_figure.py` |
| `figure_a2.svg` — `K`-ramp bifurcation + hysteresis | `weight_variance_ramp_micro.py` + `weight_variance_ramp_meanfield.py` | `weight_variance_ramp_figure.py` |

## Core module

`weight_variance_analysis.py` holds the closed-form theory that everything else
imports — `branches()` (the `R`, `Ā`, `V_A` fixed-point branches), `S_order()` (the
tabulated on-manifold `S = ⟨|c|²⟩`), `sync_delta_end()`. Run standalone it produces
the analytic bifurcation and weight-statistics figures.

`weight_variance_rule_micro.py` holds the numba microscopic kernel
(`lorentzian_quantiles`, the `N×N` adaptive integrator) reused by the `*_micro*`
and `*_sweep` scripts. `weight_variance_meanfield.py` holds `order_parameter_S` and
the time-domain mean-field-vs-micro comparison.

## Secondary analyses

- `weight_variance_bifurcation.py` / `weight_variance_ramp_bifurcation.py` —
  analytic `K`-bifurcation diagrams (Section D fixed points).
- `weight_variance_rule_meanfield.py` — `V_A/Ā²` vs. `Δ` and vs. `R`, full vs. static closure.
- `weight_variance_VA_sweep.py` — `V_A(t)` micro-vs-mean-field over `(K, μ)`.
- `kmo_adaptive_single_sweep.py` → `kmo_adaptive_single_figure.py` /
  `kmo_adaptive_variance_figure.py` — single population vs. one Ott–Antonsen
  ensemble over `Δ × μ`, and the resulting weight variance.
- `kmo_lmmf_variance_bound_sweep.py` → `kmo_lmmf_variance_bound_figure.py` —
  LMMF accuracy as a function of the `V_A/Ā²` budget (inverts Eq. 37 to get `Δ_max`,
  hence `M*`). This is the bridge to the [LMMF](../LMMF) manuscript.
- `kmo_adaptive_rules_figure.py` — coupling structure and coherence across all three rules.
- `kmo_heterogeneity_analysis.py` — bifurcation structure of `R` from Eqs. 8 & 10.

## `exploratory/`

- `weight_variance_mixed_locking.py` — variance closures with a static (DC) drift term.
- `weight_variance_weak_coupling.py` — validates the variance–covariance closure in
  isolation (phases free-running, no feedback of `A` into `θ`), the regime it is derived for.
