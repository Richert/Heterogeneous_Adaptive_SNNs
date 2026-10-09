# LMMF resubmission — open TODOs (after the PRL rejection, report lu21625)

Status as of 2026-10-08. Background: `continuity/continuity_notes.md`; supplement draft
`~/OneDrive/manuscripts/PRL_2026/LMMF/supplement_continuity.tex`.

## 1. New fitting algorithm (shared/lorentzian_mixture.py)

* **Warm start** (`warm_start=True`, default): each fixed-M fit is also initialised from the
  previous (pruned) fit, grown at the largest CDF residual. The loss is now monotone in M and
  fits no longer collapse to fewer effective components.
* **Noise-floor cap** (`floor_c=1.0`, opt-in): the greedy search stops once
  D(M) ≤ 1/(6N). M* is the argmin over the visited M of D(M) + λ·M. This replaces the α
  goodness-of-fit acceptance.
* **λ is absolute (N-independent).** Do NOT express λ in units of 1/(6N); then M* grows with N.

Fig. 1 already uses all three (`kmo_lorentzian_fit_sweep.py`, `kmo_lorentzian_M_stability.py`).
Its network also starts from wrapped-Cauchy phases (`ic="wrapped_cauchy"`).

## 2. Regenerate all other manuscript figures with the new algorithm

Each pipeline below still calls `LM.fit` with the old α acceptance. Its next run will pick up the
warm start automatically, but not the noise-floor cap. For each one: add `floor_c=1.0`, keep or
re-tune the absolute λ, then regenerate.

| Figure | Fit call / settings to update | Then rerun |
|---|---|---|
| Fig. 2 (Skardal benchmark) | **done 2026-10-09**: new two-column subcritical figure comparing Skardal, LMMF with noise floor, and LMMF without it (`variant=nofloor`). Network starts from wrapped-Cauchy phases. | `skardal_benchmark_sweep.py` → `skardal_benchmark_lmmf.py` (+ `variant=nofloor regime=subcritical`) → `skardal_benchmark_twocol_figure.py` |
| Fig. 3 (heterogeneity) | `kmo_heterogeneity_sim.py` fit config (α=1e-2, λ=5e-5, M_max=8) | `kmo_heterogeneity_sim.py fit` → `kmo_heterogeneity_bifurcation.py {lumped,twoknob}` → `kmo_heterogeneity_sim.py` → `kmo_heterogeneity_figure.py` |
| Fig. A1 (PV+) | `allen_lorentzian_fit.py` (ALPHA=1e-3, LAMBDA_M=1e-5) | `allen_lorentzian_fit.py` (both layers) → `pv_figure2_bifurcation.py` + `pv_figure2_rate.py` → `pv_figure2.py` |
| Fig. A2 (pyramidal) | `allen_lorentzian_fit.py` (same) | → `pyramidal_fig_bifurcation.py` + `pyramidal_fig_rate.py` → `pyramidal_fig.py` |
| Fig. S1 (finite-size fidelity) | `skardal_benchmark_figure.py` constants (ALPHA, LAMBDA_M) used by `skardal_benchmark_supplement.py` | `skardal_benchmark_simulate.py` → `skardal_benchmark_supplement.py` |

Notes:
* Allen fits have small N (46–488 cells). The noise floor 1/(6N) is large there, so expect smaller M*.
* If M* changes in the Allen or heterogeneity fits, bifurcation diagrams move. Re-check the
  Hopf/fold loci and the PyCoBi seeds (see memory notes on PyCoBi continuation rules).
* For network simulations, switch the initial phases to wrapped Cauchy where applicable (Remark S4).

## 3. Update figure captions and manuscript text

* **Fig. 1 caption.** Describe the new panel (b): mean M* over 30 resampled data sets vs λ and N,
  with cell text mean and range. Relabel the examples (b–d) → (c–e). Panel (a) is transposed.
* **Main text, Fig. 1 paragraph.** Replace "M ≈ 6–8 across random realizations" with the
  stability result: M* ≈ 6 (λ = 1e-5), ≈ 3 (λ = 1e-4), 2 (λ = 1e-3), saturating in N with
  spread ≤ ±1; the noise floor caps M at small N.
* **Appendix B.** Replace the GoF/α stopping with: warm start; noise-floor cap D ≤ 1/(6N) with
  its derivation, E[D] = 1/(6N) for the true CDF; M* = argmin D + λM over the visited M.
* **Continuity.** Add the main-text pointer to the supplement (Prop. S1, Cor. S1–S2) and cite
  Lancellotti 2005 and Chiba–Medvedev 2019.
* **Fig. 2, later (user request 2026-10-09):** add an LMMF fitted to the POPULATION density g_n
  (quantile nodes, no sampling noise) as a third reference. This separates fit error from sampling error.
* **Fig. 2 caption and text.** The figure is now two-column, subcritical only, with examples at N=1000.
  Without the floor, the LMMF beats Skardal at every N (mean spectral RMSE ×10⁻³: 2.4/1.3/1.3 vs
  8.3/3.6/1.5 at N=200/1000/5000). With the floor it wins only at N ≤ 1000. The old text claimed the
  LMMF is "consistently" better; that was never true in the supercritical regime, even in the old run.
* **Fig. 3 caption (error in the submitted version).** The shaded regions in (e,f) are NOT network
  results. `kmo_heterogeneity_sim.py run_regions()` classifies them from LMMF (ensemble OA)
  simulations started from two seeds. Either fix the caption ("…where the LMMF equations converged
  to…") or recompute the shading from network simulations, which is the stronger option.
* **Skardal paragraph.** Reframe per referee point 3: both are OA reductions of rational
  densities. Add the fair comparisons from `continuity/continuity_notes.md` §5.
* **All other captions and text** that quote M*, λ, α, fit p-values or numbers from the figures in
  §2: update after regeneration.
* **Supplement Table S1 / Sec. S7.** These use `continuity/continuity_check.py` (standalone
  integrator, fixed-M fits). Either say so or regenerate with the final pipeline.
