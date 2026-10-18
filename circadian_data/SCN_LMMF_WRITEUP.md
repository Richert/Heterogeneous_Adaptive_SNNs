# Mean-field model of SCN desynchronisation and resynchronisation under TTX

*Status: 2026-10-17. Slices SCN1–4 of Abel et al. (2016); SCN5 is set aside as an outlier.*

## Summary

A low-dimensional Lorentzian-mixture mean field (LMMF) built from the measured intrinsic frequencies reproduces the
phase-coherence dynamics of SCN1–4 through TTX application and washout. It needs four ingredients:

1. a single global Kuramoto–Sakaguchi coupling whose strength follows the network's spike rate through a
   homeostatic controller driven by the tissue TTX level;
2. phase noise proportional to the tissue TTX level, persisting partly after an incomplete washout;
3. a transient extra dephasing in one lobe during the first day of TTX;
4. a weak TTX-insensitive coupling within each lobe.

Compared with the minimal model (three constant coupling levels), the error in R(t) drops from 0.095 to 0.019
(SCN1) and from 0.097 to 0.021 (SCN2). SCN3 improves moderately (0.034 → 0.024) and SCN4 is unchanged (0.026).
The winning model captures:

- the desynchronisation rates of SCN1–3, including which lobe dephases faster;
- the lowered coherence after washout.

It does not capture the early resynchronisation that starts up to 2.5 days *before* washout in SCN2. It also does
not reproduce the slow, weak dephasing of SCN4.

![Fit results](/mnt/kennedy_labdata/richard_turbulence/data/mpmf_simulations/circadian_data/writeup/scn_lmmf_fits_prl.png)

**Figure.** LMMF fits of the SCN slices. *(a)* Intrinsic frequency distributions:
- grey histogram: per-cell phase slopes under TTX;
- black: noise-floor Lorentzian mixture;
- dashed: its lobe parts, weighted by the lobe fractions.

*(b)* Slice coherence R(t), with TTX shaded and the ±24 h windows around the treatment changes hatched (excluded
from the fit):
- data (epoch-peak phases);
- minimal model (dashed);
- winning model (red). Models are shown as c(t)|Z(t)| with the measurement ceiling c(t).

*(c)* TTX-sensitive coupling K_s(t)/K_pre of the winning model; ticks on the time axis mark washout.

*(d)* Desynchronisation rate κ_d = −d ln R/dt over days 1–3 of TTX:
- circles: slice;
- triangles: lobes;
- filled: winning model; open: minimal model.

*(e)* Resynchronisation time t_½ relative to washout: when R first passes halfway from its minimum to the
post-washout plateau.

*(f)* Post-washout plateau relative to pre-TTX coherence.

Figure script: `circadian_data/scn_writeup_figure.py`. Statistics: `writeup/scn_lmmf_fit_stats.json`.

## Data and preprocessing

**Data.** PER2::LUC single-cell traces, hourly. The `full_time_trajectories` cell set covers pre-TTX, TTX and
washout for 260–373 cells per slice. Only author-smoothed traces are available, no raw photon counts. Each slice
is one population, split into its two lobes (left and right SCN).

**Phases: epoch-wise peak pipeline** (`scn_epoch_processing.py`). Within each epoch (pre-TTX, TTX, washout):
1. subtract a polynomial baseline;
2. normalise the amplitude;
3. detect peaks with a threshold set within the epoch, refined to sub-hour precision;
4. interpolate phase linearly between peaks across the epochs, counting missed cycles.

**Validation.** Synthetic traces with known phases were generated from each cell's measured trend, envelope and
noise, then processed (`scn_pipeline_validation.py`). This pipeline was the most accurate of the three tested:
- The legacy global band-pass/Hilbert pipeline delays abrupt drops by about a day. It also produces spurious
  coherence features, including an artificial dip in SCN4.
- A peak method with one threshold for the whole recording slips cycles once amplitudes collapse under TTX.

The epoch-peak pipeline's limits:
- it spreads an abrupt change over about ±1 cycle, starting up to about half a day early;
- its coherence ceiling is about 0.93–0.97.

**Fit target** (`scn_targets.py`):
- R(t) of the slice and of both lobes;
- a measurement ceiling c(t): the R that perfectly synchronised synthetic traces reach through the same pipeline,
  with the noise level calibrated per slice so that c ≥ R in the coupled epochs;
- a mask excluding ±24 h around TTX application and washout.

The model is always compared with the data as c(t)|Z(t)|. Under TTX, c(t) falls to about 0.83 in SCN2/3 and about
0.93 in SCN1, so part of the measured coherence loss is a measurement effect and is not attributed to the dynamics.

**Ruled out before modelling:**
- **Slow frequency drift under TTX:** not detected.
- **Phase jumps at TTX application:** single-cell jumps are not separable from the low-amplitude measurement
  noise. In a mid-TTX control, which has the same amplitude, the spread of jumps is the same.
- **Lobes drifting apart:** this accounts for only 2–4% of the coherence lost under TTX (`scn_lobe_phase.py`).
  The lobes lose coherence internally.

## Model

The mean field is built from one Lorentzian mixture per lobe, fitted with a noise floor to the TTX-epoch
frequencies (panel a). Each component m (centre Ω_m, width Δ_m) evolves as

  ż_m = [iΩ_m − Δ_m − γ u(t) − γ_tr,L(m) e^{−(t − t_ttx)/τ_tr}] z_m + ½(H_m − H̄_m z_m²),

  H_m = K_s(t) e^{−iα_s} Z + K_loc Z_L(m).

- Z is the slice mean field and Z_L(m) the mean field of the component's own lobe.
- The simulation starts from the measured phases and runs continuously through all three epochs.
- The Cauchy noise γ u(t) and the lobe transient enter as extra Lorentzian widths, which is exact in this
  reduction.

**Tissue TTX level u(t)** (`scn_homeostatic_coupling.tissue_level`):
- 0 before TTX;
- rises to 1 within about 0.5 h of application;
- after washout, decays with time constant τ_off to a residual level u_res.

**Homeostatic coupling.** K_s = K_pre r/r*, where:
- r = σ(g − θ₀ − Δθ u) is the network spike rate;
- the excitability g follows τ_g ġ = (r* − r) − λg;
- the set point r* is fixed at 0.05, because it is not identifiable on its own.

The controller is reported in identifiable form, which maps one-to-one onto (Δθ, λ, τ_g):
- block depth ρ_min = K_min/K_pre;
- plateau under sustained TTX ρ_ttx = K_ttx/K_pre;
- recovery time t_hom ≈ τ_g g_ss/r*.

In this form it produces, from one mechanism:
- a sudden block;
- a delayed, S-shaped recovery under TTX;
- a rebound after washout;
- a coupling level after washout that depends on u_res.

**Models compared** (all with noise-floor mixtures):

| model | free parameters |
|---|---|
| minimal | K_pre, K_ttx, K_wash, α_s (4) |
| step + noise | + γ, u_res, τ_off (7) |
| sigmoid profile + noise | 12 |
| homeostatic + noise | 8 |
| A: + reset of the mixture state to the data at TTX + 24 h | 8 |
| B: + lobe transient γ_tr,A, γ_tr,B, τ_tr | 11 |
| **C (winning): B + K_loc** | **12** |

Rejected along the way:
- **an instantaneous phase scramble at TTX application:** it produces discontinuous R(t), which the data don't
  show;
- **phase lag fixed at 0:** α_s is needed to match the collective frequency.

**Fitting** (`scn_model_fit.py`):
- **Loss:** mean squared error of c|Z| against R for the slice and both lobes over the mask, plus a
  collective-frequency term.
- **Optimiser:** staged, nested Nelder–Mead, run on about 80 cores:
  - 64 random starts for each base model and 32 for each extension, plus warm starts from the nested models;
  - each extended model is refitted until it fits at least as well as every model nested in it.

## Results

**Fit quality** (error in R for slice / lobe A / lobe B; loss ×10⁻³):

| | minimal | winning (C) | loss minimal → C |
|---|---|---|---|
| SCN1 | 0.095 / 0.108 / 0.074 | 0.019 / 0.014 / 0.024 | 8.83 → 0.38 |
| SCN2 | 0.097 / 0.067 / 0.109 | 0.021 / 0.024 / 0.035 | 11.64 → 2.47 |
| SCN3 | 0.034 / 0.044 / 0.035 | 0.024 / 0.033 / 0.027 | 1.69 → 1.12 |
| SCN4 | 0.026 / 0.025 / 0.029 | 0.026 / 0.027 / 0.024 | 3.39 → 1.84 |

C has the lowest loss in every slice. Its extra mechanisms switch off where they aren't needed (the transient in
SCN4, the local coupling in SCN2). With about one independent sample per day, an information criterion still
favours simpler models in SCN3/4 when slices are ranked one by one, but a single model for all slices needs C's
ingredients.

**Statistics, data vs winning (minimal) model:**

| | κ_d slice (1/day) | κ_d lobes A / B | t_½ (h after washout) | R minimum (h before washout) | R_post/R_pre |
|---|---|---|---|---|---|
| SCN1 | 0.22 vs 0.21 (0.30) | 0.25/0.16 vs 0.21/0.20 | −7 vs +4 (+16) | 21 vs 0 | 0.90 vs 0.94 (0.99) |
| SCN2 | 0.42 vs 0.35 (0.21) | 0.18/0.63 vs 0.27/0.47 | −18 vs +5 (+15) | 62 vs 51 | 0.90 vs 0.91 (1.12) |
| SCN3 | 0.50 vs 0.44 (0.49) | 0.51/0.36 vs 0.58/0.34 | −13 vs −7 (+8) | 50 vs 43 | 1.06 vs 1.04 (1.09) |
| SCN4 | 0.07 vs 0.02 (0.04) | 0.08/0.06 vs 0.02/0.02 | −9 vs +8 (+20) | 54 vs 24 | 0.96 vs 0.93 (0.93) |

- **Desynchronisation:** captured in SCN1–3, including which lobe dephases faster. The minimal model gets SCN1 and
  SCN2 wrong.
- **Resynchronisation:**
  - The winning model resynchronises 10–15 h earlier than the minimal model.
  - It is still 6–23 h later than the data's t_½. About half a day of that offset is expected from the peak
    method's smearing of abrupt changes.
  - SCN2's coherence minimum 62 h before washout (51 h in the model) and its early rise are genuine re-coherence
    during TTX. The model reproduces this only partly.
- **Plateau after washout:** the winning model reproduces the data's R_post/R_pre (0.90–1.06) within 0.035, through
  the residual TTX level. The minimal model overshoots in SCN1 and SCN2 (0.99 and 1.12 vs 0.90).

**Winning-model parameters** (rad/h for couplings and widths; h for times):

| | SCN1 | SCN2 | SCN3 | SCN4 |
|---|---|---|---|---|
| K_pre | 0.17 | 0.30* | 0.09 | 0.30* |
| α_s (rad) | 0.044 | −0.013 | 0.001 | 0.041 |
| block depth ρ_min | ≈0 | ≈0 | ≈0 | ≈0 |
| plateau ρ_ttx | 0.20 | 0.16 | 1.0 | (no recovery) |
| recovery time t_hom | 88 | 41 | 111 | ≫ TTX epoch |
| rebound K_max/K_pre (peak after washout; 1/e decay) | 6.2 (1.8 h; 2.7 h) | 0.75 (6 h; 10 h) | 8.3 (3.7 h; 13 h) | — |
| K_post/K_pre | 0.50 | 0.31 | 1.0 | 0.006 |
| noise γ | 0.0055 | 0.0040 | 0.0020 | 0.0072 |
| residual TTX u_res; washout τ_off | 0.62; 0.9 | 0.80*; 6.1 | 0; — | 0.30; 0.1* |
| lobe transient γ_tr,A / γ_tr,B; τ_tr | 0.040 / 0; 15 | 0.008 / 0.028; 39 | 0.20* / 0; 1* | 0 / 0 |
| local coupling K_loc | 0.025 | 0.007 | 0.010 | 0.080 |

\* at a parameter bound.

## Interpretation (tentative)

- **Coupling under TTX.** TTX blocks the spike-driven coupling completely in every slice. In SCN1–3 it recovers
  partly after about 2–5 days (t_hom = 41–111 h), consistent with homeostatic up-regulation of excitability during
  prolonged activity block. In SCN1 and SCN3 it overshoots strongly at washout (6–8 × K_pre within hours); this
  rebound produces the fast resynchronisation.
- **Noise.** TTX-related phase noise (γ ≈ 0.002–0.007 rad/h) dephases cells beyond their frequency spread. Where
  the washout is incomplete (u_res = 0.3–0.8), it lowers coherence after washout.
- **Lobe asymmetry.** One lobe per slice dephases faster during the first day after TTX application, and the model
  attributes this to a transient extra noise in that lobe:
  - SCN1 lobe A;
  - SCN2 lobe B;
  - SCN3 lobe A, where the fit makes it a short dephasing kick.

  Once that first-day difference exists, each lobe's frequency distribution predicts the order of coherence
  between the lobes. No lobe-specific coupling is needed.
- **Local coupling.** A weak TTX-insensitive coupling within each lobe (0.007–0.025 rad/h in SCN1–3) improves the
  lobe fits. Candidate substrates are gap junctions and astrocytes, but the evidence is moderate. In SCN4 the fit
  instead carries all coupling under and after TTX through K_loc (0.08) and removes the TTX-sensitive coupling.
  SCN4's TTX effect is weak, so this is not well determined.

## Limitations and open points

- **Identifiability:**
  - K_pre is a lower bound only, because coherence before TTX sits at the measurement ceiling.
  - Several parameters sit at their bounds (marked * above).
  - The controller always chooses a complete block.
  - SCN4's solution is degenerate.

  The identifiable parametrisation (model family `hr-` in `scn_model_fit.py`) is implemented. It reproduces the
  current fits exactly, but a refit in it, and profile-likelihood uncertainties, are still to do.
- **SCN2's early recovery** under TTX would need a stronger and earlier coupling recovery than the fit allows
  together with the noise. It may remain unexplained.
- **SCN4:** its weak desynchronisation rate is underestimated (0.02 vs 0.07 per day).
- **The ±24 h masks** hide the fastest dynamics: the first-day transient and the onset of resynchronisation.
- **Scope of the fits so far:**
  - only noise-floor mixtures were used for C; the no-floor mixtures gave equal or worse fits for the earlier
    models;
  - SCN5 is not modelled.
