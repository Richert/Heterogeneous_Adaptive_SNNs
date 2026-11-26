# SCN analyses and modelling: review before integration into the LMMF paper (PNAS)

*2026-11-17. Scope: everything in `circadian_data/` to date. Each item lists what was done, how solid it is, and
what a referee would ask. The last two sections propose how the material fits a PNAS LMMF paper and list the
remaining work.*

---

## 1. Data

- Abel et al. 2016 (PNAS 113:4512). PER2::LUC imaging of 5 bilateral SCN explants from 7-day-old mice. After
  14 d in culture: 2.5 µM TTX for 6 d, washout, then 8–12 d of recording. The `full_time_trajectories` set has
  228–373 cells per slice tracked through all three epochs.
- Only author-smoothed, rescaled traces are available, not raw photon counts.
- Abel et al. report that about 40 % of cells lose observable rhythmicity under TTX. **Our analyses keep only cells
  that stay rhythmic under TTX**, because ω_i cannot be defined otherwise. The distributions are therefore those of
  the robustly rhythmic subpopulation. This must be stated; it may bias the frequency spread, in either direction.
- The two lobes are the **left and right SCN**, not core and shell. Image orientation is unknown, so dorsal–ventral
  cannot be assigned.
- SCN5 is set aside as an outlier in the dynamical fits, but it is kept in the mixture-only, bifurcation and
  entrainment analyses. **This needs one consistent rule for the paper.**

**Verdict:** a solid, public data set. The cell selection and the SCN5 rule are the only presentation issues.

## 2. Preprocessing and fit targets

- **Pipeline** (`scn_epoch_processing`, `scn_targets`):
  - each epoch is processed separately: polynomial baseline, then scalar normalisation;
  - local-threshold peaks, refined to sub-hour precision;
  - phase interpolated linearly between peaks, with cycle counting across gaps.
- **Validation** (`scn_pipeline_validation`): synthetic traces with known phases, built from each cell's own trend,
  envelope and noise.
  - The legacy global Hilbert pipeline delays abrupt drops by about 1 d and creates spurious dips, including
    SCN4's apparent TTX dip.
  - Epoch-wise Hilbert pins phases at the epoch edges.
  - The epoch-peak method has no edge artefact, but it spreads a change over about ±1 cycle and saturates at
    R ≈ 0.93–0.97.
- **Measurement ceiling c(t):** the R of perfectly synchronised synthetic traces after the same pipeline. Its noise
  scale is calibrated so that c ≥ R in the coupled epochs. Under TTX it falls to 0.83 (SCN2/3) and 0.93 (SCN1).
  The model is always compared as c(t)|Z(t)|.
- **Mask:** ±24 h around TTX application and washout.
- **Ruled out:**
  - slow frequency drift under TTX (`scn_ttx_frequency_drift`, against an AR(1) null);
  - phase jumps at TTX application: the spread equals a mid-TTX control (`scn_ttx_phase_reset`);
  - lobes drifting apart: 2–4 % of the coherence loss (`scn_lobe_phase`).

**Verdict:** strong, and a selling point for PNAS: R(t) is a validated observable with an explicit measurement
model. Referee risks:
- The ceiling calibration is a one-parameter fudge. Show its sensitivity: results with s = 1 vs the calibrated s.
- The ±24 h mask hides the fastest dynamics (the first-day lobe transient and the onset of resynchronisation),
  which are exactly what the more complex model ingredients explain. The paper should say this.

## 3. Intrinsic frequencies

- ω_i is the least-squares phase slope over the TTX epoch, excluding its first 24 h.
- Per-cell measurement error from the ground-truth synthetics is **0.010–0.025 c/d**.
- The mixtures use the fitter of the revised LMMF paper: CvM loss, warm start, noise-floor cap, λ = 1e-5.
  - Pooled noise-floor fits: SCN1 M = 2 (0.91 and 1.01 c/d), SCN3 M = 2, SCN5 M = 2; SCN2 and SCN4 M = 1.
  - The no-floor fits have more components.

**Three open problems, in decreasing severity:**

1. **The widths are of the same order as the measurement error.** The fitted component widths are
   Δ ≈ 0.005 rad/h ≈ 0.02 c/d (SCN1), which is the same size as the per-cell measurement error. A Lorentzian fitted
   to noisy ω_i estimates the *convolution* of the intrinsic distribution with the error distribution. The mean
   field therefore over-states the intrinsic heterogeneity, and so over-states K_c and the inferred couplings.
   - The empirical reference already uses h = 0.1 × error to avoid double counting.
   - The proper treatment, "option 2" (fit with the measurement error convolved in, i.e. deconvolve the widths),
     is still TODO.
   - **This is the first thing a PNAS referee in the field would raise.** For a Lorentzian error kernel the
     deconvolution is exact and trivial (Δ_intrinsic = Δ_fit − h). For a Gaussian error it is approximately so.
2. **"Intrinsic" frequencies are measured with residual coupling present.** The winning model has a TTX-insensitive
   coupling K_loc (0.007–0.025 rad/h), and K_s recovers partly under TTX. With coupling present the measured slopes
   are pulled toward each other. The TTX distribution is then a *lower* bound on the intrinsic spread, which partly
   offsets point 1.
   - **Self-consistency test:** simulate the fitted network model under TTX, re-estimate the slopes, and compare
     them with the ω_i it was given.
   - This is cheap (seconds to minutes) and would turn a vulnerability into a result.
3. **Spatial structure (new, today; `scn_radial_frequency.py`).** Frequency correlates with distance from the lobe
   centroid:
   - **Peripheral cells are faster.** The signed Spearman ρ > 0 in 8/10 lobes. Pooled: ρ = 0.13, p = 4e-7;
     outer − inner tercile = +0.018 c/d (about 27 min shorter period), p_perm = 1e-4.
   - The periphery also has a larger spread (pooled ρ = 0.18, p = 3e-12).
   - In the bimodal slices the **fast cluster is enriched at the periphery**: fraction in the inner vs outer
     tercile is 0.18 → 0.60 (SCN1 lobe B), 0.17 → 0.36 and 0.02 → 0.29 (SCN3).

   **Caveats:**
   - Peripheral cells may have lower SNR, which would explain the larger spread, though not obviously a signed
     shift.
   - The radius from the centroid is a crude stand-in for shell vs core.

   **Implications:**
   - The two frequency clusters may be partly *anatomical* populations, roughly core and shell.
   - This weakens the claim in the literature assessment that our internal-desynchronisation mechanism is
     "frequency-only and distinct from the anatomical one". It could instead *connect* the two.
   - Recalled, not yet verified: earlier reports of regional period differences within the SCN (shorter dorsal
     periods). Check before citing.

## 4. Dynamical model and fits (`scn_model_fit`, winning setup C)

- **Model:** LMMF per lobe plus:
  - homeostatic TTX-sensitive coupling K_s(t), from a rate controller on the tissue TTX level u(t);
  - TTX-linked Cauchy noise γ u(t);
  - a lobe-specific first-day transient;
  - a TTX-insensitive lobe-local coupling K_loc;
  - a phase lag α_s.

  That is 12 parameters per slice. The fits use staged nested Nelder–Mead on about 80 cores.
- **Fit quality, RMSE of R(t), minimal → C:**

  | Slice | Minimal | C |
  |---|---|---|
  | SCN1 | 0.095 | 0.019 |
  | SCN2 | 0.097 | 0.021 |
  | SCN3 | 0.034 | 0.024 |
  | SCN4 | 0.026 | 0.026 |

  C captures the desynchronisation rates, including which lobe is faster, and the post-washout plateau.
  Resynchronisation is still 6–23 h late. SCN2's early recovery under TTX and SCN4's weak dephasing are not
  captured.
- **Weaknesses:**
  - 12 parameters against about one independent sample per day per curve.
  - Several parameters at bounds; the controller always chooses a complete block.
  - SCN4 is degenerate; K_pre is only a lower bound.
  - No profile likelihoods yet. The identifiable parametrisation exists (`hr-`), but the refit is not done.
- **Mechanism plausibility:** a partial recovery of coupling under TTX plus a rebound at washout.
  - Supported only indirectly (slide 8 of the deck): TTX-resistant Ca²⁺ oscillations in SCN neurons, intrinsic
    plasticity after 2 d of TTX, firing-rate homeostasis.
  - There is no direct evidence for recovered spiking in SCN under days of TTX.
  - Abel et al.'s Fig. 1C shows no recovery of peak-time correlation under TTX.

**Verdict:** good enough as a *vehicle*: it gives realistic coupling trajectories against which the frequency
distribution can be tested. It is not strong enough as a mechanistic claim for a main-text PNAS result. Present it
as a phenomenological coupling model, with the homeostatic interpretation as a hypothesis, and put details in the SI.

## 5. Level of detail of the distribution: the frozen comparison (`scn_main_figure`, `scn_supp_figure`)

- **Design:** fit the dynamics once with the empirical distribution (one narrow Lorentzian per cell), then freeze
  them. Swap in three mixtures (no floor, noise floor, single Lorentzian) without refitting.
- **Results:**
  - The error relative to the reference grows with the CvM distance: error ∝ D^0.48 (r = 0.88).
  - The ranking no floor < floor < single holds in every slice.
  - SCN1 RMSE vs data: empirical 0.026, no floor 0.026, floor 0.034, single 0.075.
  - Single Lorentzians fail where the distribution is multimodal (SCN1/3).
- **Mixture-only tests** (`scn_bifurcation_mixtures`, global K only, no fit):
  - coarser distributions inflate the coupling inferred from the plateaus: K_pre/K_c = 9.9 / 17.7 / 28.1 for
    no floor / floor / single in SCN1;
  - frequencies alone over-predict the dephasing rate in SCN1;
  - SCN2 lobe B dephases beyond its frequencies;
  - with a constant K_post, SCN1 resynchronises too slowly (t½ model 40 h vs data 13 h), which motivates the
    rebound.

**Verdict:** this is **the result that belongs in the LMMF paper**. It is the empirical counterpart of the paper's
continuity argument (Z_M → Z as ρ_M → ρ): distribution error maps monotonically onto dynamical error *in real
data*. Issues:
- Point 3.1 applies: the "empirical" reference inherits the measurement error.
- n = 4 slices. Present per-slice points, not a fitted exponent with a p-value.

## 6. Bifurcation structure (`scn_pycobi_bifurcation`, `scn_bif_ramp_figure`)

- **Model:** the minimal model (pooled mixtures, global K, α = 0, no noise). Synchrony onset at K_c is a
  supercritical Hopf in the lab frame (pitchfork in the co-rotating frame).
- **SCN1, noise-floor mixture:**
  - K_c = 0.018;
  - Hopf at 0.0235;
  - stable limit cycle (cluster beating) up to a homoclinic at about 0.0353;
  - folds at 0.0353 / 0.0368, with bistability.

  The single Lorentzian has K_c = 0.025 and none of this.
- **SCN3:** a limit-cycle window between Hopfs at 0.029 and 0.043 for the pooled noise-floor fit. **The per-lobe
  M = 4 fit has none, so this is fit-sensitive.**
- **SCN2 and SCN4:** M = 1, so the two models are identical.

**Concerns:**
1. **Noise is ignored.** The fits need γ ≈ 0.002–0.007 rad/h of phase noise. Under OA, Cauchy noise adds to every
   width (Δ_m → Δ_m + γ), which would roughly double SCN1's cluster widths.
   - **Does the limit-cycle window survive Δ_m + γ?**
   - If not, the beating prediction is irrelevant for real slices.
   - This is a cheap PyCoBi run per slice and should be done before any claim.
2. The interesting window (K ≈ 1.3–2 K_c) lies far below the estimated intact coupling (K_post ≈ 3.6 K_c,
   K_pre ≈ 10–18 K_c). The relevance argument has to be "weakened networks" (ageing, VIP-receptor block, recovery
   after TTX). Note that the fitted K_s(t) passes through this window during washout recovery, which gives a
   natural link.
3. Point 3.1 again: deconvolved widths would shift K_c and possibly widen the limit-cycle window.

## 7. Entrainment (`scn_entrainment`, `scn_arnold_tongue`, `scn_entrainment_examples`)

- 30 × 30 coherence maps over (f_d, F) at K_c, K_mid and K_post per slice, with PyCoBi tongue edges (folds and
  tori, kept where the entrained cycle changes stability).
- **At K_post** the tongues are almost independent of the mixture.
- **At K_mid in SCN1** they differ: there are partial-entrainment tongues of single clusters. In the example, the
  1.01 c/d cluster locks at f_d = 1.08, F = 0.07 while the 0.91 cluster drifts.
- **Caveats:**
  - All cells receive the same light; in vivo only retinorecipient (core) cells do.
  - F is uncalibrated.
  - Explants are not light-responsive.
  - The forced bimodal OA system has apparently not been analysed (limited literature search).

## 8. Jet lag (`scn_jetlag`)

- The east–west asymmetry, the "long way round" for 8 h east above about 2 K_c, and a re-entrainment time nearly
  flat in K are all known from Lu et al. 2016 (OA, single Lorentzian).
- **New:**
  - in bimodal SCN1 the two clusters re-entrain in opposite directions (coherence dip to about 0.25);
  - eastward re-entrainment slows with K (SCN2: 7 → 12 d), i.e. coupling buys coherence, not speed.
- In view of point 3.3, the cluster split may map onto core vs shell. That is an interesting connection to the
  ventral/dorsal resetting literature.

## 9. Positioning (from `writeup/scn_literature_assessment`)

- **Not new:** the forced-Kuramoto bifurcation structure (Childs & Strogatz 2008), bimodal limit cycles (Martens
  et al. 2009) and the jet-lag phenomenology (Lu et al. 2016).
- **New:**
  - a data-driven mean field, with mixtures fitted per slice and couplings inferred from TTX/washout;
  - a quantitative statement of *when the distribution's detail matters* (intermediate coupling);
  - slice-specific predictions.

  OA-based human circadian models (Hannay, Booth & Forger 2018/19) use one Lorentzian. Our result says when that
  is adequate (intact SCN) and when not (weakened SCN).

---

## 10. Proposal: how the SCN work enters a PNAS LMMF paper

PNAS research article: Significance statement (≤ 120 words), about 4,500 words of main text with 4–6 figures, and
an SI. The current letter is method-first, with the Allen PV/pyramidal QIF applications in an appendix.
For PNAS, the SCN data could be the **flagship application**. It is the only data set where the LMMF is tested
against measured *dynamics*, not just used to make predictions.

**Proposed main-text arc:**

1. **Method** (the current Fig. 1 plus the continuity result): mixtures with a noise floor; ρ_M → ρ ⇒ Z_M → Z.
2. **Benchmarks:** Skardal / rational distributions (Fig. 2) and the unimodal/bimodal network figures (the new
   PNAS figures).
3. **SCN, part 1: distribution detail predicts measured coherence dynamics.**
   - New main figure:
     - (a) data overview, R(t) through TTX;
     - (b) intrinsic frequencies and mixtures, SCN1 vs SCN2;
     - (c) frozen comparison, error vs CvM distance;
     - (d) inferred couplings by distribution detail.

     These are the existing `scn_main_figure` panels, restyled for PNAS.
   - The fitted coupling model is described briefly; details in the SI.
4. **SCN, part 2: what the mean field predicts.** Bifurcation in K, limit-cycle window, tongues, and the
   jet-lag cluster split. Uses the existing `scn_bif_entrain_figure` plus one jet-lag panel; SCN1 bimodal vs SCN2
   unimodal.
5. **Neural heterogeneity** (Allen PV/pyramidal QIF): shortened, or moved to the SI, to keep the paper focused.
   **This is the main structural decision.**
6. **Discussion:** when a single Lorentzian is enough; weakened networks; the core/shell link (point 3.3).

## 11. Work list before writing

Ordered by how much a referee would care. "Run" marks items that need compute; estimates are for this machine.

| # | Item | Why | Effort |
|---|---|---|---|
| 1 | Measurement-error deconvolution of the mixtures (option 2). Redo the frozen comparison and K_c | Point 3.1, the obvious referee objection | Code ~1 h. Run: mixtures seconds; frozen re-evaluation minutes; a re-fit of the empirical reference is a server run |
| 2 | Self-consistency of the TTX frequencies: simulate the fitted model under TTX and re-estimate the slopes | Point 3.2 | Minutes |
| 3 | Limit-cycle window with noise (Δ_m + γ), SCN1/SCN3 | Point 6.1; decides whether the beating prediction survives | PyCoBi, ~10 min per slice, run sequentially |
| 4 | Sensitivity to the ceiling calibration (s = 1 vs calibrated) | Point 2 | Re-evaluation minutes; refit is a server run |
| 5 | Profile likelihoods of the C fits (`hr-` family) | Point 4; only needed if the dynamics stay in the main text | Server |
| 6 | Radial frequency: SNR control (partial correlation given SNR); orientation-free core proxy (e.g. distance to the Abel hub centre) | Point 3.3 | Minutes |
| 7 | One rule for SCN5 (include everywhere, or exclude everywhere with a stated reason) | Consistency | Decision |
| 8 | PNAS restyle of the SCN figures (`pnas` preset, 17.8 cm) | Submission | After 1–3 |
| 9 | Verify the literature items marked "URL only" in the assessment, and the regional-period citation | Citations | Reading |

---

## 12. Robustness checks, risks 1–3 (2026-11-17, `scn_robustness.py`; outputs in `circadian_data/robustness/`)

### Risk 1: measurement error vs mixture widths (`deconv`)

The per-cell error h (median |ω_est − ω_true| from the synthetic traces, i.e. the half-width of a Cauchy error) is
0.016 / 0.025 / 0.023 / 0.010 c/d for SCN1–4. **It is as large as almost every fitted component width.**

- **Deconvolving collapses the widths.** Subtracting h from the noise-floor widths drives most of them to ≈ 0:
  - SCN1 widths go from 0.018/0.023/0.020/0.014 c/d to 0.002/0.008/0.004/0.0;
  - K_c drops to ≈ 0 in SCN1–3, because a zero-width component synchronises at any K.
- **h is probably overestimated in places.** SCN2 lobe A has an *observed* width of 0.015 c/d, smaller than
  h = 0.025. That is impossible for a Cauchy error of that size, so the synthetic noise model overestimates the error
  there, or the error is not Cauchy-like.
- **What is robust and what is not:**
  - The component *centres and weights* are robust: SCN1's cluster separation of about 0.1 c/d is far larger than h.
  - The *widths*, and with them K_c, are **not identifiable** from these data.
- **Coupling-free dephasing from the measured state** (κ_d, 1/day, lobe A | B):

  | Slice | Data | Measured ω | Floor | Floor, deconvolved |
  |---|---|---|---|---|
  | SCN1 | 0.23 \| 0.15 | 0.34 \| 0.25 | 0.40 \| 0.33 | 0.31 \| 0.23 |
  | SCN2 | 0.11 \| 0.55 | 0.07 \| 0.07 | — | — |
  | SCN3 | 0.43 \| 0.30 | 0.31 \| 0.20 | — | — |
  | SCN4 | 0.08 \| 0.06 | 0.05 \| 0.02 | — | — |

  The frequencies over-predict dephasing in SCN1 but under-predict it in SCN2–4. There is no single direction.
- **Frozen at the empirical C fit, deconvolved mixtures are worse** in SCN1–3 (e.g. SCN1 slice RMSE 0.053 vs 0.020
  floor). That is expected: the dynamics were fitted to the broadened distribution. A **refit with `floor-dc`** (server)
  is needed before concluding anything; the kinds are registered in `scn_model_fit` (`--kinds floor-dc,single-dc`).

### Risk 2: are the TTX slopes intrinsic? (`selfcons`; microscopic model C, empirical fit, 3 seeds)

| Slice | Full model ρ_S | Full model scale ratio | Coupling only ρ_S | Coupling only scale ratio | Noise only ρ_S | Noise only scale ratio | Noise only med \|s − ω\| |
|---|---|---|---|---|---|---|---|
| SCN1 | 0.49 | 1.10 | 0.92 | 0.14 | 0.55 | 1.40 | 0.042 c/d |
| SCN2 | 0.40 | 2.31 | −0.35 | 0.04 | 0.46 | 2.93 | — |
| SCN3 | 0.68 | 0.89 | 0.99 | 0.51 | 0.74 | 1.19 | — |
| SCN4 | 0.00 | 0.99 | 0.44 | 0.16 | 0.30 | 2.83 | — |

ρ_S is the Spearman correlation of the TTX-window phase slopes with the intrinsic ω. The scale ratio is the Cauchy
scale of the slopes divided by that of ω.

- **Coupling only:** the fitted coupling under TTX compresses the slope spread to 4–50 %. The cells partly lock,
  because K_s recovers (SCN1: ρ_ttx · K_pre ≈ 0.08 rad/h, about 4.6 K_c) and K_loc ≈ 0.035 rad/h.
- **Noise only:** Cauchy noise of scale γ gives each slope an error of scale γ, *independent of the window length*,
  because Cauchy increments do not average out. In OA, this noise is *mathematically identical* to extra Lorentzian
  frequency heterogeneity. The fitted γ (SCN1 empirical fit: 0.011 rad/h = 0.043 c/d) is as large as the whole
  observed frequency spread.
- **Verdict: model C is not self-consistent with using the TTX slopes as intrinsic frequencies.** If the model were
  true, the measured slopes would be only moderately related to the intrinsic ω (ρ_S 0.0–0.7). The γ term
  double-counts heterogeneity that the measured slopes already contain.

### Risk 3: does the limit-cycle window survive noise? (`lcnoise`; minimal model, pooled mixture, Δ_m → Δ_m + g)

At g = 0 the classification reproduces the PyCoBi window (SCN1 0.022–0.035 vs Hopf 0.0235 to homoclinic 0.0353).

- **SCN1, observed widths:** the window survives up to g ≈ 0.011 c/d and is gone at ≥ 0.015. The fitted noise is
  γ = 0.021 c/d under TTX (floor C fit) and γ u_res = 0.013 c/d after washout: **at both levels the window is gone.**
- **SCN1, deconvolved widths:** the window survives up to g ≈ 0.027 c/d, so it is present at both noise levels.
- **SCN3, observed widths:** a window only at g = 0. **Deconvolved:** a window up to g ≈ 0.008; the fitted γ of
  0.0085 c/d is borderline.
- **Verdict:** the limit-cycle (beating) prediction is **conditional** on the widths and the noise level, which are
  exactly the quantities risk 1 shows to be unidentified. Present observed widths (upper bound on heterogeneity) and
  deconvolved widths (lower bound) as a *bracket*. Claim the window only where it exists across the bracket.

### Consequences

- **Survives:**
  - The frozen comparison as a statement about the *measured* distribution, since its reference is the measured
    distribution.
  - The SCN1 (bimodal) vs SCN2 (unimodal) contrast in centres and weights.
- **Must be re-done or re-framed:**
  - Any quantitative K_c, limit-cycle window or near-K_c tongue.
  - The mechanistic reading of model C: the partial recovery of K_s under TTX and γ trade off against each other and
    against the broadened frequency spread.
- **Next steps:**
  - (a) A data-driven error estimate: split-half slopes per cell (first vs second half of the TTX window), with no
    synthetic noise model.
  - (b) A self-consistent noise treatment: either drop γ under TTX (the measured widths already contain the noise
    broadening), or fit the intrinsic distribution so that the model *generates* the observed slope distribution.
  - (c) A server refit of C with `floor-dc`, to test whether the K_s recovery and K_loc survive.

## 13. Premise tests for a revised frequency inference (2026-11-17, `scn_frequency_inference_tests.py`; outputs in `freq_inference/`)

### T1: phase diffusion vs peak-timing error, from the TTX peak intervals

Method: mean-free moments of the interval differences, so no synthetic noise model is involved.
- Under TTX the interval jitter is almost entirely **non-accumulating** peak-timing error:
  σ_m = 2.7 / 4.7 / 4.4 / 1.4 / 1.6 h for SCN1–5.
- Phase diffusion σ_D is not detectable (≈ 0) except in SCN4 (1.4 h).
- In the coupled epochs there are too few consecutive cycles to estimate it.
- **Caveat:** spurious extra peaks in the low-amplitude TTX traces would also inflate σ_m.

What this means for the per-cell frequencies:
- The per-cell frequency SE (median 0.03–0.09 c/d) is **as large as or larger than the observed spread** (half-IQR
  0.014–0.076 c/d).
- Two estimators of the same cells' frequencies (endpoints vs least squares) agree only with ρ = 0.48–0.81.
- **Per-cell TTX frequencies are imprecise, but unbiased.**
- Deconvolving with each cell's own Gaussian SE (Voigt maximum likelihood, M = 1) leaves **finite widths**:

  | Slice | Observed width (c/d) | Deconvolved width (c/d) |
  |---|---|---|
  | SCN1 | 0.040 | 0.017 |
  | SCN2 | 0.040 | 0.025 |
  | SCN3 | 0.068 | 0.040 |
  | SCN4 | 0.018 | 0.004 |
  | SCN5 | 0.012 | 0.001 |

  The M = 2 deconvolutions are unstable: outlier components appear near 0.6 c/d.
- **Because the noise does not accumulate, the population dephasing under TTX measures the true frequency spread.**
  For a Lorentzian, κ_d = Δ: SCN1 0.215/d gives 0.034 c/d, SCN3 0.50/d gives 0.080 c/d, SCN4 0.07/d gives 0.011 c/d.
  These lie between the observed and the deconvolved per-cell widths. So R(t) under TTX should constrain the
  *widths*, and the per-cell slopes the *centres and shape*.

### T2: the locked-state premise (ω_i vs the relative phase when coupled)

- In SCN1–3, faster cells lead before TTX and after washout:

  | Slice | ρ(φ_pre, ω̂) | ρ(φ_post, ω̂) |
  |---|---|---|
  | SCN1 | 0.29 | 0.33 |
  | SCN2 | 0.21 | 0.35 |
  | SCN3 | 0.13 | 0.17 |

  All are significant. The correlations are attenuated by the noisy ω̂.
- SCN4 and SCN5 show no relation.
- The relative phase before TTX predicts the one after washout: ρ = 0.30–0.64.
- The implied coupling K = 1/(slope·R) is about 0.2–1.3 rad/h. These are upper bounds, because of the attenuation.
- **Verdict:** the premise holds, but per cell it is too noisy to sharpen ω_i much. Its use is as a population-level
  constraint on K.

### T3: core vs shell (inner vs outer tercile of the distance from the lobe centroid)

- **In all 5 slices the outer tercile:**
  - **leads in phase** before TTX and after washout, by +0.3 to +1.9 h;
  - **dephases 1.5–6× faster** under TTX: κ_d is 0.36 vs 0.15 (SCN1) and 0.80 vs 0.24 (SCN2);
  - reaches a much lower minimum R;
  - is less coherent even when coupled.
- In SCN3 the outer cells drift steadily ahead of the inner ones under TTX, which means they are faster.
- **Caveats:**
  - There is no ceiling correction per tercile. Lower SNR at the edge would lower R but would not produce a mean
    phase lead.
  - The radius is only a stand-in for core vs shell.
- **Verdict:** strong support for a two-ensemble (core/shell) model. The core is coherent, slower and probably more
  strongly or residually coupled; the shell is faster and broader, and probably weakly coupled.

## 14. Two-ensemble (core/shell) model, first fits (2026-11-18, `scn_coreshell_targets.py`, `scn_coreshell_model.py`; outputs in `coreshell/`)

### Targets and checks

- Core = inner half of each lobe, shell = outer half; lobes pooled.
- **The ceilings of core and shell are equal**, so the shell's lower coherence is real.
  - When coupled: 0.96–0.99 for both.
  - Under TTX: equal.
  - After washout the shell lies below its ceiling (SCN1: 0.81 vs 0.97).
- **Abel's MIC networks agree with the split.**
  - Degree vs radius: ρ = −0.23 to −0.42 in all slices.
  - Core cells have 1.1–4× the shell's degree.
  - Unconnected cells: 30–71 % of the shell vs 21–40 % of the core.
  - Only 82–132 cells per slice could be matched by position.

### First fits

- Amplitude-linked noise is included: D₀ (A_pre/A_e(t))².
- The measured single-cell amplitudes fall to 10–30 % of pre-TTX under TTX. SCN4 drops as much as SCN1 but barely desynchronises.
- **The directed model beats the symmetric one** in SCN1, SCN3 and SCN4 (loss 6.3 → 1.1, 3.5 → 1.8, 2.8 → 1.6 ×1e-3). There is no gain in SCN2 and SCN5.
- **Slice RMSE** (data vs model):

  | | SCN1 | SCN2 | SCN3 | SCN4 | SCN5 |
  |---|---|---|---|---|---|
  | Directed core/shell | 0.024 | 0.077 | 0.033 | 0.021 | 0.031 |
  | Model C (for comparison) | 0.019 | 0.021 | 0.024 | 0.026 | — |

- SCN3 strongly favours the directed structure: K_cc = 1.6, K_cs = 0.018, ε = 0.03. In words, a strongly coupled core with a weakly driven shell. Its core–shell phase excursion is partly captured.

### Problems

1. **Resynchronisation after washout is too slow** (SCN2, SCN3). ρ_post goes to its bound of 3 with τ = 0.5 h, so the fit wants a rebound, as model C had.
2. **Block vs noise is not identifiable in SCN1.** The fit keeps full coupling under TTX (ρ_ttx = 1) and desynchronises through amplitude noise alone; after washout the coupling decays to 5 %. That is biologically implausible.
3. Many parameters sit at their bounds: K_cc = 2, K_g = 0.3, ρ_post = 3.
4. The shell's steady phase drift ahead of the core under TTX in SCN1 (to +6 h) is not reproduced.

## 15. Core/shell fits with the user's coupling time course (server scan, 2026-11-18)

- **Coupling time course:** block at t_ttx → linear recovery ramp from t_rec → washout level → relaxation to the
  post-washout level.
- **D₀ scan** (shared D₀, directed model), total loss:

  | D₀ | 2e-4 | 5e-4 | 1e-3 |
  |---|---|---|---|
  | Total loss (×1e-3) | 11.2 | 9.7 | 8.2 |

  The best D₀ (1e-3) is at the edge of the scan. SCN2 drives the trend; SCN1 is best at 5e-4.
- **Slice RMSE at D₀ = 1e-3:**

  | | SCN1 | SCN2 | SCN3 | SCN4 | SCN5 |
  |---|---|---|---|---|---|
  | Directed | 0.023 | 0.030 | 0.024 | 0.022 | 0.015 |
  | Model C | 0.019 | 0.021 | 0.024 | 0.026 | — |

- **The directed model beats the symmetric one in every slice**: loss 6.3 → 1.0, 8.2 → 3.5, 2.1 → 0.85,
  3.6 → 1.4 and 2.2 → 1.5 (×1e-3). The symmetric model cannot separate core and shell (RMSE ≈ 0.1 in SCN1/2).
- **The asymmetry is in the input: the core receives stronger coupling than the shell.**

  | | SCN1 | SCN2 | SCN3 |
  |---|---|---|---|
  | K_cc (rad/h) | 0.18 | 2.6 | 0.032 |
  | K_cs (rad/h) | 0.064 | 0.41 | 0.009 |

  The shell's output factor ε sits at 1 in SCN1–4, so a weak shell output is not supported.
- **SCN2's early recovery is captured.** The ramp starts 107 h before washout, from a complete block. SCN3's starts
  54 h before washout, with a strong rebound (f_wash 7.9, f_post 2.6).
- **Open problems:**
  1. SCN1 still has no block at D₀ = 1e-3 (f_ttx = 1). At 5e-4 it has a partial block (f_ttx = 0.47) and a slightly
     lower loss.
  2. SCN1's shell drifting ahead under TTX (to +6 h) is not reproduced. A drift from the core–shell frequency
     difference requires a block, so the Δψ term (weight 1e-3, only where both R > 0.3) is too weak to arbitrate
     between block and noise.
  3. Several parameters sit at their bounds: ε, K_g → 0, and the width factors s_c and s_s at 0.05 or 2 in
     SCN2/4/5.
  4. The implied noise is large: D₀·q reaches about 0.1 rad/h under TTX (q ≈ 100 in SCN1). This contrasts with the
     undetectable accumulating phase diffusion in the interval test (§13, T1), and it would make the TTX dephasing
     noise-dominated rather than set by the frequency distribution.

## 16. Phase term weighted by sampling error, ε = 1 (local, D₀ = 1e-3, 16 starts, 2026-11-18)

**SCN1 now has a block, but at a cost in coherence fit.**
- The TTX block is now partial (f_ttx = 0.11; before, f_ttx = 1).
- The coupling *declines further* during TTX: the ramp runs from 0.11 down to f_wash = 0.027, starting
  108 h before washout. After washout it jumps to 0.6.
- A strong TTX-insensitive local coupling (K_g = 0.19) keeps each ensemble internally coherent. With the core–shell
  coupling blocked, core and shell then drift apart, and the shell runs ahead (model +2 h; data up to +6 h).
- The coherence fit worsens: slice RMSE 0.023 → 0.039, and the shell after washout is too high (0.90 vs 0.81).
- Both width factors are at their upper bound of 2.

**The other slices:**
- SCN2 is unchanged.
- SCN3 is similar. Its phase term (2.5e-3) is now dominated by the phase wrap near the coherence minimum.
- SCN4 and SCN5 are fitted well, but with degenerate width factors (s = 0.05) and K_g at its bounds.

**Identifiability:**
- τ_post sits at its lower bound (0.5 h) in SCN1, SCN2 and SCN4. The ±24 h mask around washout hides how fast
  coupling recovers.
- With 16 starts, the best loss is reproduced less consistently than with 32 (SCN5: 1.8 vs 2.5 ×1e-3 before the
  polish).

## 17. Server fits, width factors in [0.02, 4], τ_post = 2 h, 32 starts (2026-11-18)

**Total loss:** 11.6e-3 at D₀ = 5e-4 vs 14.7e-3 at 2e-3. The 1e-3 run on the server was not done; the existing
1e-3 files are the old local setup.

**At D₀ = 5e-4:**

| | SCN1 | SCN2 | SCN3 | SCN4 | SCN5 |
|---|---|---|---|---|---|
| Slice RMSE | 0.025 | 0.019 | 0.026 | 0.025 | 0.029 |

The starts agree (the top three are within 0.01e-3 in SCN1–4).

**SCN1 now tells a coherent story:**
- complete TTX block (f_ttx at its lower bound);
- a TTX-insensitive local coupling (K_g = 0.13) keeps each ensemble coherent;
- core and shell drift apart: model +2.8 h, data up to +6 h, with phase term 0.25e-3;
- a strong rebound at washout (f_wash at its bound of 10, i.e. 0.5 rad/h), relaxing to 0.47 × K_pre.

**SCN2:** a complete block, recovery starting under TTX, and the core receiving more than the shell
(K_cc = 2.6 vs K_cs = 0.77).

**The width factors run to opposite bounds:**
- core → 0.02 in SCN1, SCN4 and SCN5 (SCN3 at 2e-3);
- shell → 4 in SCN1, SCN2, SCN4 and SCN5.

So the core–shell difference under TTX is explained by a near-zero core width and a much broader shell, instead of
by the slope-derived mixtures or by coupling. The input asymmetry K_cc > K_cs is now inconsistent across slices: in
SCN1 0.050 vs 0.049, in SCN5 0.59 vs 0.67. Widths and coupling trade off against each other.

**Artefact:** SCN5 shows a spike at washout, where the rebound reaches its bound of 10 with τ_post = 2 h. It lies
behind the mask.

**Proposal:**
- Share one width factor between the ensembles, so that the core–shell frequency differences come from the data.
- Give each ensemble its own local coupling, K_g,c and K_g,s.
- Cap f_wash.

## 18. dirg: shared width factor, ensemble-specific local coupling, rebound cap 5 (server, 2026-11-18)

**Total loss:** 10.78e-3 at D₀ = 2.5e-4 and 11.36e-3 at 5e-4, against 11.60e-3 for dir at 5e-4 with the same number
of parameters. The starts are consistent.

**Slice RMSE at D₀ = 2.5e-4:**

| | SCN1 | SCN2 | SCN3 | SCN4 | SCN5 |
|---|---|---|---|---|---|
| Slice RMSE | 0.024 | 0.012 | 0.027 | 0.025 | 0.027 |

SCN1's core–shell drift is reproduced up to +3 h.

**The input asymmetry is now consistent: the core receives more TTX-sensitive coupling than the shell in SCN1–4.**

| | SCN1 | SCN2 | SCN3 | SCN4 | SCN5 |
|---|---|---|---|---|---|
| K_cc / K_cs (rad/h) | 0.37 / 0.068 | 2.5 / 1.34 | 0.049 / 0.014 | 2.55 / 0.25 | 0.67 / 0.76 |
| Ratio | 5.4× | 1.9× | 3.4× | 10× | ≈ 1 |

**Local (TTX-insensitive) coupling is larger in the core** in SCN1, SCN2, SCN4 and SCN5:

| | SCN1 | SCN2 | SCN4 | SCN5 |
|---|---|---|---|---|
| K_g,c / K_g,s (rad/h) | 0.11 / 0.065 | 0.045 / 0 | 0.21 / 0.12 | 0.085 / 0 |

SCN3 is reversed (0.005 vs 0.018).

**The width factor exceeds 1:** s = 3.5 (SCN1), 4 (SCN2, at its bound), 1.0 (SCN3), 1.2 (SCN4), 1.3 (SCN5).
- The model needs more dephasing than the slope-derived widths give.
- This rises as D₀ falls, so noise and width share one dephasing budget.
- In SCN1, s = 3.5 widens each cluster to about 0.07 c/d against a separation of 0.09 c/d. That is close to the point
  where the mixture stops being bimodal, which matters for the bimodal story.
- Peak-timing error makes the measured slope spread *larger* than the intrinsic one, so s > 1 cannot be measurement
  error. It points to extra dephasing that the model is missing, or to the fast drop after TTX application being
  absorbed into the widths.

## 19. dirg at D₀ = 1e-4 and 0 (server, 2026-11-18)

**Total loss** (×1e-3):

| D₀ | 5e-4 | 2.5e-4 | 1e-4 | 0 |
|---|---|---|---|---|
| Total loss | 11.36 | 10.78 | **10.56** | 10.94 |

**The amplitude noise barely matters.** D₀ = 0 is within 4 % of the best value, so the noise term can be dropped.

**The width factor goes to its bound** (s = 4) in SCN1 and SCN2 at D₀ ≤ 1e-4.
- In SCN1 that widens each cluster to a half-width of about 0.076 c/d, against a separation of 0.094 c/d.
- For equal Lorentzians the mixture is bimodal only if the separation exceeds 2γ/√3 ≈ 0.088, so SCN1's bimodality is
  effectively gone.

**Robust across D₀:**
- TTX blocks the coupling completely (f_ttx at its lower bound) in SCN1–3.
- SCN1 and SCN3 show a strong rebound at washout (f_wash at the cap of 5).
- The core receives more coupling than the shell in SCN1, SCN3 and SCN4 (about 3.5–11×).

**Not robust:**
- SCN2's coupling asymmetry (1.9× at 2.5e-4, 0.84× at 0).
- The core-vs-shell local coupling in SCN3 and SCN4, which flips sign.
- SCN1's u_rec stays at its start value of 0.8 at every D₀, so the recovery onset is not identifiable there.
- SCN3's phase term (≈ 2.2e-3) stays the largest single error: the model does not follow the phase wrap at the
  coherence minimum.

## 20. Intrinsic-frequency figure, step 1 of the revised plan (2026-11-25, `scn_frequency_figure.py`)

Figures: `writeup/scn_frequency_inference_{epoch,wavelet}.{png,pdf,json}`.

**Method.**
- Per-cell frequencies come from regressing peak times on cycle number. Their SE comes from the residual scatter,
  shrunk towards the slice pool.
- Phase prediction uses one minimal Kuramoto model throughout:
  - for TTX, integrated from the measured phases at the midpoint of the condition;
  - for pre and post, at its locked fixed point.
- The synthetic validation feeds known frequencies through the same pipeline.
- The wavelet peaks follow Abel et al.: a level-4 db8 detail (16–32 h band) and plain band maxima. A prominence
  threshold rejected real TTX peaks because the whole-trace band leaks into the TTX edges.

**Epoch method:**
- **Coupled conditions:** the frequency spread equals the SE (pre: spread 0.002–0.014 vs SE 0.006–0.018 c/d; post:
  0.005–0.008 vs 0.004–0.006), so the cells are locked and the spread is measurement error.
- **TTX:**

  | | SCN1 | SCN2 | SCN3 | SCN4 | SCN5 |
  |---|---|---|---|---|---|
  | Spread (c/d) | 0.040 | 0.090 | 0.085 | 0.020 | 0.013 |
  | Median SE (c/d) | 0.031 | 0.051 | 0.047 | 0.024 | 0.018 |

  An intrinsic spread is resolvable in SCN1–3; in SCN4–5 it is indistinguishable from the error.
- **Validation:** the TTX SEs are calibrated in the core (robust z-SD 1.0–1.16; estimated vs true ρ = 0.74; median
  error 0.017 c/d), but there are heavy tails: 1–22 % of cells have |z| > 3, i.e. cycle slips. SCN1's true bimodal
  distribution is recovered.
- **Phase prediction is weak:**
  - In the coupled conditions, frequencies from the same condition explain a little of the locked pattern (diagonal
    0.05–0.33).
  - TTX frequencies explain 0.06–0.10 of the pre and post patterns in SCN1/2.
  - The TTX diagonal is negative everywhere. Frequencies from half of the TTX window (2–3 cycles) are too noisy
    relative to the spread, so this is not evidence for unstable frequencies.

**Wavelet method:**
- The TTX distributions are broader (SCN1 spread 0.094) and distorted. The synthetic test shows spurious mass at
  0.8–0.9 c/d for SCN1.
- The SEs are mis-calibrated in pre and post (robust z-SD up to 3.5): the smoothing correlates the timing errors.
- **The epoch method is the better frequency estimator.**

**Implication for step 2:** the error model needs a heavy-tailed (outlier) component.

## 21. Step 2: deconvolving noise-floor fit, first run (2026-11-25, `scn_deconvolution.py`; outputs in `deconvolution/`)

**Error model**, calibrated on the synthetic validation:

| | Gaussian scale c | Outlier fraction π_o | Outlier scale σ_o (c/d) |
|---|---|---|---|
| Epoch | 0.98 | 0.10 | 0.23 |
| Wavelet | 1.14 | 0.12 | 0.10 |

**Synthetic recovery, epoch method** (true M recovered, naive vs deconvolved):
- Bimodal: 100 % vs 100 %. Smallest width 0.018 vs 0.004, true 0.012: the naive fit overestimates the width and the
  deconvolved fit underestimates it.
- Broad unimodal: 67 % vs 67 %. Width 0.039 vs 0.020, true 0.025.
- Narrow unimodal: 50 % vs 33 %. The deconvolved fit overcounts with zero-width components.

**Synthetic recovery, wavelet method:** the deconvolved fit fails, with D above the floor and spurious zero-width
components. The independent-Gaussian error model is wrong for wavelet peaks, whose timing errors are correlated.

**Data, epoch method:**

| | SCN1 | SCN2 | SCN3 | SCN4 | SCN5 |
|---|---|---|---|---|---|
| Naive M* | 2 | 2 | 2 | 1 | 2 |
| Deconvolved M* | 1 | 2 | 2 | 1 | 2 |
| Deconvolved M* = 1 in bootstrap | 57 % | — | — | — | — |

- **SCN1:** after deconvolution, M* = 1 is preferred (γ = 0.020 c/d), so SCN1's bimodality is MARGINAL given the
  errors.
- **SCN2/3:** the second component sits at 0.56–0.64 c/d (≈ 40 h). That looks like cycle-slip artefacts (a missed
  peak not counted) rather than biology, and the outlier model does not absorb them.
- **SCN4:** deconvolved width 0.003.
- **SCN5:** the deconvolved fit is degenerate (zero-width components).

**Conclusions:**
1. Deconvolving widths smaller than the error is ill-posed. The fit then drives widths to zero and adds components,
   so it needs regularisation, e.g. a width prior or a profile likelihood with intervals.
2. The error model should be calibrated per slice. Slices differ in SNR.
3. Cycle slips must be handled at the source: a missed-peak check, or rejecting cells with an implausible TTX period.
4. Use the wavelet method only with a correlated-error model, or report it naive.

## 22. Cycle-slip handling + naive noise-floor fits with bootstrap (2026-11-25)

**Cycle-slip handling.** Cycles are now counted against a slice-wide reference period, extra peaks are dropped, and
cells with out-of-band periods are excluded. The ≈0.6 c/d modes are gone. Epoch frequencies now agree with the
original phase-slope estimates (per-cell r = 0.65–0.82); wavelet frequencies do not (r = 0.06–0.50). SCN2 (epoch)
loses 57/261 cells, which have out-of-band periods.

**Naive fits, epoch method** (M*, bootstrap share of M*, B = 50):

| Slice | M* (share) | Centres (c/d) | Widths (c/d) | 95 % CI |
|---|---|---|---|---|
| SCN1 | 2 (100 %) | 0.93 / 0.99 | 0.017 / 0.029 | separation 0.053–0.068 |
| SCN2 | 1 (84 %) | 0.984 | 0.037 | width 0.032–0.043 |
| SCN3 | 2 (98 %) | 0.95 / 1.05 | 0.030 / 0.039 | separation 0.08–0.13 |
| SCN4 | 1 (44 %; M = 2 in 56 %) | 0.968 | 0.019 | width 0.016–0.020 |
| SCN5 | 2 (96 %) | 0.961 / 0.983 | 0.006 / 0.012 | separation 0.016–0.032, overlapping (narrow) |

**Recovery, epoch method (naive):**
- Bimodal: M = 2 in 100 %, widths overestimated (0.020–0.023 vs a true 0.012).
- Broad unimodal: M = 1 in 75 %, width 0.030–0.034 vs a true 0.025.
- Narrow unimodal: M = 1 in 58 %, width about 0.015 vs a true 0.004.

So naive widths are upper bounds, inflated by the per-cell error (≈ 0.015 c/d floor). Modality is reliable for
separations ≳ 0.05 c/d, and spurious second modes appear when the distribution is narrow (SCN4/5).

**Wavelet method:** recovery is poor (broad unimodal → M = 2 in 92 %).

## 23. Core vs shell: frequencies by lobe and position, epoch method (2026-11-25, `scn_coreshell_frequency.py`)

Figure: `writeup/scn_core_shell_frequency.{png,pdf,json}`.

**Method.**
- Position = radial rank within the lobe.
- Location test = Mann–Whitney z of the lobe-centred TTX frequencies, shell vs core, scanned over the threshold q*.
- 2000 within-lobe permutations, maximum over thresholds (scan-corrected).

**Shell faster: pooled max z = 7.2 at q* = 0.35, scan-corrected p = 0.0065.**

| | SCN1 | SCN2 | SCN3 | SCN4 | SCN5 |
|---|---|---|---|---|---|
| z at q* | +1.9 | −0.2 | +2.5 | +3.3 | −0.2 |
| Core / shell median (c/d) | 0.962 / 0.976 | — | 0.966 / 1.002 | 0.963 / 0.973 | — |

q* = 0.35 is near the low edge of the scanned grid (0.30–0.70), so the core boundary may be even more central.

**The measurement error does not depend on position:** median SE in core vs shell is 0.026 vs 0.025 (SCN1) and
0.034 vs 0.035 (SCN3). So the shell's larger spread is INTRINSIC. Error-corrected spread, core / shell (c/d):

| SCN1 | SCN2 | SCN3 | SCN4 | SCN5 |
|---|---|---|---|---|
| 0.037 / 0.056 | 0.022 / 0.050 | 0.053 / 0.085 | 0 / 0.012 | 0 / 0.004 |

**Lobes do not differ systematically:** median difference |ΔA−B| ≤ 0.015 c/d; p < 0.05 only in SCN4 (0.033).

**Modality lives in the shell.**

| | Core M* (support) | Shell M* (support) | Shell centres (c/d) |
|---|---|---|---|
| SCN1 | 1 (60 %) | 2 (100 %) | 0.927 / 0.995 |
| SCN3 | 1 (77 %) | 2 (100 %) | 0.949 / 1.053 |
| SCN2 | 1 | 1 | — |
| SCN4 | 1 | 1 | — |
| SCN5 | 1 | 2 (73 %; narrow, overlapping, likely spurious) | — |

**Interpretation:** a slow, homogeneous core (≈ 0.96 c/d) and a faster, broader shell. In the bimodal slices the shell
contains a slow cluster aligned with the core plus a fast cluster (≈ 1.0–1.05 c/d). SCN4/5 are homogeneous throughout.

## 24. Mirror-symmetric core/shell geometry (2026-11-25, `scn_coreshell_geometry.py`; `writeup/scn_core_shell_geometry.*`)

Literature basis: each nucleus has its own ventral(-lateral) core and a dorsomedial shell, and the two nuclei are
mirror images. Hence per-lobe but mirror-symmetric models. Score = 10-fold CV R² of a two-group step function, with
1000 within-lobe permutations.

**10-fold CV R²:**

| | SCN1 | SCN2 | SCN3 | SCN4 | SCN5 | Pooled Σ (p) |
|---|---|---|---|---|---|---|
| Centred radial | −0.01 | −0.02 | 0.00 | 0.02 | 0.01 | 0.006 (0.025) |
| Mirror planar | 0.12 (p .001) | −0.03 (ns) | 0.025 (p .03) | 0.14 (p .001) | 0.03 (p .02) | 0.29 (0.001) |
| Mirror offset radial | 0.08 | — | 0.025 | 0.175 | 0.027 | 0.27 (0.001) |

**The planar model is the parsimonious winner.** The centred radial model explains essentially nothing out of sample.

**What the split finds:** a minority group at one end of the axis, 20–25 % of cells (45 % in SCN5), that is faster by
0.03–0.05 c/d.
- In SCN1, SCN4 and SCN5 this group lies within 14–25° of the axis perpendicular to the inter-lobe line, i.e. the
  dorsoventral axis if the lobes sit side by side. In SCN3 it is at 60°; SCN2 is not significant.
- **In all five slices the fast group has the lower MIC degree:**

  | | SCN1 | SCN2 | SCN3 | SCN4 | SCN5 |
  |---|---|---|---|---|---|
  | Fast group | 5.6 | 1.5 | 0.5 | 3.3 | 2.2 |
  | Rest | 7.3 | 4.1 | 4.2 | 5.4 | 9.3 |

  So the fast group is SHELL-like (sparsely connected) and the rest contains the hubs. This agrees with the shorter
  period of isolated dorsal SCN (Noguchi et al. 2004, via Leise et al.).

**Labelling (fixed):** shell = the faster group, MIC-confirmed. After relabelling the core is bimodal in SCN1 (0.927 / 0.981), SCN3 and SCN4 (spread small),
and the shell is unimodal and fast (SCN1 1.018, SCN3 1.038, SCN4 0.997 c/d).

**Modality:** SCN1 and SCN3 bimodality persists in the slow majority (0.93 / 0.98 and 0.95 / 1.05), so the fast mode
is only partly spatially segregated.

**R² is modest.** Per-cell error variance is about half of the observed variance, so the attainable R² is ≲ 0.5.

## 25. Four-ensemble model, first server fits (2026-11-25, `scn_coreshell4.py`; outputs in `coreshell4/`)

**Setup:** fixed per-ensemble naive mixtures (planar mirror split); literature-constrained coupling; one TTX
multiplier; no noise and no width factors. Models: base (9 parameters) and tr (+ TTX-onset transient, 11).
Total loss: base 37.6e-3, tr 33.9e-3. The starts agree.

**Slice RMSE:**

| | SCN1 | SCN2 | SCN3 | SCN4 | SCN5 |
|---|---|---|---|---|---|
| base | 0.053 | 0.058 | 0.044 | 0.023 | 0.027 |
| tr | 0.037 | 0.036 | 0.040 | 0.039 | 0.022 |
| Model C (with free noise and lobe transient) | 0.019 | 0.021 | 0.024 | 0.026 | — |

The four-ensemble numbers also include core and shell R per lobe.

**What works (base, frequencies alone):**
- **TTX block:** complete in SCN1–3 (f_ttx at its lower bound); 1 % residual in SCN4/5.
- **Shell–core drift under TTX** in SCN1/2/3 (shell drifting ahead to +6 h) is reproduced from the measured
  core/shell frequency difference alone.
- **The post-washout shell lead** of +2–4 h (SCN1/3) is reproduced.
- **The small TTX dips of SCN4/5** follow from their narrow distributions.

**Coupling hierarchy:** K_cc ≫ K_cs. The core→shell coupling is WEAK: 0.013–0.035 rad/h in SCN1–3.
- This is required by the 2–4 h shell lead: φ ≈ Δω / (K_cs R) with Δω ≈ 0.016 rad/h.
- It conflicts with the "dense core→shell projection" of the literature, unless the projection is anatomically dense
  but weak in effective phase coupling.
- K_ss > K_cs in SCN1–3.
- ε is inconsistent: at its bound of 0.3 in SCN3–5, 0 in SCN2, 0.08 in SCN1.

**Problems:**
1. The fast early TTX drop in SCN1/2 cores is missed by base; tr needs g_tr = 0.02–0.03 rad/h with τ = 5–24 h there,
   and none in SCN3.
2. The shell targets are noisy (21–52 cells per lobe), so shell RMSE is 0.04–0.15.
3. SCN4/5 coupling levels are not identifiable (homogeneous): parameters at bounds and a washout spike in SCN4.
4. f_wash sits at its cap (5) in SCN1–3, i.e. a strong rebound.
