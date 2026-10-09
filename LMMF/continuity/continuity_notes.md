# Continuity / convergence of the LMMF reduction — notes for the resubmission

Response to referee report lu21625 (point 1: continuity ρ_M → ρ ⇒ Z_M → Z; point 2: sampling
stability of M; point 3: the Skardal comparison). Numerical checks are in `continuity_check.py`
(experiments A–D) and `continuity_figure.png`.

## 0. Key reformulation: one Riccati equation for all three models

For OA-class coupling (shown here for Kuramoto–Sakaguchi with lag α; the other OA-class G work the
same way), define the local order parameter z(ω,t) = ∫e^{iθ} f(θ|ω,t) dθ. On the OA manifold:

    ∂_t z = iω z + (K/2)(e^{-iα} Z − e^{iα} Z̄ z²),      Z(t) = ∫ z(ω,t) μ(dω).          (R)

Three special cases of (R):

* **True mean field:** μ = ρ.
* **LMMF:** μ = ρ_M. By the residue theorem, Z = Σ_m w_m z(Ω_m + iΔ_m), which gives Eqs. (8)–(9) of
  the manuscript.
* **The finite network itself:** μ = ρ_N = N⁻¹Σδ_{ω_i}, with z(ω_i,0) = e^{iθ_i(0)}. When |z| = 1,
  z² is exactly the second moment, so (R) with z = e^{iθ} *is* the Kuramoto–Sakaguchi equation for
  θ_i.

So the N-oscillator network, the true mean field and the LMMF are the same equation, driven by three
different frequency measures ρ_N, ρ and ρ_M. They also differ in initial data: unit-modulus z for
the network, z ≡ z₀ for the other two. The referee's question becomes a question about how (R)
depends on μ. The code integrates all three models with this single RHS
(`continuity_check.riccati`).

## 1. Finite-time Lipschitz continuity (answers the referee's main point)

**Proposition 1.**

*Assumptions:*
* μ, ν are arbitrary probability measures on ℝ. Heavy tails are allowed, so Lorentzians are
  included.
* (R) is solved with the same initial datum z₀(ω), where |z₀| ≤ 1 and Lip(z₀) ≤ L₀. The OA-manifold
  initial condition z₀ ≡ R₀ used in the paper has L₀ = 0.

*Claim:* for all t ≥ 0,

    |Z_μ(t) − Z_ν(t)| ≤ 2 (1 + L₀ + t) e^{2Kt} d_BL(μ, ν),

where d_BL(μ,ν) = sup{ |∫f d(μ−ν)| : ‖f‖_∞ ≤ 1, Lip f ≤ 1 } is the bounded-Lipschitz
(Fortet–Mourier) distance. It metrizes weak convergence and, unlike W₁, is finite for Cauchy tails.

*Proof sketch (about 10 lines, fits in an appendix):*

1. The unit disc is invariant: d|z|²/dt = K Re(e^{-iα}Z z̄)(1−|z|²). Hence |z|, |Z| ≤ 1.
2. u = ∂_ω z obeys u̇ = iz + iωu − K e^{iα} Z̄ z u. Because iω is skew,
   d|u|/dt ≤ 1 + K|u|, so Lip_ω z(·,t) ≤ L(t) ≤ (L₀ + t) e^{Kt}.
3. Split the error E = Z_μ − Z_ν = ∫ z_μ d(μ−ν) + ∫ (z_μ − z_ν) dν. The first term is
   ≤ 2(1 + L(t)) d_BL by step 2.
4. δ = z_μ − z_ν obeys d|δ|/dt ≤ K|E| + K|δ|, again because the iωδ term is skew. This bound is
   uniform in ω.
5. Combining steps 3 and 4 and applying Gronwall gives the claim. ∎

*Literature context:*
* The same statement on the full (θ, ω) phase space, without the OA assumption and with an
  unspecified constant e^{CT}, is the Neunzert/Dobrushin stability estimate for the kinetic Kuramoto
  equation:
  * Lancellotti, *Transport Theory Stat. Phys.* 34, 523 (2005);
  * Chiba & Medvedev, *DCDS-A* 39, 131 (2019), Thm 2.2 / Eq. (2.27);
  * Kaliuzhnyi-Verbovetskyi & Medvedev, *SIAM J. Math. Anal.* 50, 2441 (2018), Sec. 4, which
    handles distributed ω explicitly.
* These hold for arbitrary initial measures, so they also cover different frequency marginals and
  off-manifold initial phases. They are what to cite. Prop. 1 is the self-contained, OA-native
  version with an explicit constant.

**Link to what the fit actually minimizes (CvM → d_BL).** All three steps are elementary.

* *CvM controls KS.* With D_CvM = ∫(F−G)² dG, we have ‖F−G‖_∞ ≤ 2 D_CvM^{1/3}. In quantile
  coordinates u = G(ω), a sup-gap δ forces |F−G| ≥ δ/2 on a u-interval of length ≥ δ/2.
* *KS controls d_BL.* Integrate by parts on [−A, A] and bound the tails:
  d_BL ≤ 2(1+A)‖F−G‖_∞ + μ(|ω|>A) + ν(|ω|>A).
* *Rate.* The Lorentzian tail mass is ≈ (2/π) Σ w_m Δ_m / A. Optimizing over A gives
  d_BL = O(√(‖F−G‖_∞ · Σ w_m Δ_m)).

Hence CvM → 0 ⇒ KS → 0 ⇒ d_BL → 0 ⇒ Z_M → Z uniformly on compact time intervals. Triangle
inequality: d_BL(ρ, ρ_M) ≤ d_BL(ρ, ρ_N) + d_BL(ρ_N, ρ_M). The first term is O(N^{-1/2}) with high
probability (DKW), so the result holds w.r.t. the *true* ρ as well, which is what the referee asked
for.

**Sharper KS → d_BL for the actual fit (added after the walkthrough).** Use the lighter-tailed
measure for the tail term: d_BL ≤ 2(2+A)‖F−G‖_∞ + 2 min(μ(|ω|>A), ν(|ω|>A)). The fit target ρ_N
has compact support, so with A = max_i|ω_i|:
d_BL(ρ_N, ρ_M) ≤ 2(2 + max_i|ω_i|) KS(F_N, F_M) ≤ 4(2 + max_i|ω_i|) D_CvM^{1/3}.

**Corollary (spectrum of R on the window [0,T]).** Use ||Z|−|Z_M|| ≤ |Z−Z_M|, the reverse
triangle inequality for |·| of Fourier coefficients, and Parseval:
* every finite-window Fourier coefficient of R satisfies ||R̂(ν)| − |R̂_M(ν)|| ≤ sup_{t≤T}|Z−Z_M|;
* the Fig. 1a metric (n samples, n_f = n//2+1 rfft bins) satisfies
  specRMSE ≤ n_f^{-1/2} sup_{t≤T}|Z−Z_M|.

Check on exp. A (T=30, n=300): with the measured sup error this bound is within a factor of 2–9 of
the measured specRMSE. The looseness is in the Gronwall step, not the Fourier step. Errors are
ordered by d_BL, not by M: the M=8 fit has a larger d_BL than M=7 and correspondingly larger errors.

**Numerical check (exp. A, Fig. panel a).** Setup: the Fig. 1 Gaussian mixture, K=3, z₀ = R₀ = 0.88.
The reference is (R) on 20k quantile nodes, converged to 6·10⁻⁴. Fits are fixed-M CvM fits, M=1..10.

* sup_{t≤T}|Z_ρ − Z_M| is ≈ 0.4–0.6 × d_BL for T=1, 2, 5. Log-log slopes are 1.2–1.4, i.e. Lipschitz
  or better.
* The Gronwall constant (e^{2KT}) is wildly pessimistic. The empirical constant is O(1).
* The spectral RMSE of R (the Fig. 1a metric) scales ≈ d_BL².
* Over long windows (T=30) the time-domain sup error is **not** monotone in d_BL (see §3).

## 2. Fourier picture: exact linear response = Cauchy transform of ρ

Linearize (R) about incoherence (z = 0). With κ = K e^{-iα}/2:

    Z(t) = z₀ φ(t) + κ ∫₀ᵗ φ(t−s) Z(s) ds,     φ(t) = ∫ e^{iωt} μ(dω)   (characteristic function)
    Ẑ(λ) = z₀ Φ(λ) / (1 − κ Φ(λ)),             Φ(λ) = ∫ μ(dω)/(λ − iω)  (Cauchy/Stieltjes transform)

Consequences:

1. **Only φ (equivalently Φ) of the distribution enters the linear dynamics.** Fitting ρ_M amounts
   to approximating φ(t) by a positive sum of damped exponentials, φ_M(t) = Σ w_m e^{(iΩ_m − Δ_m)t}.
   This is a Prony / exponential-sum problem. The linearized LMMF eigenvalues are exactly the M roots
   of 1 = κΦ_M(λ), i.e. the LMMF's discrete approximation of the Landau-damping resonances of ρ.
2. **Locality in frequency.** On the line λ = ε + iν:

       Φ(ε+iν) = π[(P_ε * μ)(ν) − i (Q_ε * μ)(ν)],

   where P_ε is the Lorentzian of width ε and Q_ε is its conjugate (Hilbert) kernel. So the spectral
   content of Z at frequency ν is set by the density **smoothed at resolution ε** around ν. An
   observation window T resolves ε ~ 1/T.
3. **Spectral error formula.** With χ = 1/(1 − κΦ) the susceptibility:

       δẐ = z₀ δΦ / [(1 − κΦ)(1 − κΦ_M)] ≈ z₀ χ² δΦ.

   The spectral error is the Lorentzian-smoothed density error times the squared susceptibility,
   which **diverges at the incoherence threshold**.
4. **Finite N is itself an N-component Lorentzian mixture at resolution ε.** This statement is
   exact. In the linear regime, the damped trace Z_N(t)e^{−εt} of the network equals the
   linear-regime trace of an N-component LMMF with Ω_m = ω_i, Δ_m = ε, w_m = 1/N.
   * The LMMF is therefore a model-order reduction N → M of a Lorentzian mixture, not an
     uncontrolled substitute.
   * Sample-specific structure becomes visible once T ≳ N ρ_max: the relative noise of P_ε*ρ_N is
     ~(Nερ)^{-1/2}.
   * This is the mechanism behind the "finite-size bias" captured in Fig. 2.

**Numerical check (exp. B, Fig. panel b).**
* The closed form matches simulation to < 5·10⁻⁴ relative error at K=1 and K=1.6.
* max|χ| is 1.8 at K=1 and 3.7 at K=1.6.
* The relative spectral error of the LMMF is 0.43/0.40/0.18 (M=2/4/8) at K=1, and grows to
  1.27/1.17/0.33 at K=1.6. Low-M fits predict resonances that are far too sharp near threshold.

## 3. Long times, attractors, and why the amplitude spectrum is the right metric

* **Weak closeness cannot control thresholds or attractor type.**
  * K_c and the stability of incoherent or partially locked states depend on pointwise and analytic
    properties of ρ: K_c = 2/(πρ(0)); Landau damping (Chiba 2015, ETDS; Dietert–Fernandez–
    Gérard-Varet 2018, CPAM).
  * Counterexample: Omel'chenko & Wolfrum, PRL 109, 164101 (2012). Small shape changes alter the
    bifurcation scenario in Sakaguchi–Kuramoto.
  * In our formula: continuity of unstable modes with growth rate γ holds with constant ~1/γ², from
    |δΦ(λ)| ≤ 2 max(1/γ, 1/γ²) d_BL on Re λ ≥ γ plus Rouché. That constant blows up as γ → 0.
  * Honest statement: **finite-time continuity in weak topology always; attractor continuity only
    away from bifurcation points (hyperbolicity), with sensitivity ∝ 1/distance-to-bifurcation.**
    The same holds for stationary states through the Kuramoto self-consistency equation. Its
    integrand is bounded and Hölder-½ in ω, and the implicit-function constant is
    1/|1 − ∂_r S|, which diverges at folds.
* **Why |FFT(R)| and not R(t).**
  * On a periodic or quasi-periodic attractor, an O(d) frequency mismatch produces phase drift.
    |R − R_M| then grows linearly in t and saturates at O(amplitude) for any d > 0. This is visible in
    exp. A at T=30 (non-monotone) and exp. C (finite-N errors of 0.2–0.4 over [0,30] even at
    N=5·10⁴).
  * The amplitude spectrum is shift-invariant, and R = |Z| removes the global rotation. Both are
    therefore continuous in *orbital* (not trajectory) distance, which is what persists under
    perturbation of a hyperbolic attractor.
  * This justifies the spectral RMSE in Fig. 1a and should be said explicitly.

## 4. Sampling stability of M (referee point 2)

**Sandwich argument (rigorous, given a global fixed-M optimum).** Let a(M) = inf_{ρ_M} d(ρ, ρ_M) be
the sample-free approximation curve, and ε_N = d(ρ, ρ_N) ≤ √(log(2/η)/2N) w.p. ≥ 1−η (DKW, KS
metric). The sample fit error lies in [a(M) − ε_N, a(M) + ε_N]. So any threshold rule "smallest M
with fit error ≤ τ" selects M̂ ∈ [M₋, M₊], where M± = min{M : a(M) ≤ τ ∓ ε_N}. Both bounds are
deterministic. M̂ is stable w.h.p. whenever a(M) is not flat at τ, and the window shrinks as N grows.
The λ-penalty rule admits the same argument.

**Numerics (exp. D, panels c–d; 20 samples per N; manuscript rule, M_max=16):**

| N | λ=1e-5 | λ=1e-4 |
|---|---|---|
| 500 | 5–8 (mode 6) | 2–7, bimodal 3/6 |
| 2000 | 5–10 (mode 6) | 2–4 (mode 3) |
| 5000 | 6–7 | 2–4 (mode 3) |
| 20000 | 6–8 (mode 6) | 3–4 (mode 3) |

* M̂ is set by λ and is stable across samples. It does **not** grow with N, which a hypothesis-test
  stopping rule would cause.
* The text's "M≈6–8" corresponds to λ=1e-5. At λ=1e-4 (used e.g. for Allen) M≈3.

**Important structural finding (panel d).**
* The best Lorentzian mixture approaches the Gaussian mixture slowly in d_BL: 0.2 (M=2), 0.11 (M=6),
  0.07 (M=9). The N=5000 sample sits at d_BL(ρ, ρ_N) ≈ 0.018.
* The excess is **tail mass**: Lorentzians put O(Σ w_m Δ_m / A) mass beyond |ω| > A.
* Dynamically this is benign. Far-detuned oscillators contribute to Z only with weight ~K/(2|ω−Ω|),
  which is why the empirical Lipschitz constant is ≈ 0.5.
* A sharper statement would use a tail-weighted test-function class
  (|f(ω)| ≤ min(1, c/|ω|)). Worth trying if a tighter constant is needed.

## 5. Skardal comparison (referee point 3)

* Both reductions are OA reductions for **rational** densities. Skardal uses the n poles of g_n,
  with complex residues. LMMF uses M fitted poles, with positive residues w_m.
* The referee is right that "LMMF beats exact Skardal" mixes a sample-informed with an uninformed
  model. Section 2.4 gives the precise reason: in a window T the network sees P_{1/T} * ρ_N, not g_n,
  and |φ_N − φ| ~ N^{-1/2} is amplified by |χ|².
* Fairer comparisons for the revision:
  1. Fit the LMMF to g_n itself (or a 10⁶ sample) and compare with Skardal at matched dimension:
     a(M) vs n.
  2. Out-of-sample: fit on sample A, simulate a network with an independent sample B.
  3. Report both against the N → ∞ reference.

## 6. Suggested changes for the revision

1. **Main text:** add Prop. 1 (one paragraph plus an end-matter proof) and the CvM → KS → d_BL chain.
   Cite Lancellotti 2005 and Chiba–Medvedev 2019 for the general kinetic version. State the
   finite-time / weak-topology scope and the bifurcation caveat (Omel'chenko–Wolfrum 2012;
   Dietert et al. 2018).
2. **Fourier section:** the linear-response formula, the χ² amplification and the N-Lorentzian
   interpretation of a finite network. One display equation and a figure panel like panel b.
3. **Initial conditions:** initialize micro phases from the wrapped Cauchy with parameter R₀. This
   puts the N → ∞ initial condition on the OA manifold, so Prop. 1 applies verbatim.
   * The current wrapped Gaussian is off-manifold.
   * Exp. C shows this makes little practical difference (errors within seed noise).
   * Cite Engelbrecht & Mirollo 2020 (PRR 2, 023057) on OA attraction as a caveat.
4. **Sampling study:** add a panel like (c) to address point 2, plus the sandwich argument. Choose M
   by the dynamical tolerance relative to the sampling level ε_N, not by a GoF test (whose power
   grows with N).
5. **Optional loss:** consider a loss closer to the dynamics, e.g. a weighted L² error of φ on [0, T]
   or of Φ on Re λ = 1/T. By Plancherel, ∫|F−G|² dω = (2π)⁻¹ ∫|φ_F − φ_G|²/t² dt, so an
   unweighted-dω CDF L² loss already *is* a characteristic-function loss with weight 1/t².
6. **Reframe Skardal** as in §5.

## 7. Tested alternative loss (negative result, 2026-10-08)

`spectral_loss_prototype.py` fits ρ_M by minimizing the closed-form Cauchy-transform loss
L_ε = ∫|Φ_N(ε+iν) − Φ_M(ε+iν)|² dν = 2π∫₀^∞|φ_N − φ_M|² e^{−2εs} ds, the linear-response quantity
of §2. It is equivalent to an MMD with a Cauchy kernel of width 2ε. The closed form was checked
against numerical integration (1e-11).

Results (Fig. 1 system, N=5000, 2 seeds, M=2/4/6, error vs MF(ρ) on [0,5] and spectral RMSE on [0,30]):
* ε = 0.05, 0.2: worse than CvM at M=2 (sup5 0.51–0.57 vs 0.18), although L_ε itself is lower.
  The incoherence linear response is not what governs the K=3 nonlinear dynamics.
* ε = 1: much better at M=2 (sup5 0.08 vs 0.15–0.18, spectral error up to 4× lower), mixed at M=4–6.
* ε = 0.5: better at M=6, partly because the CvM M=6 fits collapse to 4 effective components.
* There is no ε that wins consistently, and d_BL does not rank these fits (it is an upper bound only).
⇒ Keep CvM as the primary loss. Do not claim the spectral loss as an improvement.
