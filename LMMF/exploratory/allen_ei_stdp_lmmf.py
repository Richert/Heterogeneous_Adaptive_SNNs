#!/usr/bin/env python3
r"""
Allen-constrained E/I QIF network with plastic e→e synapses vs. its LMMF mean field
====================================================================================

An excitatory/inhibitory network of conductance-based quadratic integrate-and-fire (QIF)
neurons whose *single-cell* parameters and *excitability distribution* are taken from the
Allen Cell Types Database, together with the Lorentzian-Mixture Mean Field (LMMF) reduction
of the same network. The e→e synapses are plastic (a soft-bounded STDP rule); everything
else is static. The script runs both models for a SYMMETRIC and an ASYMMETRIC STDP rule and
compares firing-rate dynamics and final e→e coupling weights.

Units throughout: ms, mV, pA, pF, nS  (so k is in pF/(ms·mV) = nS/mV and rates are in
spikes/ms = kHz; rates are converted to Hz only for plotting).

1. Allen data  (``allensdk``, cached ``cell_types/manifest.json``)
------------------------------------------------------------------
For every mouse cortical cell classified as a pyramidal cell (PC: ``dendrite_type ==
"spiny"``) or a PV+ fast-spiking interneuron (FSI: ``Pvalb`` transgenic line) we load

    V_R,i = ``vrest``                          resting membrane potential      [mV]
    V_T,i = ``threshold_v_long_square``        spike threshold                 [mV]
    τ_i   = ``tau``                            membrane time constant          [ms]
    R_i   = ``input_resistance_mohm``          input resistance                [MΩ]

The whole-cell membrane capacitance is not a stored feature, so it is reconstructed from the
two quantities that determine it, C_i = τ_i / R_i (1000·ms/MΩ = pF):

    C_i = 1000 · τ_i / R_i                                                     [pF]

2. Neuron-specific excitability η_i
-----------------------------------
Near rest the QIF membrane equation C_i V̇ = k_i (V−V_R,i)(V−V_T,i) linearises to
C_i V̇ ≈ −k_i ΔV_i (V−V_R,i) with ΔV_i = V_T,i − V_R,i, i.e. the resting membrane conductance
is g_i = k_i ΔV_i = C_i/τ_i. This fixes the quadratic coefficient of each cell,

    k_i = C_i / (τ_i ΔV_i) = 1000 / (R_i ΔV_i)                                 [nS/mV]

With a constant current I the quadratic k_i(V−V_R,i)(V−V_T,i) + I has real fixed points only
for I below the value of the parabola's minimum (attained at V = (V_R,i+V_T,i)/2). The
*current required to elicit a spike starting from rest* — the QIF rheobase — is therefore

    I_rheo,i = k_i ΔV_i² / 4 = C_i ΔV_i / (4 τ_i)                              [pA]

which is the neuron-specific quantity the network model must preserve. The network, however,
uses POPULATION-AVERAGE membrane parameters C_e = ⟨C_i⟩_e, k_e, V_R,e, V_T,e (as in the model
equations below), whose own rheobase is η_rheo,x = k_x (V_T,x − V_R,x)²/4. Absorbing all
single-cell variability into the additive current then gives the excitability

    η_i = η_rheo,x − I_rheo,i          (x = e for PCs, x = i for FSIs)         [pA]

so that a neuron embedded in the homogeneous-parameter population needs exactly its own
measured rheobase to fire: η_i > 0 ⇒ intrinsically more excitable than the average cell
(tonically firing at rest), η_i < 0 ⇒ needs η_rheo,x − η_i of extra current. This is the
form of heterogeneity the Ott–Antonsen/MPR reduction can absorb exactly (additive current
heterogeneity), which is what makes step 4 possible.

3. Network model (microscopic)
------------------------------
N_e excitatory PCs (j = 1…N_e) and N_i inhibitory FSIs, conductance-based synapses with
reversal potentials E_e (excitatory) and E_i (inhibitory):

    C_e V̇_j = k_e (V_j−V_R,e)(V_j−V_T,e) + η_j + I_e(t) + g_e s_{e,e},j (E_e−V_j)
                                                        + g_i s_{e,i},j (E_i−V_j)
    C_i V̇_j = k_i (V_j−V_R,i)(V_j−V_T,i) + η_j + I_i(t) + g_e s_{i,e},j (E_e−V_j)
                                                        + g_i s_{i,i},j (E_i−V_j)

The drive of population x is a constant plus a COLORED-NOISE fluctuation, I_x(t) = I_x + ξ_x(t),
where ξ_x is an Ornstein–Uhlenbeck process with zero mean, a PER-POPULATION stationary s.d.
σ_x (``noise_amp_e`` / ``noise_amp_i``, either of which may be 0) and a shared
autocorrelation time τ_ξ (``tau_noise``):

    τ_ξ ξ̇_x = −ξ_x + √(2 τ_ξ) σ_x η(t) ,   ⟨ξ_x(t) ξ_x(t+s)⟩ = σ_x² e^{−|s|/τ_ξ}

The two populations get independent realisations, but ξ_x is COMMON to every neuron within a
population: a shared input fluctuation is what a mean field can follow, whereas per-neuron
noise would have to enter the Ott–Antonsen reduction as extra heterogeneity. The network and
the mean field are driven by the SAME realisation (generated once on a grid resolving τ_ξ and
linearly interpolated by both), so their traces can be compared cycle by cycle rather than
only in distribution.

Every neuron carries ONE synaptic activation s_j — the alpha-kernel convolution of its own
spike train, a cascade of two first-order filters with the time constant of its cell type
(τ_s,e for PCs, τ_s,i for FSIs; unit DC gain, so ⟨s_j⟩ = r_j):

    ȧ_j = −a_j/τ_s ,  ṡ_j = (a_j − s_j)/τ_s ,   a_j → a_j + 1/τ_s  on each spike

The combined inputs s_{x,y} to a neuron in population x from population y are the
weight-averaged presynaptic activations,

    s_{e,e},i = (1/N_e) Σ_j A_ij s_j^e      (PLASTIC weights A_ij ∈ [0,1])
    s_{e,i},i = (w_ei/N_i) Σ_j s_j^i ,  s_{i,e},i = (w_ie/N_e) Σ_j s_j^e ,
    s_{i,i},i = (w_ii/N_i) Σ_j s_j^i        (static, all-to-all)

4. Plasticity (e→e only)
------------------------
    τ_A Ȧ_ij = (1 − A_ij) P_ij − A_ij D_ij

with the soft bounds keeping A_ij in [0,1] and P_ij / D_ij the LTP / LTD drivers built from
two per-neuron traces — a fast potentiation trace u^p (τ_p) and a slow depression trace u^d
(τ_d), each low-pass filtering the neuron's own SYNAPTIC ACTIVATION s_j (rather than being
kicked by its spikes). Since the alpha synapse has unit DC gain, ⟨s_j⟩ = r_j, the traces have
the same steady state ⟨u^p_j⟩ = τ_p r_j as spike-driven ones, but they are continuous and are
written identically in the network and in the mean field:

    u̇^p_j = s_j − u^p_j/τ_p ,   u̇^d_j = s_j − u^d_j/τ_d

    asymmetric (causal, pair-based Hebbian STDP)
        P_ij = a_p r_i u^p_j     LTP when the PRE trace is high at a POST spike (pre→post)
        D_ij = a_d u^d_i r_j     LTD when the POST trace is high at a PRE spike (post→pre)

    symmetric (coincidence detection; τ_p < τ_d ⇒ Mexican-hat in Δt)
        P_ij = a_p u^p_i u^p_j   LTP for coincidence within τ_p
        D_ij = a_d u^d_i u^d_j   LTD for coincidence within τ_d

(These are the ``stdp_asym`` / ``stdp_sym`` wirings of ``config/fre_equations.yaml``'s
``stdp_op`` with b = 1.) Both P and D are RANK-1 outer products of per-neuron vectors, which
is what makes the N_e×N_e update affordable and what carries over to the mean field.

NOTE on what drives the final weight structure. The fixed point of the weight equation is
A*_ij = P̄_ij/(P̄_ij + D̄_ij). If the two populations fired as independent Poisson processes,
every driver above would factorise into ⟨r_i⟩⟨r_j⟩ times a constant, and A* would be the
SAME number for every synapse. All structure in the final weight matrix therefore comes from
spike-timing correlations, i.e. from the network's collective (oscillatory) dynamics — which
is exactly the part the mean field can reproduce, and the reason the default parameters put
the network in a PING-like oscillatory regime.

5. Mean field (LMMF)
--------------------
A Lorentzian mixture ρ(η) ≈ Σ_m w_m L(Ω_m, Δ_m) is fitted to the EMPIRICAL η distribution of
each population separately (``shared/lorentzian_mixture.py``, Cramér–von Mises, greedy M),
and each component becomes an Ott–Antonsen/MPR ensemble. With the total synaptic conductance
of ensemble m, g_tot,m = g_e s_{x,e},m + g_i s_{x,i},m (its contribution to the coefficient
of v is −g_tot,m, hence the minus sign in the ṙ equation):

    C_x ṙ_m = k_x Δ_m/(π C_x) + r_m [ k_x (2 v_m − V_R,x − V_T,x) − g_tot,m ]
    C_x v̇_m = k_x (v_m−V_R,x)(v_m−V_T,x) − (π C_x r_m)²/k_x + Ω_m + I_x(t)
              + g_e s_{x,e},m (E_e−v_m) + g_i s_{x,i},m (E_i−v_m)
    τ_s ȧ_m = r_m − a_m ,   τ_s ṡ_m = a_m − s_m
    u̇^p_m  = s_m − u^p_m/τ_p ,   u̇^d_m = s_m − u^d_m/τ_d          (e-ensembles only)

The plastic weights reduce to an M_e×M_e block matrix Ā_mn (the ensemble average of A_ij over
i ∈ m, j ∈ n), obeying the SAME equation with ensemble-level drivers,

    τ_A Ā̇_mn = (1 − Ā_mn) P_mn − Ā_mn D_mn ,
    P_mn = a_p r_m u^p_n  |  a_p u^p_m u^p_n ,   D_mn = a_d u^d_m r_n  |  a_d u^d_m u^d_n

and the e→e drive closes on the ensemble-weighted synaptic activations

    s_{e,e},m = Σ_n w_n Ā_mn s_n^e ,   s_{e,i},m = w_ei s̄^i ,  s̄^y = Σ_n w_n^y s_n^y , …

i.e. the reduction replaces A_ij by its ensemble averages and drops within-ensemble
correlations (which vanish ∝ 1/√N).

The M_e×M_e mean-field weights are compared with the microscopic N_e×N_e matrix through the
soft ensemble responsibilities P_im = w_m L(η_i; Ω_m, Δ_m) / Σ_l w_l L(η_i; Ω_l, Δ_l), in
both directions:

  * BLOCK RMSE — coarse-grain the network's A_ij onto the ensembles, Ā^net_mn = Σ_ij P_im P_jn
    A_ij / (Σ_i P_im)(Σ_j P_jn), and compare with Ā_mn. This is the apples-to-apples test of
    the closure: Ā^net_mn is exactly the quantity Ā_mn is a model of.
  * FULL RMSE — extrapolate the mean field back to the neurons, Ã = P Ā Pᵀ, and compare with
    A_ij. This additionally asks the M ensembles to resolve the neuron-level weight spread,
    so it is bounded below by the within-block variance the reduction discards by construction.

With the defaults (M_e = 8) the symmetric rule gives block/full RMSE ≈ 0.01 / 0.04 (2% / 9%
of ⟨A⟩) and the asymmetric rule ≈ 0.07 / 0.19 (16% / 45%): the mean field reproduces the
symmetric weight structure almost exactly, and captures the causal (antisymmetric) block
structure of the asymmetric rule while smoothing the near-saturated microscopic matrix.
(Driving the traces with s_j rather than with spike kicks roughly halves both asymmetric-rule
errors, because the drivers are then continuous and the network and mean field low-pass the
same signal.)

Implementation note
-------------------
Unlike the companion scripts (``qif_conductance_meanfield.py``,
``qif_structured_coupling_lmmf.py``) this model is NOT built with PyRates: the coupling
weights are state variables (an N_e×N_e / M_e×M_e plastic matrix), which PyRates'
constant-weight ``Connectivity`` edges cannot express, and the per-edge ``EdgeTemplate``
route used in ``qif_mpmf_stdp_simulation.py`` does not scale past a few hundred edges. Both
models are therefore written directly as numba-compiled vector fields — the micro network
with forward Euler (spike reset + trace kicks in the loop, plastic weights on a slower
clock), the mean field with ``scipy.integrate.solve_ivp``.

Minimal Python environment (the ``allen`` conda env this was developed in)
--------------------------------------------------------------------------
    Python >= 3.10, allensdk >= 2.16, numba >= 0.57, numpy >= 1.23, scipy >= 1.10,
    pandas >= 1.5, matplotlib >= 3.6      (+ repo-local ``shared/lorentzian_mixture.py``)

Run:
    PATH="$HOME/conda/envs/allen/bin:$PATH" python allen_ei_stdp_lmmf.py
    ... --rules sym asym --n-e 400 --n-i 100 --T 3000 --fig-only
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..", "shared")]
# ---------------------------------------------------------------------------
import argparse
import os
import sys
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import solve_ivp

import lorentzian_mixture as LM          # CvM Lorentzian-mixture fitter

OUT = os.path.join(_HERE, "allen_ei_stdp_lmmf")
CELL_CACHE = os.path.join(_HERE, "allen_ei_stdp_lmmf_cells.npz")
MANIFEST = os.path.join(_HERE, "..", "data_fitting", "cell_types", "manifest.json")

RULES = ("sym", "asym")


# ════════════════════════════════════════════════════════════════════════════
#  configuration
# ════════════════════════════════════════════════════════════════════════════
P = dict(
    # ── synapses (reversal potentials, peak conductances, kinetics) ─────────
    E_e=0.0, E_i=-70.0,          # synaptic reversal potentials [mV]
    g_e=160.0, g_i=300.0,        # peak conductances of the e / i synapse [nS]
    tau_s_e=4.0, tau_s_i=8.0,    # alpha-synapse time constants (AMPA / GABA_A) [ms]
    # static, all-to-all weights of the three non-plastic projections
    w_ei=1.0, w_ie=1.0, w_ii=1.0,
    # ── plasticity (e→e) ────────────────────────────────────────────────────
    A0=0.5,                      # initial e→e weight (all synapses)
    tau_A=1000.0,                 # weight time constant [ms]
    tau_p=10.0, tau_d=40.0,      # LTP / LTD trace time constants [ms]
    # LTP/LTD gains per rule. The symmetric drivers are products of two traces
    # (~(τ_x r)²) whereas the asymmetric ones are a rate × a trace (~τ_p r²), so the
    # two rules need gains differing by ≈ τ_p to act on comparable time scales. The
    # values below give P̄ ≈ D̄ ≈ 0.5 at the operating-point rate (≈12 Hz), i.e. the
    # weights relax over ~τ_A/(P̄+D̄) ≈ 1 s and settle near A ≈ 0.5.
    stdp=dict(sym=dict(a_p=70.0, a_d=10.0),
              asym=dict(a_p=50.0, a_d=20.0)),
    dt_A=0.5,                    # plastic-weight update step (slow clock) [ms]. Now that
                                 # the traces are driven by the continuous s_j, the drivers
                                 # are smooth and dt_A barely matters (asym block-RMSE at
                                 # dt_A = 1/0.5/0.2 → 0.047/0.046/0.048); only the asym
                                 # rule's per-neuron rate estimate cnt/dt_A still samples
                                 # spike times, so keep it well below τ_p.
    # ── external drive: tonic I_x plus a common colored-noise fluctuation ξ_x(t) ──
    # ξ_x is an Ornstein–Uhlenbeck process, τ_ξ ξ̇ = −ξ + √(2τ_ξ) σ_x η(t), i.e. zero mean,
    # stationary s.d. σ_x (noise_amp_e / noise_amp_i, set PER POPULATION) and a shared
    # autocorrelation time τ_ξ = tau_noise. One INDEPENDENT process per population, shared
    # by every neuron in it: a common input fluctuation is what the mean field can follow
    # (per-neuron noise would instead have to enter the OA reduction as extra
    # heterogeneity). The SAME realisation drives the network and the mean field, so the
    # two traces can be compared cycle by cycle.
    I_e=260.0, I_i=120.0, noise_amp_e=1.0, noise_amp_i=1.0, tau_noise=200.0,
    # ── protocol / integration ──────────────────────────────────────────────
    T=2500.0, t_plast_on=500.0,  # plasticity switched on after the network has settled
    dt=2e-3, dts=1.0,            # Euler step / recording step [ms]
    N_e=400, N_i=100,
    v_peak=500.0, v_reset=-500.0,
    seed=1,
    # ── LMMF fit of the empirical η-distributions ───────────────────────────
    # a generous lower Δ bound keeps the ensembles broad: very narrow Lorentzians give
    # weakly damped OA ensembles that ring where the finite-N network does not. The
    # penalty λ controls M and matters a lot for the WEIGHT comparison (the rate dynamics
    # are already well reproduced at M=2): λ=3e-4 → M_e=2, only the coarsest block
    # structure survives; λ=1e-4 → M_e=3; λ≤1e-5 → M_e=6…8, which resolves the
    # η-dependence of the final weights while keeping the mean field ~90-dimensional.
    fit_delta_lo=0.02,            # Δ_min as a fraction of the η standard deviation
    fit_delta_hi=0.2,            # Δ_max as a fraction of the η standard deviation
    fit_M_max=20, fit_alpha=1e-3, fit_lambda=1e-6, fit_patience=2,
    fit_restarts=10, fit_method="slsqp",
    # ── Allen data selection ────────────────────────────────────────────────
    eta_clip=(0.5, 99.5),        # percentile window kept (drops ephys outliers)
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rules", nargs="+", default=list(RULES), choices=list(RULES))
    ap.add_argument("--n-e", type=int, default=P["N_e"])
    ap.add_argument("--n-i", type=int, default=P["N_i"])
    ap.add_argument("--T", type=float, default=P["T"])
    ap.add_argument("--dt", type=float, default=P["dt"])
    ap.add_argument("--seed", type=int, default=P["seed"])
    ap.add_argument("--g-e", type=float, default=P["g_e"])
    ap.add_argument("--g-i", type=float, default=P["g_i"])
    ap.add_argument("--I-e", type=float, default=P["I_e"])
    ap.add_argument("--noise-amp-e", type=float, default=P["noise_amp_e"],
                    help="s.d. of the colored input noise to the PCs [pA] (0 disables it)")
    ap.add_argument("--noise-amp-i", type=float, default=P["noise_amp_i"],
                    help="s.d. of the colored input noise to the FSIs [pA] (0 disables it)")
    ap.add_argument("--tau-noise", type=float, default=P["tau_noise"],
                    help="autocorrelation time of the colored input noise [ms]")
    ap.add_argument("--refresh-cells", action="store_true",
                    help="re-query the AllenSDK instead of using the cached cell table")
    ap.add_argument("--fig-only", action="store_true",
                    help="re-make the figure from the saved .npz (no simulation)")
    a = ap.parse_args()
    P.update(N_e=a.n_e, N_i=a.n_i, T=a.T, dt=a.dt, seed=a.seed,
             g_e=a.g_e, g_i=a.g_i, I_e=a.I_e,
             noise_amp_e=a.noise_amp_e, noise_amp_i=a.noise_amp_i,
             tau_noise=a.tau_noise)
    return a


# ════════════════════════════════════════════════════════════════════════════
#  figure style — matched to the neuromodulation research proposal (project_1.svg)
# ════════════════════════════════════════════════════════════════════════════
# The proposal figures are composed in Inkscape from full-page-width panels embedded 1:1
# (matplotlib pt == SVG pt), so the figure's own width in points IS its width on the page.
# Matching the reference panel script (DynamicalSystems/BasalGanglia/STR/
# plotting_spn_rate_distributions.py) and the panel labels measured in project_1.svg:
#   * page/figure width  989.276 pt
#   * default sans-serif family (project_1.svg uses 'Sans'; matplotlib's DejaVu Sans),
#     NOT the STIX serif of the PRL manuscript figures
#   * base font size 16 pt, lines.linewidth 2.0
#   * panel labels: BARE capitals A, B, C… (no parentheses), bold sans, 24 pt
# _SC is the geometric scale from the original 7-inch PRL layout: enlarging the figure and
# every font/linewidth by the same factor reproduces that layout at proposal size.
FIG_WIDTH_PT = 989.27631
BASE_FONTSIZE = 16.0
PANEL_FONTSIZE = 24.0
_SC = (FIG_WIDTH_PT / 72.0) / 7.0            # ≈1.96


def set_proposal_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": BASE_FONTSIZE, "axes.titlesize": BASE_FONTSIZE,
        # panel TITLES are bold and centred on the axes; the panel LABELS A, B, … stay
        # bold but sit outside, above the top-left corner (see panel()).
        "axes.titleweight": "bold", "axes.titlelocation": "center",
        # ticks at the base size, as in the reference script; legends a touch smaller
        # (that script also shrinks its legends relative to the axis labels)
        "legend.fontsize": 0.875 * BASE_FONTSIZE,
        "xtick.labelsize": BASE_FONTSIZE, "ytick.labelsize": BASE_FONTSIZE,
        "axes.linewidth": 0.6 * _SC, "lines.linewidth": 2.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 100,
    })


def panel(ax, letter):
    """Bold bare-capital panel label outside the axes, above its top-left corner."""
    ax.annotate(letter.upper(), xy=(0, 1), xycoords="axes fraction", xytext=(-34, 10),
                textcoords="offset points", fontsize=PANEL_FONTSIZE, fontweight="bold",
                ha="left", va="bottom")


# ════════════════════════════════════════════════════════════════════════════
#  (1) AllenSDK: V_R, V_T, C for cortical PCs and PV+ FSIs
# ════════════════════════════════════════════════════════════════════════════
def _cell_class(line, dendrite_type):
    """PC = spiny (excitatory) cell, FSI = PV+ (Pvalb) interneuron."""
    line = str(line)
    if "Pvalb" in line:
        return "FSI"
    if dendrite_type == "spiny":
        return "PC"
    return None


def load_allen_cells(refresh=False):
    """Return {'PC': dict, 'FSI': dict} of per-cell arrays V_R, V_T, C, tau, R_in.

    Cached to ``allen_ei_stdp_lmmf_cells.npz`` so the (slow) AllenSDK query runs once.
    """
    if os.path.exists(CELL_CACHE) and not refresh:
        d = np.load(CELL_CACHE)
        return {c: {k: d[f"{c}_{k}"] for k in ("v_r", "v_th", "C", "tau", "R_in")}
                for c in ("PC", "FSI")}

    import pandas as pd
    from allensdk.core.cell_types_cache import CellTypesCache
    from allensdk.api.queries.cell_types_api import CellTypesApi

    ctc = CellTypesCache(manifest_file=MANIFEST)
    cells = pd.DataFrame(ctc.get_cells(species=[CellTypesApi.MOUSE]))
    ephys = pd.DataFrame(ctc.get_ephys_features())
    df = cells.merge(ephys, left_on="id", right_on="specimen_id", how="inner")
    df = df.dropna(subset=["vrest", "threshold_v_long_square", "tau", "input_resistance_mohm"])
    df["cls"] = [_cell_class(l, d) for l, d in zip(df["transgenic_line"], df["dendrite_type"])]

    out, save = {}, {}
    for c in ("PC", "FSI"):
        d = df[df["cls"] == c]
        v_r = d["vrest"].to_numpy(float)
        v_th = d["threshold_v_long_square"].to_numpy(float)
        tau = d["tau"].to_numpy(float)
        R_in = d["input_resistance_mohm"].to_numpy(float)
        keep = (v_th - v_r) > 1.0                       # drop degenerate/misfit cells
        v_r, v_th, tau, R_in = v_r[keep], v_th[keep], tau[keep], R_in[keep]
        C = 1000.0 * tau / R_in                          # whole-cell capacitance [pF]
        out[c] = dict(v_r=v_r, v_th=v_th, C=C, tau=tau, R_in=R_in)
        save.update({f"{c}_{k}": v for k, v in out[c].items()})
    np.savez(CELL_CACHE, **save)
    print(f"[saved] {CELL_CACHE}")
    return out


# ════════════════════════════════════════════════════════════════════════════
#  (2) neuron-specific excitability η_i
# ════════════════════════════════════════════════════════════════════════════
def derive_excitability(cell, clip=(0.5, 99.5)):
    """Population-average QIF parameters + the per-cell excitability η_i [pA].

    k_i = C_i/(τ_i ΔV_i) is the quadratic coefficient reproducing the cell's resting
    conductance C_i/τ_i; I_rheo,i = k_i ΔV_i²/4 is the current needed to make the QIF
    fixed points collide, i.e. to elicit a spike from rest. The network runs on the
    population-average membrane (C, k, V_R, V_T), whose rheobase is η_rheo, so the
    single-cell variability is carried entirely by η_i = η_rheo − I_rheo,i.
    """
    v_r, v_th, C, tau = cell["v_r"], cell["v_th"], cell["C"], cell["tau"]
    dv = v_th - v_r
    k = C / (tau * dv)                                   # per-cell quadratic coefficient
    I_rheo = k * dv ** 2 / 4.0                           # per-cell QIF rheobase [pA]

    pop = dict(C=float(C.mean()), k=float(k.mean()),
               v_r=float(v_r.mean()), v_th=float(v_th.mean()))
    pop["eta_rheo"] = pop["k"] * (pop["v_th"] - pop["v_r"]) ** 2 / 4.0
    eta = pop["eta_rheo"] - I_rheo

    lo, hi = np.percentile(eta, clip)                    # drop ephys outliers in the tails
    eta = eta[(eta >= lo) & (eta <= hi)]
    return pop, eta, k, I_rheo


def make_noise(P, rng):
    """One Ornstein–Uhlenbeck trace per population, τ_ξ ξ̇_x = −ξ_x + √(2τ_ξ) σ_x η(t).

    The two populations have SEPARATE amplitudes σ_e = ``noise_amp_e``, σ_i =
    ``noise_amp_i`` (either may be 0 to drive only the other population) and share the
    autocorrelation time τ_ξ. Sampled on a grid fine enough to resolve τ_ξ (and no finer),
    using the EXACT discrete-time update ξ_{k+1} = ξ_k e^{−Δ/τ_ξ} + σ_x √(1−e^{−2Δ/τ_ξ}) z_k,
    so the stationary s.d. is exactly σ_x regardless of the grid. Both models then linearly
    interpolate the SAME traces, which decouples the realisation from each model's own time
    step. Returns (dt_n, t_n, ξ_e, ξ_i).
    """
    tau = P["tau_noise"]
    dt_n = min(P["dts"], tau / 20.0)
    n = int(round(P["T"] / dt_n)) + 2
    t_n = np.arange(n) * dt_n
    a = np.exp(-dt_n / tau)
    out = []
    for sigma in (P["noise_amp_e"], P["noise_amp_i"]):   # independent process per population
        z = rng.standard_normal(n)                       # drawn even if σ=0, so that the
        xi = np.zeros(n)                                 # other population's realisation
        if sigma > 0.0:                                  # does not depend on this σ
            b = sigma * np.sqrt(1.0 - a * a)
            xi[0] = sigma * z[0]                         # start from the stationary distribution
            for k in range(1, n):
                xi[k] = a * xi[k - 1] + b * z[k]
        out.append(xi)
    return dt_n, t_n, out[0], out[1]


def sample_eta(eta, N, rng):
    """Draw N excitabilities from the empirical η pool (bootstrap, sorted)."""
    return np.sort(rng.choice(eta, size=N, replace=True))


# ════════════════════════════════════════════════════════════════════════════
#  (4) LMMF fit of an empirical η distribution
# ════════════════════════════════════════════════════════════════════════════

def fit_lmmf(eta, P, seed=0):
    sd = float(np.std(eta))
    bounds = (P["fit_delta_lo"] * sd, P["fit_delta_hi"] * sd)
    res = LM.fit(eta, bounds, M_max=P["fit_M_max"], alpha=P["fit_alpha"],
                 lambda_M=P["fit_lambda"], patience=P["fit_patience"],
                 loss="cvm", n_restarts=P["fit_restarts"], seed=seed,
                 method=P["fit_method"])
    m = res["model"]
    w = np.asarray(m.w, float)
    return dict(w=w / w.sum(), Omega=np.asarray(m.Omega, float),
                Delta=np.asarray(m.Delta, float), M=int(m.M),
                loss=float(res["data_loss"]), delta_bounds=bounds)


def responsibilities(eta, fit):
    """Soft ensemble membership P_im ∝ w_m L(η_i; Ω_m, Δ_m), rows summing to 1."""
    w, Om, De = fit["w"], fit["Omega"], fit["Delta"]
    L = w[None, :] * (De[None, :] / np.pi) / ((eta[:, None] - Om[None, :]) ** 2 + De[None, :] ** 2)
    return L / L.sum(axis=1, keepdims=True)


def block_average(A, Pi):
    """Coarse-grain a microscopic N×N weight matrix onto the M×M ensemble blocks,
    Ā_mn = Σ_ij P_im P_jn A_ij / (Σ_i P_im)(Σ_j P_jn) — the quantity the LMMF's Ā_mn
    is a model OF, so comparing the two is the apples-to-apples test of the closure
    (as opposed to P Ā Pᵀ, which additionally asks the M ensembles to resolve the
    neuron-level weight spread)."""
    return (Pi.T @ A @ Pi) / np.outer(Pi.sum(axis=0), Pi.sum(axis=0))


# ════════════════════════════════════════════════════════════════════════════
#  (3) microscopic network — numba forward Euler
# ════════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _micro_loop(V_e, a_e, s_e, up, ud, eta_e, V_i, a_i, s_i, eta_i, A,
                C_e, k_e, vr_e, vt_e, C_i, k_i, vr_i, vt_i,
                E_e, E_i, g_e, g_i, w_ei, w_ie, w_ii, tau_se, tau_si,
                tau_p, tau_d, tau_A, a_p, a_d, sym,
                I_e, I_i, xi_e, xi_i, dt_n, t_plast_on,
                v_peak, v_reset, dt, n_steps, sr, n_A):
    """Forward-Euler stepping of the spiking E/I network with plastic e→e weights.

    The membrane/synapse/trace states are advanced every step (spike reset and the synapse
    kick a += 1/τ_s handled in the loop); the N_e×N_e weight matrix is advanced on the
    slower clock dt_A = n_A·dt, using per-neuron spike counts accumulated over that window
    as the instantaneous rate. Both LTP and LTD drivers are rank-1 (outer products of
    per-neuron vectors), so the update is a single pass over A. The common colored-noise
    drive ξ_x is linearly interpolated from its own grid (step dt_n).
    """
    Ne = V_e.shape[0]
    Ni = V_i.shape[0]
    dt_A = n_A * dt
    n_save = n_steps // sr + 1

    t_rec = np.empty(n_save)
    re_rec = np.empty(n_save)
    ri_rec = np.empty(n_save)
    Abar_rec = np.empty(n_save)
    ve_rec = np.empty(n_save)
    vi_rec = np.empty(n_save)

    cnt = np.zeros(Ne)                    # per-neuron spikes in the plasticity window
    acc_e = 0.0
    acc_i = 0.0
    pp = np.empty(Ne)
    pq = np.empty(Ne)
    dp = np.empty(Ne)
    dq = np.empty(Ne)
    ss = 0

    for step in range(n_steps):
        t = step * dt

        # ── synaptic drives ────────────────────────────────────────────────
        see = np.dot(A, s_e) / Ne                          # (1/N_e) Σ_j A_ij s_j
        sbar_e = 0.0
        for j in range(Ne):
            sbar_e += s_e[j]
        sbar_e /= Ne
        sbar_i = 0.0
        for j in range(Ni):
            sbar_i += s_i[j]
        sbar_i /= Ni
        sei = w_ei * sbar_i
        sie = w_ie * sbar_e
        sii = w_ii * sbar_i

        # common colored-noise drive, linearly interpolated on its own grid
        q = t / dt_n
        kq = int(q)
        fq = q - kq
        Ie = I_e + (1.0 - fq) * xi_e[kq] + fq * xi_e[kq + 1]
        Ii = I_i + (1.0 - fq) * xi_i[kq] + fq * xi_i[kq + 1]

        # ── excitatory population ──────────────────────────────────────────
        nsp = 0
        for j in range(Ne):
            v = V_e[j]
            ge = g_e * see[j]
            gi = g_i * sei
            dv = (k_e * (v - vr_e) * (v - vt_e) + eta_e[j] + Ie
                  + ge * (E_e - v) + gi * (E_i - v)) / C_e
            da = -a_e[j] / tau_se
            ds = (a_e[j] - s_e[j]) / tau_se
            dup = s_e[j] - up[j] / tau_p           # traces are driven by the neuron's own
            dud = s_e[j] - ud[j] / tau_d           # synaptic activation, not by spike kicks
            V_e[j] = v + dt * dv
            a_e[j] += dt * da
            s_e[j] += dt * ds
            up[j] += dt * dup
            ud[j] += dt * dud
            if V_e[j] >= v_peak:
                V_e[j] = v_reset
                a_e[j] += 1.0 / tau_se
                cnt[j] += 1.0
                nsp += 1
        acc_e += nsp

        # ── inhibitory population ──────────────────────────────────────────
        nsp = 0
        for j in range(Ni):
            v = V_i[j]
            ge = g_e * sie
            gi = g_i * sii
            dv = (k_i * (v - vr_i) * (v - vt_i) + eta_i[j] + Ii
                  + ge * (E_e - v) + gi * (E_i - v)) / C_i
            da = -a_i[j] / tau_si
            ds = (a_i[j] - s_i[j]) / tau_si
            V_i[j] = v + dt * dv
            a_i[j] += dt * da
            s_i[j] += dt * ds
            if V_i[j] >= v_peak:
                V_i[j] = v_reset
                a_i[j] += 1.0 / tau_si
                nsp += 1
        acc_i += nsp

        # ── plastic e→e weights (slow clock) ───────────────────────────────
        if (step + 1) % n_A == 0:
            if t >= t_plast_on:
                if sym == 1:
                    for j in range(Ne):
                        pp[j] = a_p * up[j]
                        pq[j] = up[j]
                        dp[j] = ud[j]
                        dq[j] = a_d * ud[j]
                else:
                    for j in range(Ne):
                        rj = cnt[j] / dt_A
                        pp[j] = a_p * rj
                        pq[j] = up[j]
                        dp[j] = ud[j]
                        dq[j] = a_d * rj
                c = dt_A / tau_A
                for i in range(Ne):
                    for j in range(Ne):
                        A[i, j] += c * ((1.0 - A[i, j]) * pp[i] * pq[j]
                                        - A[i, j] * dp[i] * dq[j])
            for j in range(Ne):
                cnt[j] = 0.0

        # ── recording ──────────────────────────────────────────────────────
        if step % sr == 0:
            Asum = 0.0
            for i in range(Ne):
                for j in range(Ne):
                    Asum += A[i, j]
            t_rec[ss] = t
            re_rec[ss] = acc_e / (Ne * sr * dt)
            ri_rec[ss] = acc_i / (Ni * sr * dt)
            Abar_rec[ss] = Asum / (Ne * Ne)
            ve_rec[ss] = np.median(V_e)
            vi_rec[ss] = np.median(V_i)
            acc_e = 0.0
            acc_i = 0.0
            ss += 1

    return (t_rec[:ss], re_rec[:ss], ri_rec[:ss], Abar_rec[:ss], ve_rec[:ss], vi_rec[:ss])


def run_micro(P, pops, eta_e, eta_i, rule, rng, noise):
    dt_n, _, xi_e, xi_i = noise
    e, i = pops["PC"], pops["FSI"]
    Ne, Ni = eta_e.size, eta_i.size
    V_e = e["v_r"] + rng.random(Ne) * (e["v_th"] - e["v_r"])
    V_i = i["v_r"] + rng.random(Ni) * (i["v_th"] - i["v_r"])
    A = np.full((Ne, Ne), P["A0"])
    st = P["stdp"][rule]

    out = _micro_loop(
        V_e, np.zeros(Ne), np.zeros(Ne), np.zeros(Ne), np.zeros(Ne), eta_e,
        V_i, np.zeros(Ni), np.zeros(Ni), eta_i, A,
        e["C"], e["k"], e["v_r"], e["v_th"], i["C"], i["k"], i["v_r"], i["v_th"],
        P["E_e"], P["E_i"], P["g_e"], P["g_i"], P["w_ei"], P["w_ie"], P["w_ii"],
        P["tau_s_e"], P["tau_s_i"], P["tau_p"], P["tau_d"], P["tau_A"],
        st["a_p"], st["a_d"], 1 if rule == "sym" else 0,
        P["I_e"], P["I_i"], xi_e, xi_i, dt_n, P["t_plast_on"],
        P["v_peak"], P["v_reset"], P["dt"], int(round(P["T"] / P["dt"])),
        max(1, int(round(P["dts"] / P["dt"]))), max(1, int(round(P["dt_A"] / P["dt"]))))
    t, r_e, r_i, Abar, v_e, v_i = out
    return dict(t=t, r_e=r_e, r_i=r_i, Abar=Abar, v_e=v_e, v_i=v_i, A=A)


# ════════════════════════════════════════════════════════════════════════════
#  (5) LMMF mean field — numba vector field + solve_ivp
# ════════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _mf_rhs(y, dy, we, Om_e, De_e, wi, Om_i, De_i,
            C_e, k_e, vr_e, vt_e, C_i, k_i, vr_i, vt_i,
            E_e, E_i, g_e, g_i, w_ei, w_ie, w_ii, tau_se, tau_si,
            tau_p, tau_d, tau_A, a_p, a_d, sym, Ie, Ii, plastic):
    """Vector field of the LMMF equations.

    Layout of y (Me e-ensembles, Mi i-ensembles):
        [r_e, v_e, a_e, s_e, u^p, u^d | r_i, v_i, a_i, s_i | Ā (Me·Me, row-major)]
    """
    Me = we.shape[0]
    Mi = wi.shape[0]
    o_re, o_ve, o_ae, o_se, o_up, o_ud = 0, Me, 2 * Me, 3 * Me, 4 * Me, 5 * Me
    o_ri = 6 * Me
    o_vi, o_ai, o_si = o_ri + Mi, o_ri + 2 * Mi, o_ri + 3 * Mi
    o_A = o_ri + 4 * Mi

    # ensemble-weighted synaptic activations
    sbar_e = 0.0
    for n in range(Me):
        sbar_e += we[n] * y[o_se + n]
    sbar_i = 0.0
    for n in range(Mi):
        sbar_i += wi[n] * y[o_si + n]
    sei = w_ei * sbar_i
    sie = w_ie * sbar_e
    sii = w_ii * sbar_i

    # ── e-ensembles ────────────────────────────────────────────────────────
    for m in range(Me):
        see = 0.0
        for n in range(Me):
            see += y[o_A + m * Me + n] * we[n] * y[o_se + n]
        r = y[o_re + m]
        v = y[o_ve + m]
        ge = g_e * see
        gi = g_i * sei
        dy[o_re + m] = (k_e * De_e[m] / (np.pi * C_e)
                        + r * (k_e * (2.0 * v - vr_e - vt_e) - ge - gi)) / C_e
        dy[o_ve + m] = (k_e * (v - vr_e) * (v - vt_e) - (np.pi * C_e * r) ** 2 / k_e
                        + Om_e[m] + Ie + ge * (E_e - v) + gi * (E_i - v)) / C_e
        dy[o_ae + m] = (r - y[o_ae + m]) / tau_se
        dy[o_se + m] = (y[o_ae + m] - y[o_se + m]) / tau_se
        dy[o_up + m] = y[o_se + m] - y[o_up + m] / tau_p     # driven by s_m, as in the network
        dy[o_ud + m] = y[o_se + m] - y[o_ud + m] / tau_d

    # ── i-ensembles ────────────────────────────────────────────────────────
    for m in range(Mi):
        r = y[o_ri + m]
        v = y[o_vi + m]
        ge = g_e * sie
        gi = g_i * sii
        dy[o_ri + m] = (k_i * De_i[m] / (np.pi * C_i)
                        + r * (k_i * (2.0 * v - vr_i - vt_i) - ge - gi)) / C_i
        dy[o_vi + m] = (k_i * (v - vr_i) * (v - vt_i) - (np.pi * C_i * r) ** 2 / k_i
                        + Om_i[m] + Ii + ge * (E_e - v) + gi * (E_i - v)) / C_i
        dy[o_ai + m] = (r - y[o_ai + m]) / tau_si
        dy[o_si + m] = (y[o_ai + m] - y[o_si + m]) / tau_si

    # ── plastic block weights Ā_mn ─────────────────────────────────────────
    for m in range(Me):
        for n in range(Me):
            idx = o_A + m * Me + n
            if plastic == 1:
                if sym == 1:
                    p = a_p * y[o_up + m] * y[o_up + n]
                    d = a_d * y[o_ud + m] * y[o_ud + n]
                else:
                    p = a_p * y[o_re + m] * y[o_up + n]
                    d = a_d * y[o_ud + m] * y[o_re + n]
                A = y[idx]
                dy[idx] = ((1.0 - A) * p - A * d) / tau_A
            else:
                dy[idx] = 0.0
    return dy


def run_mf(P, pops, fit_e, fit_i, rule, noise):
    _, t_n, xi_e, xi_i = noise
    e, i = pops["PC"], pops["FSI"]
    Me, Mi = fit_e["M"], fit_i["M"]
    n = 6 * Me + 4 * Mi + Me * Me
    o_re, o_ve, o_se = 0, Me, 3 * Me
    o_ri, o_vi, o_si = 6 * Me, 6 * Me + Mi, 6 * Me + 3 * Mi
    o_A = 6 * Me + 4 * Mi

    y0 = np.zeros(n)
    y0[o_ve:o_ve + Me] = e["v_r"]
    y0[o_vi:o_vi + Mi] = i["v_r"]
    y0[o_A:] = P["A0"]
    st = P["stdp"][rule]
    sym = 1 if rule == "sym" else 0

    base = (fit_e["w"], fit_e["Omega"], fit_e["Delta"],
            fit_i["w"], fit_i["Omega"], fit_i["Delta"],
            e["C"], e["k"], e["v_r"], e["v_th"], i["C"], i["k"], i["v_r"], i["v_th"],
            P["E_e"], P["E_i"], P["g_e"], P["g_i"], P["w_ei"], P["w_ie"], P["w_ii"],
            P["tau_s_e"], P["tau_s_i"], P["tau_p"], P["tau_d"], P["tau_A"],
            st["a_p"], st["a_d"], sym)
    dy = np.zeros(n)

    # integrate piecewise around the plasticity switch (the only discontinuity left);
    # the colored-noise drive is continuous and is interpolated inside the RHS. max_step
    # is capped so the adaptive stepper cannot stride over noise fluctuations.
    max_step = min(P["tau_s_e"], P["tau_noise"] / 4.0)
    breaks = sorted({0.0, P["t_plast_on"], P["T"]})
    breaks = [b for b in breaks if 0.0 <= b <= P["T"]]
    ts, ys = [], []
    y = y0
    for t0, t1 in zip(breaks[:-1], breaks[1:]):
        plastic = 1 if 0.5 * (t0 + t1) >= P["t_plast_on"] else 0

        def rhs(t, yy, _p=plastic):
            Ie = P["I_e"] + np.interp(t, t_n, xi_e)
            Ii = P["I_i"] + np.interp(t, t_n, xi_i)
            return _mf_rhs(yy, dy, *base, Ie, Ii, _p).copy()

        t_eval = np.arange(t0, t1, P["dts"])
        sol = solve_ivp(rhs, (t0, t1), y, method="LSODA", t_eval=t_eval,
                        rtol=1e-7, atol=1e-9, max_step=max_step)
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1]

    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    we, wi = fit_e["w"], fit_i["w"]
    Abar_t = Y[o_A:].reshape(Me, Me, -1).mean(axis=(0, 1))
    return dict(t=t, r_e=we @ Y[o_re:o_re + Me], r_i=wi @ Y[o_ri:o_ri + Mi],
                v_e=we @ Y[o_ve:o_ve + Me], v_i=wi @ Y[o_vi:o_vi + Mi],
                Abar=Abar_t, A=Y[o_A:, -1].reshape(Me, Me),
                r_e_ens=Y[o_re:o_re + Me], s_e_ens=Y[o_se:o_se + Me])


# ════════════════════════════════════════════════════════════════════════════
#  STDP kernels
# ════════════════════════════════════════════════════════════════════════════
def stdp_kernels(P, t_max=150.0, dt=0.05):
    """Weight change of an isolated pre/post spike pair vs. the lag Δt = t_post − t_pre.

    Each spike deposits one alpha kernel α(t) = (t/τ_s²)e^{−t/τ_s} (unit area) on the
    emitting neuron's synaptic activation, and the traces low-pass that, so a single spike
    leaves the trace K_x(t) = (α ∗ e^{−·/τ_x})(t), x ∈ {p, d}. Integrating the drivers over
    one pair and evaluating τ_A Ȧ = (1−A)P − A D at A = A_0 gives

      asymmetric   W(Δt) = (1−A_0) a_p K_p(Δt)          Δt > 0   (pre→post ⇒ LTP)
                         = −A_0    a_d K_d(−Δt)         Δt < 0   (post→pre ⇒ LTD)
      symmetric    W(Δt) = (1−A_0) a_p C_p(Δt) − A_0 a_d C_d(Δt)
                   with C_x(Δt) = ∫ K_x(t) K_x(t−Δt) dt the trace autocorrelation

    i.e. a causal two-lobed kernel for the asymmetric rule and an even difference-of-widths
    (Mexican-hat) kernel for the symmetric one. Returns (lags, {rule: W}).
    """
    t = np.arange(0.0, t_max, dt)
    alpha = t / P["tau_s_e"] ** 2 * np.exp(-t / P["tau_s_e"])
    K = {x: np.convolve(alpha, np.exp(-t / P[f"tau_{x}"]))[:t.size] * dt for x in ("p", "d")}

    A0 = P["A0"]
    lags = np.concatenate((-t[:0:-1], t))
    out = {}
    for rule in RULES:
        a_p, a_d = P["stdp"][rule]["a_p"], P["stdp"][rule]["a_d"]
        if rule == "asym":
            W = np.where(lags >= 0.0,
                         (1.0 - A0) * a_p * np.interp(np.abs(lags), t, K["p"]),
                         -A0 * a_d * np.interp(np.abs(lags), t, K["d"]))
        else:
            C = {x: np.correlate(K[x], K[x], mode="full") * dt for x in K}
            W = (1.0 - A0) * a_p * C["p"] - A0 * a_d * C["d"]
        out[rule] = W
    return lags, out


# ════════════════════════════════════════════════════════════════════════════
#  figure
# ════════════════════════════════════════════════════════════════════════════
def make_figure(d, out):
    set_proposal_style()
    C_MICRO, C_MF, C_E, C_I, C_HIST, C_MIX = "0.25", "#c1121f", "#2e6f95", "#e07a5f", "0.8", "#2e6f95"
    rules = [str(r) for r in d["rules"]]
    nr = len(rules)

    # NESTED gridspecs, one per row (3 | nr | 2·nr panels). A single flat grid with mixed
    # column spans makes constrained_layout give up ("axes sizes collapsed to zero");
    # nested subgridspecs lay each row out independently and are robust.
    # row 2's ratio should be just big enough for the square panels: any surplus shows up
    # as whitespace between rows 1 and 2 (set_box_aspect shrinks the axes inside the slot).
    fig = plt.figure(figsize=(7.0 * _SC, 5.3 * _SC), layout="constrained")
    rows = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.85, 1.05])
    gs0 = rows[0].subgridspec(1, 3)
    gs1 = rows[1].subgridspec(1, nr)
    gs2 = rows[2].subgridspec(1, 2 * nr)
    letters = iter("abcdefghijklmnopqrstuvwxyz")

    # ── (a) η distributions + LMMF fits: ONE panel label and ONE y-axis label for the
    #        two cell classes, which share the y axis (both are densities in pA⁻¹) ──
    ax_eta = None
    for c, (tag, lbl) in enumerate((("e", "PC (excitatory)"), ("i", "FSI (PV+, inhibitory)"))):
        ax = fig.add_subplot(gs0[0, c], sharey=ax_eta)
        eta = d[f"eta_{tag}_pool"]
        w, Om, De = d[f"w_{tag}"], d[f"Omega_{tag}"], d[f"Delta_{tag}"]
        gx = np.linspace(eta.min(), eta.max(), 600)
        ax.hist(eta, bins=40, density=True, color=C_HIST, label=f"Allen (n={eta.size})")
        comps = w[None, :] * (De[None, :] / np.pi) / ((gx[:, None] - Om[None, :]) ** 2 + De[None, :] ** 2)
        for m in range(w.size):
            ax.plot(gx, comps[:, m], lw=0.6 * _SC, color=C_MIX, alpha=0.6)
        ax.plot(gx, comps.sum(axis=1), lw=1.4 * _SC, color="#c1121f", label=f"LMMF (M={w.size})")
        ax.set_xlabel(r"excitability $\eta$ (pA)")
        ax.set_title(lbl)
        ax.legend(loc="upper left")
        if c == 0:
            ax.set_ylabel("density")
            panel(ax, next(letters))
            ax_eta = ax
        else:
            ax.tick_params(labelleft=False)

    # ── (b) the two STDP kernels ───────────────────────────────────────────
    ax = fig.add_subplot(gs0[0, 2])
    lags, kern = stdp_kernels(P)
    ax.axhline(0.0, color="0.7", lw=0.5 * _SC)
    ax.axvline(0.0, color="0.7", lw=0.5 * _SC)
    for rule, col in (("sym", C_E), ("asym", C_MF)):
        W = kern[rule]
        ax.plot(lags, W / np.abs(W).max(), color=col, lw=1.2 * _SC,
                label="symmetric" if rule == "sym" else "asymmetric")
    ax.set_xlabel(r"$\Delta t = t_{\rm post}-t_{\rm pre}$ (ms)")
    ax.set_ylabel(r"$\Delta A$ / max$|\Delta A|$")
    ax.set_xlim(-100, 100)
    ax.set_title("STDP kernels")
    ax.legend(loc="upper left")
    panel(ax, next(letters))

    # zoom window for the rate traces: the network runs a ~12 Hz PING rhythm for seconds,
    # so the full trace is an unreadable forest of cycles — show the last ~10 cycles, by
    # which time the weights have settled.
    T = float(d["T"])
    zoom = (max(0.0, T - 1000.0), T)

    # ── (c) PC firing rate, one panel per rule: ONE panel label and ONE y-axis label,
    #        the two panels sharing the y axis so the rules are directly comparable ──
    ax_rate = None
    ymax = max(dd.item()["r_e"][(dd.item()["t"] >= zoom[0]) & (dd.item()["t"] <= zoom[1])].max()
               for dd in (d[f"micro_{r}"] for r in rules)) * 1e3
    for c, rule in enumerate(rules):
        name = "symmetric STDP" if rule == "sym" else "asymmetric STDP"
        m, f = d[f"micro_{rule}"].item(), d[f"mf_{rule}"].item()
        ax = fig.add_subplot(gs1[0, c], sharey=ax_rate)
        ax.plot(m["t"], m["r_e"] * 1e3, color=C_MICRO, lw=0.9 * _SC, label="QIF network")
        ax.plot(f["t"], f["r_e"] * 1e3, color=C_MF, lw=1.1 * _SC, ls="--", label="LMMF")
        ax.set_xlabel("time (ms)")
        ax.set_xlim(*zoom)
        ax.set_ylim(0, 1.1 * ymax)
        ax.set_title(f"{name} — PC rate")
        if c == 0:
            ax.set_ylabel(r"PC rate $r_e$ (Hz)")
            ax.legend(ncol=2, loc="upper left")
            panel(ax, next(letters))
            ax_rate = ax
        else:
            ax.tick_params(labelleft=False)

    # ── (d…g) final weight matrices AT LMMF ENSEMBLE RESOLUTION: the network's A_ij
    #          block-averaged onto the M_e ensembles vs. the mean field's own Ā_mn ──
    for c, rule in enumerate(rules):
        f = d[f"mf_{rule}"].item()
        A_blk, A_mf = d[f"A_blk_{rule}"], f["A"]
        ticks = np.arange(d["Omega_e"].size)
        lbl = [f"{o:.0f}" for o in d["Omega_e"]]
        vmin = min(A_blk.min(), A_mf.min())
        vmax = max(A_blk.max(), A_mf.max())
        for kk, (Amat, ttl) in enumerate(((A_blk, "QIF network (block-avg.)"),
                                          (A_mf, r"LMMF $\bar{A}_{mn}$"))):
            ax = fig.add_subplot(gs2[0, 2 * c + kk])
            im = ax.imshow(Amat, origin="lower", aspect="auto", cmap="magma",
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_box_aspect(1.0)         # square panels; set_box_aspect keeps
            #                                constrained_layout happy where aspect="equal"
            #                                collapses the row
            ax.set_title(ttl, fontsize=0.875 * BASE_FONTSIZE)
            ax.set_xlabel(r"pre $\bar{\eta}_n$ (pA)")
            ax.set_xticks(ticks, lbl, rotation=90)
            ax.set_yticks(ticks, lbl if kk == 0 else [])
            if kk == 0:
                ax.set_ylabel(r"post $\bar{\eta}_m$ (pA)")
            if c == 0 and kk == 0:
                panel(ax, next(letters))   # one label for the whole matrix row
            if kk == 1:
                # colorbar as an inset of the RIGHT matrix: axes-fraction coordinates
                # follow the (square) aspect-adjusted box, so it is exactly as tall as
                # the matrix — fig.colorbar(ax=...) would instead span the taller slot.
                cax = ax.inset_axes([1.06, 0.0, 0.07, 1.0])
                fig.colorbar(im, cax=cax)     # unlabelled; the block RMSE is reported in
                #                               the run log and stored in the .npz

    # SVG as well: the proposal figures are assembled from SVG panels in Inkscape.
    # NB no bbox_inches="tight" for the SVG, so the file keeps the exact figure size
    # (989.276 pt wide) and drops into project_*.svg at 1:1 without rescaling.
    fig.savefig(out + ".svg", format="svg")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    print(f"[saved] {out}.svg / .pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    if args.fig_only:
        make_figure(np.load(OUT + ".npz", allow_pickle=True), OUT)
        return

    rng = np.random.default_rng(P["seed"])

    # ── (1)+(2) Allen data → population parameters + excitabilities ────────
    cells = load_allen_cells(refresh=args.refresh_cells)
    pops, etas = {}, {}
    for tag, cls in (("PC", "PC"), ("FSI", "FSI")):
        pop, eta, k_cell, I_rheo = derive_excitability(cells[cls], P["eta_clip"])
        pops[tag] = pop
        etas[tag] = eta
        print(f"{cls}: n={eta.size}  C={pop['C']:.1f} pF  k={pop['k']:.3f} nS/mV  "
              f"V_R={pop['v_r']:.1f} mV  V_T={pop['v_th']:.1f} mV  "
              f"η_rheo={pop['eta_rheo']:.1f} pA")
        print(f"      η: mean={eta.mean():+.1f}  sd={eta.std():.1f}  "
              f"range=[{eta.min():+.1f},{eta.max():+.1f}] pA  "
              f"({100 * np.mean(eta > 0):.0f}% supra-rheobase)")

    eta_e = sample_eta(etas["PC"], P["N_e"], rng)
    eta_i = sample_eta(etas["FSI"], P["N_i"], rng)

    # ── (4) LMMF fits ──────────────────────────────────────────────────────
    fit_e = fit_lmmf(etas["PC"], P, seed=P["seed"])
    fit_i = fit_lmmf(etas["FSI"], P, seed=P["seed"])
    for nm, f in (("PC", fit_e), ("FSI", fit_i)):
        print(f"LMMF {nm}: M={f['M']}  D={f['loss']:.2e}  "
              f"Ω={np.round(f['Omega'], 1)}  Δ={np.round(f['Delta'], 1)}  "
              f"w={np.round(f['w'], 3)}")
    Pi_e = responsibilities(eta_e, fit_e)                # N_e × M_e responsibilities

    # ── colored-noise drive: ONE realisation, shared by both models and both rules ──
    noise = make_noise(P, np.random.default_rng(P["seed"] + 1))
    dt_n, t_n, xi_e, xi_i = noise
    print(f"input noise: σ_e={P['noise_amp_e']:.1f} pA, σ_i={P['noise_amp_i']:.1f} pA, "
          f"τ={P['tau_noise']:.1f} ms "
          f"(grid {dt_n:g} ms; realised s.d. {xi_e.std():.1f}/{xi_i.std():.1f} pA)")

    # ── (3)+(5) simulate both models for each rule ─────────────────────────
    store = dict(rules=np.array(args.rules), eta_e=eta_e, eta_i=eta_i,
                 eta_e_pool=etas["PC"], eta_i_pool=etas["FSI"],
                 w_e=fit_e["w"], Omega_e=fit_e["Omega"], Delta_e=fit_e["Delta"],
                 w_i=fit_i["w"], Omega_i=fit_i["Omega"], Delta_i=fit_i["Delta"],
                 t_noise=t_n, xi_e=xi_e, xi_i=xi_i,
                 **{f"pop_{p}_{k}": v for p, dd in pops.items() for k, v in dd.items()},
                 **{k: float(P[k]) for k in ("T", "dt", "dts", "t_plast_on", "g_e", "g_i",
                                             "I_e", "I_i", "noise_amp_e", "noise_amp_i",
                                             "tau_noise",
                                             "A0", "tau_A", "tau_p", "tau_d")})
    for rule in args.rules:
        t0 = perf_counter()
        print(f"[micro/{rule}] N_e={P['N_e']}, N_i={P['N_i']}, T={P['T']} ms ...")
        mic = run_micro(P, pops, eta_e, eta_i, rule, np.random.default_rng(P["seed"]), noise)
        print(f"   done in {perf_counter() - t0:.1f}s   "
              f"⟨r_e⟩={mic['r_e'].mean() * 1e3:.1f} Hz  ⟨r_i⟩={mic['r_i'].mean() * 1e3:.1f} Hz  "
              f"⟨A⟩_final={mic['A'].mean():.3f}")

        t0 = perf_counter()
        print(f"[LMMF/{rule}] M_e={fit_e['M']}, M_i={fit_i['M']} ...")
        mf = run_mf(P, pops, fit_e, fit_i, rule, noise)
        print(f"   done in {perf_counter() - t0:.1f}s   "
              f"⟨r_e⟩={mf['r_e'].mean() * 1e3:.1f} Hz  ⟨r_i⟩={mf['r_i'].mean() * 1e3:.1f} Hz  "
              f"⟨A⟩_final={mf['A'].mean():.3f}")

        A_rec = Pi_e @ mf["A"] @ Pi_e.T                   # Ā_mn extrapolated to neurons
        A_blk = block_average(mic["A"], Pi_e)             # micro coarse-grained to M_e×M_e
        err = np.sqrt(np.mean((mic["A"] - A_rec) ** 2))
        err_blk = np.sqrt(np.mean((A_blk - mf["A"]) ** 2))
        Am = max(mic["A"].mean(), 1e-12)
        print(f"   weight RMSE: ensemble blocks {err_blk:.4f} ({100 * err_blk / Am:.1f}% of ⟨A⟩)"
              f" | full N×N reconstruction {err:.4f} ({100 * err / Am:.1f}%)")
        store[f"micro_{rule}"] = mic
        store[f"mf_{rule}"] = mf
        store[f"A_rec_{rule}"] = A_rec
        store[f"A_blk_{rule}"] = A_blk
        store[f"A_rmse_{rule}"] = err
        store[f"A_rmse_blk_{rule}"] = err_blk

    np.savez(OUT + ".npz", **store)
    print(f"[saved] {OUT}.npz")
    make_figure(np.load(OUT + ".npz", allow_pickle=True), OUT)


if __name__ == "__main__":
    main()
