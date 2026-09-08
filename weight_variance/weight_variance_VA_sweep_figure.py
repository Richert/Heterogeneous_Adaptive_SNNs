r"""
Adaptive-coupling Kuramoto: self-contained mean-field vs micro — RMSE of R, Ā, V_A vs Δ (per rule)
=================================================================================================

Loads the OA-manifold-initialized microscopic sweep written by
``weight_variance_rule_micro_oainit.py`` (rule × μ × Δ, with R(t), Ā(t), V_A(t) traces) and compares
it to the FULLY self-contained mean-field model, integrated from matching initial conditions (the
OA-manifold micro IC), for the two weight-variance closures

    C_A = C_S + C_F   (full)    vs.    C_A = C_S   (reduced, C_F dropped).

The self-contained mean-field system (Eqs. 7, 9 + the variance closure) is

    Ṙ  = −ΔR + (KĀ/2) R(1−R²),          Ā̇  = μ R² G(0) + γ(1−Ā)   [G(0)=1 cos, 0 sin],
    Ċ_S = −γ C_S + μ σ_S²,   Ċ_F = −(γ+2Δ) C_F + μ σ_F²,   V̇_A = 2μ C_A − 2γ V_A,
    σ_S² = ½(S²−R⁴),  σ_F² = ½(1−S²),  S = ⟨|c|²⟩  (tabulated, off-manifold-safe; b = KĀR),

with C_A = C_S+C_F (full) or C_S (reduced). The variance is slaved to R, Ā (no feedback), so BOTH
models share identical R(t), Ā(t) — the model distinction only affects V_A(t).

The weight variance is reported either as V_A itself (``variance="abs"``, the default) or as the
RELATIVE weight variance V_A/Ā² (``variance="rel"``, ``--variance rel``); in the latter case the
mean-field V_A is divided by its OWN Ā(t), and the output stem gains a ``_rel`` suffix so both
variants can be kept side by side. The choice applies to the traces AND the RMSE panels.

For each (rule, μ, Δ) the time-domain RMSE between the microscopic and mean-field R(t) and the
chosen variance quantity is computed and NORMALIZED by the time average of the microscopic quantity
it is compared against,

    NRMSE[X] = RMSE[X_mf, X_micro] / ⟨X_micro⟩_t ,

so the two rows are dimensionless relative errors and remain comparable across μ, Δ and rules (V_A
itself varies by orders of magnitude over the sweep). The R RMSE uses a FINITE-SIZE-CORRECTED mean
field, R_fs = √(R² + (1−R²)/N), since the sample coherence obeys E[R̂²] = R² + (1−R²)/N — i.e. even a
truly asynchronous state (R=0) reads R̂ ≈ 1/√N in a finite network; R_fs adds exactly that O(1/√N)
coherence floor so the RMSE reflects genuine discrepancy rather than the finite-size artifact.

Figures: ONE PER RULE (``..._cos`` = symmetric, ``..._sin`` = antisymmetric adaptation), each a
two-column PRL figure with 4 ROWS × 4 COLUMNS:

  * row 1 — two wide panels: NRMSE[R] and NRMSE[variance] vs Δ, one line per swept μ (colour) and,
    for the variance, one line style per closure. Dotted verticals mark the Δ of the trace rows.
  * rows 2-4 — example traces at three configurable Δ (rows, ``trace_deltas``) and two configurable
    μ (columns, ``trace_mus``): columns 1-2 hold R(t) at μ₁, μ₂ and columns 3-4 the variance at the
    same two μ. Each trace panel shows the microscopic run (thick, pale grey) against the mean field
    (blue solid = C_S+C_F, red dashed = C_S).

    PATH="$HOME/conda/envs/pycobi/bin:$PATH" python weight_variance_VA_sweep_figure.py [--variance rel]
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from weight_variance_analysis import set_prl_style                    # noqa: E402
from prl_style import COL_DOUBLE, panel_label                         # two-column PRL width
from weight_variance_meanfield import order_parameter_S               # tabulated S = ⟨|c|²⟩ ≤ 1

CONFIG = dict(
    micro_npz=dp.mpmf("weight_variance_rule_micro_oainit.npz"),
    cmap="viridis",
    variance="abs",                 # weight-variance quantity: "abs" -> V_A, "rel" -> V_A/Ā²
    trace_mus=[0.01, 0.04],         # μ of the two trace COLUMNS (snapped to the swept grid)
    trace_deltas=[0.25, 0.5, 0.75],  # Δ of the three trace ROWS (snapped to the swept grid)
    out=dp.mpmf("weight_variance_VA_rmse"),   # "_<rule>" (+ "_rel") is appended
)

V_STYLE = {"full": "-", "cs": "--"}
V_LABEL = {"full": r"$C_A{=}C_S{+}C_F$", "cs": r"$C_A{=}C_S$"}
#: the two weight-variance quantities: TeX symbol + trace-panel y-label
VAR_TEX = {"abs": r"V_A", "rel": r"V_A/\bar A^2"}
VAR_YLAB = {"abs": r"variance $V_A$", "rel": r"rel. variance $V_A/\bar A^2$"}
VAR_NAME = {"abs": "V_A", "rel": "V_A/Abar^2"}          # plain text, for the console log
C_MICRO, C_MF, C_CS = "0.45", "#1f77b4", "#c1121f"   # micro / mean field (full) / mean field (C_S)
RULE_TEX = {"cos": r"\cos", "sin": r"\sin"}


def variance_quantity(VA, Abar, mode):
    """V_A (mode 'abs') or the relative weight variance V_A/Ā² (mode 'rel'), elementwise in time."""
    if mode == "abs":
        return np.asarray(VA, float)
    if mode == "rel":
        A = np.asarray(Abar, float)
        return np.asarray(VA, float) / np.where(A != 0.0, A, np.nan) ** 2
    raise ValueError(f"unknown variance mode {mode!r} (use 'abs' or 'rel')")


# ════════════════════════════════════════════════════════════════════════════
#  self-contained mean-field model (integrated, both closures at once)
# ════════════════════════════════════════════════════════════════════════════
def mf_sim(rule, t, R0, A0, V0, K, gamma, mu, delta):
    """Integrate the self-contained mean field from matching IC (R0, Ā0, V_A0 from the micro; C_S/C_F
    at their steady values from R0, Ā0). Returns R(t), Ā(t) [shared by both models] and V_A(t) for
    the full (C_S+C_F) and reduced (C_S) closures."""
    G0 = 1.0 if rule == "cos" else 0.0
    S0 = order_parameter_S(R0, A0, K, delta)
    CS0 = (mu / gamma) * 0.5 * (S0 ** 2 - R0 ** 4)
    CF0 = (mu / (gamma + 2.0 * delta)) * 0.5 * (1.0 - S0 ** 2)

    def rhs(tt, y):
        R, A, CS, CF, Vf, Vc = y
        Rc = min(max(R, 0.0), 1.0)
        S = order_parameter_S(Rc, A, K, delta)
        sS2 = 0.5 * (S ** 2 - Rc ** 4)
        sF2 = 0.5 * (1.0 - S ** 2)
        dR = -delta * Rc + 0.5 * K * A * Rc * (1.0 - Rc ** 2)
        dA = mu * Rc ** 2 * G0 + gamma * (1.0 - A)
        dCS = -gamma * CS + mu * sS2
        dCF = -(gamma + 2.0 * delta) * CF + mu * sF2
        dVf = 2.0 * mu * (CS + CF) - 2.0 * gamma * Vf
        dVc = 2.0 * mu * CS - 2.0 * gamma * Vc
        return [dR, dA, dCS, dCF, dVf, dVc]

    sol = solve_ivp(rhs, (t[0], t[-1]), [R0, A0, CS0, CF0, V0, V0], t_eval=t,
                    method="RK45", rtol=1e-6, atol=1e-9)
    return sol.y[0], sol.y[1], sol.y[4], sol.y[5]                      # R, Ā, V_full, V_cs


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def nrmse(mf, micro):
    """RMSE normalized by the time average of the MICROSCOPIC quantity it is compared to."""
    ref = float(np.mean(np.abs(micro)))
    return rmse(mf, micro) / ref if ref > 0 else np.nan


def rule_errors(rule, d_arrays, mus, deltas, K, gamma, N, mode="abs"):
    """NRMSE of R and of the weight-variance quantity (both closures) over the (μ, Δ) grid.

    ``mode`` selects V_A ('abs') or V_A/Ā² ('rel') — for the microscopic reference AND for the
    mean field, which is divided by ITS OWN Ā(t). Returns (eR, eVf, eVc) each (n_mu, n_delta)
    and a dict (m, j) -> (R_fs, V_full, V_cs, V_micro) of the traces behind them."""
    R, Abar, VA, t = d_arrays
    nm, nd = len(mus), deltas.size
    eR = np.full((nm, nd), np.nan)
    eVf, eVc = np.full((nm, nd), np.nan), np.full((nm, nd), np.nan)
    traces = {}
    for m, mu in enumerate(mus):
        for j in range(nd):
            R_tr, A_tr, V_tr = R[m, j], Abar[m, j], VA[m, j]
            Rmf, Amf, Vf, Vc = mf_sim(rule, t, R_tr[0], A_tr[0], V_tr[0],
                                      K, gamma, float(mu), float(deltas[j]))
            Rmf_fs = np.sqrt(Rmf ** 2 + (1.0 - Rmf ** 2) / N)          # finite-size coherence floor
            q_mic = variance_quantity(V_tr, A_tr, mode)
            q_full = variance_quantity(Vf, Amf, mode)
            q_cs = variance_quantity(Vc, Amf, mode)
            eR[m, j] = nrmse(Rmf_fs, R_tr)
            eVf[m, j], eVc[m, j] = nrmse(q_full, q_mic), nrmse(q_cs, q_mic)
            traces[(m, j)] = (Rmf_fs, q_full, q_cs, q_mic)
    return eR, eVf, eVc, traces


def snap(values, grid):
    """Indices of the grid points closest to the requested values (order preserved)."""
    return [int(np.argmin(np.abs(np.asarray(grid) - float(v)))) for v in np.atleast_1d(values)]


# ════════════════════════════════════════════════════════════════════════════
#  one figure per adaptation rule: 4 rows × 4 columns
# ════════════════════════════════════════════════════════════════════════════
def rule_figure(rule, data, cfg):
    """RMSE row + 3×(2 μ) example traces of R and of the variance quantity, for one rule."""
    (mus, deltas, t, R, Abar, VA, K, gamma, N) = data
    mode = cfg["variance"]
    v_tex = VAR_TEX[mode]
    cmap = plt.get_cmap(cfg["cmap"])
    colors = [cmap(x) for x in np.linspace(0.15, 0.85, len(mus))]

    eR, eVf, eVc, traces = rule_errors(rule, (R, Abar, VA, t), mus, deltas, K, gamma, N, mode)
    m_idx = snap(cfg["trace_mus"], mus)                    # 2 columns of traces
    j_idx = snap(cfg["trace_deltas"], deltas)              # 3 rows of traces

    set_prl_style()
    fig = plt.figure(figsize=(COL_DOUBLE, 1.30 * (1 + len(j_idx)) + 0.4), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.05)
    gs = fig.add_gridspec(1 + len(j_idx), 4)
    letters = iter("abcdefghijklmnopqrstuvwxyz")

    # ── row 1: NRMSE vs Δ (all swept μ; the variance panel resolves both closures) ──
    ax_R = fig.add_subplot(gs[0, 0:2])
    ax_V = fig.add_subplot(gs[0, 2:4])
    for m in range(len(mus)):
        ax_R.plot(deltas, eR[m], color=colors[m], lw=1.0, marker="o", ms=1.8, zorder=3)
        ax_V.plot(deltas, eVf[m], color=colors[m], ls=V_STYLE["full"], lw=1.0, marker="o", ms=1.8,
                  zorder=3)
        ax_V.plot(deltas, eVc[m], color=colors[m], ls=V_STYLE["cs"], lw=1.0, marker="o", ms=1.8,
                  zorder=2)
    for ax, ylab in ((ax_R, r"RMSE$\,[R]/\langle R\rangle_t$"),
                     (ax_V, rf"RMSE$\,[{v_tex}]/\langle {v_tex}\rangle_t$")):
        ax.set_yscale("log")
        ax.set_xlim(0, deltas.max() * 1.02)
        ax.set_xlabel(r"heterogeneity $\Delta$", labelpad=1)
        ax.set_ylabel(ylab, labelpad=2)
        for j in j_idx:                                    # mark the Δ shown as traces below
            ax.axvline(deltas[j], color="0.75", lw=0.6, ls=":", zorder=1)
        panel_label(ax, next(letters))
    ax_R.legend(handles=[Line2D([0], [0], color=colors[m], lw=1.3, label=rf"${mu:g}$")
                         for m, mu in enumerate(mus)],
                loc="lower right", fontsize=5.2, handlelength=1.4, labelspacing=0.2,
                title=r"$\mu$", title_fontsize=5.8, borderaxespad=0.3, ncol=2)
    ax_V.legend(handles=[Line2D([0], [0], color="0.35", lw=1.0, ls=V_STYLE[v], label=V_LABEL[v])
                         for v in ("full", "cs")],
                loc="lower right", fontsize=5.2, handlelength=1.8, labelspacing=0.2,
                borderaxespad=0.3)

    # ── rows 2-4: traces, columns = (R at μ1, R at μ2, variance at μ1, variance at μ2) ──
    for r, j in enumerate(j_idx):
        for c in range(4):
            ax = fig.add_subplot(gs[1 + r, c])
            m = m_idx[c % 2]
            Rmf, q_full, q_cs, q_mic = traces[(m, j)]
            if c < 2:                                      # phase coherence
                ax.plot(t, R[m, j], color=C_MICRO, lw=1.6, alpha=0.5, zorder=2)
                ax.plot(t, Rmf, color=C_MF, lw=0.9, zorder=3)
                ax.set_ylim(0, 1.05)
                if c == 0:
                    ax.set_ylabel(r"coherence $R$", labelpad=2)
            else:                                          # weight variance
                ax.plot(t, q_mic, color=C_MICRO, lw=1.6, alpha=0.5, zorder=2)
                ax.plot(t, q_full, color=C_MF, lw=0.9, ls=V_STYLE["full"], zorder=3)
                ax.plot(t, q_cs, color=C_CS, lw=0.9, ls=V_STYLE["cs"], zorder=3)
                if c == 2:
                    ax.set_ylabel(VAR_YLAB[mode], labelpad=2)
            ax.set_xlim(t[0], t[-1])
            if r == 0:
                ax.set_title(rf"$\mu={mus[m]:g}$", fontsize=7, pad=3)
            if r == len(j_idx) - 1:
                ax.set_xlabel(r"time $t$", labelpad=1)
            if c == 0:                                     # Δ of this trace row
                ax.annotate(rf"$\Delta={deltas[j]:.2f}$", xy=(0, 0.5), xycoords="axes fraction",
                            xytext=(-38, 0), textcoords="offset points", rotation=90,
                            ha="center", va="center", fontsize=7)
            if r == 0 and c == 2:
                ax.legend(handles=[Line2D([0], [0], color=C_MICRO, lw=1.6, alpha=0.5, label="micro"),
                                   Line2D([0], [0], color=C_MF, lw=0.9, label=V_LABEL["full"]),
                                   Line2D([0], [0], color=C_CS, lw=0.9, ls=V_STYLE["cs"],
                                          label=V_LABEL["cs"])],
                          loc="best", fontsize=5.0, handlelength=1.8, labelspacing=0.18,
                          borderaxespad=0.3)
            panel_label(ax, next(letters), dx=-27)

    for j in j_idx:
        for m in m_idx:
            print(f"  G={rule:<4} μ={mus[m]:<6g} Δ={deltas[j]:.2f}: NRMSE[R]={eR[m, j]:.3f}  "
                  f"NRMSE[{VAR_NAME[mode]}]={eVf[m, j]:.3f} (full) / {eVc[m, j]:.3f} (C_S)")

    out = f"{cfg['out']}_{rule}" + ("_rel" if mode == "rel" else "")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".svg"); fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(f"[saved] {out}.svg / .png   (variance = {VAR_NAME[mode]})")


def main(cfg=CONFIG):
    d = np.load(cfg["micro_npz"], allow_pickle=True)
    rules = [str(r) for r in d["rules"]]
    mus, deltas, t = d["mus"], d["deltas"], d["t"]
    K, gamma, N = float(d["K"]), float(d["gamma"]), int(d["N"])
    for r, rule in enumerate(rules):
        rule_figure(rule, (mus, deltas, t, d["R"][r], d["Abar"][r], d["VA"][r], K, gamma, N), cfg)


def parse_args(cfg):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--variance", choices=["abs", "rel"],
                   help="weight-variance quantity: abs -> V_A (default), rel -> V_A/A^2")
    p.add_argument("--micro-npz", dest="micro_npz", help="microscopic sweep .npz")
    p.add_argument("--trace-mus", nargs=2, type=float, dest="trace_mus",
                   help="the two mu values of the trace columns")
    p.add_argument("--trace-deltas", nargs=3, type=float, dest="trace_deltas",
                   help="the three Delta values of the trace rows")
    p.add_argument("--out", help="output stem (rule and '_rel' suffixes are appended)")
    args = p.parse_args()
    return {**cfg, **{k: v for k, v in vars(args).items() if v is not None}}


if __name__ == "__main__":
    main(parse_args(CONFIG))
