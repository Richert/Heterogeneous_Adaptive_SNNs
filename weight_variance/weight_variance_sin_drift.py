r"""
Antisymmetric adaptation: collective frequency shift Ω of the adaptive Kuramoto network
========================================================================================

Under G(φ) = sin(φ) the slaved weights A_ij = 1 + κΓ_ij (κ = μ/γ, Γ_ij = ⟨sin(θ_j − θ_i)⟩_t) break the
(θ, ω) → (−θ, −ω) symmetry, and the synchronized cluster rotates at Ω ∝ κ although ω̄ = 0. This script
computes Ω (and R, per-oscillator mean frequencies) over a κ-sweep in three ways:

  * micro       — the N = 500 network (Euler, dt = 0.05, the kernel of weight_variance_rule_micro.py,
                  uniform-weight MF-fixed-point initial condition of weight_variance_rule_micro_oainit.py);
  * quasistatic — self-consistent solution of the static-field problem on the same N frequencies:
                  with c_j = ⟨e^{iθ_j}⟩_t (frame rotating at Ω, Ψ = 0), oscillator i feels the field
                  F_i = R + (κ/2i)(W c_i* − S c_i),  S = ⟨|c|²⟩, W = ⟨c²⟩,
                  and is locked (c_i = e^{i(α + asin(ω̃/ρ))}) or drifting
                  (c_i = e^{iα} i(ω̃ − sgn ω̃ √(ω̃² − ρ²))/ρ), ρe^{iα} = K F_i, ω̃ = ω − Ω; ⟨c⟩ = R real;
  * first order — the linear coefficient Ω/κ at κ → 0 of the same problem, evaluated on 10⁶ quantiles
                  (continuum limit; the N = 500 quantile sum resolves the 1/√ singularity at the
                  locking edge too coarsely).

    python weight_variance_sin_drift.py            # micro + theory → weight_variance_sin_drift.npz
    python weight_variance_sin_drift.py --theory   # theory only (fast)
"""

# --- shared library bootstrap (repo-root shared/) ---------------------------
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "shared")]
import data_paths as dp
# ---------------------------------------------------------------------------
import argparse, time
import numpy as np
from scipy.optimize import brentq

import weight_variance_rule_micro as base
import weight_variance_rule_micro_oainit as oai

CONFIG = dict(N=500, K=1.0, gamma=0.01, dt=0.05, T0=2500.0, Tw=500.0, seed=2,
              deltas=[0.1, 0.3],
              kappas=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0],
              profile=(0.3, 0.3),                 # (Δ, κ) whose per-oscillator frequencies are stored
              lock_tol=5e-3, N_continuum=1_000_000,
              out=dp.kmo_adaptive("weight_variance_sin_drift.npz"))


# ----------------------------------------------------------------------------- theory
def kuramoto_state(omega, K):
    """κ = 0 (Kuramoto) stationary state on the frequencies `omega`: R, locked mask, c_j."""
    def res(R):
        b = K * R
        lk = np.abs(omega) < b
        return np.sum(np.sqrt(1.0 - (omega[lk] / b) ** 2)) / omega.size - R
    R = brentq(res, 1e-3, 1.0)
    b = K * R
    lk = np.abs(omega) < b
    c = np.empty(omega.size, complex)
    c[lk] = np.exp(1j * np.arcsin(omega[lk] / b))
    w = omega[~lk]
    c[~lk] = 1j * (w - np.sign(w) * np.sqrt(w ** 2 - b ** 2)) / b
    return R, lk, c


def first_order(omega, K):
    """Ω/κ at κ → 0 and its two contributions (locked push, drifter coupling asymmetry)."""
    R, lk, c = kuramoto_state(omega, K)
    N, b = omega.size, K * R
    S, C = np.mean(np.abs(c) ** 2), np.mean(c ** 2).real
    phi = np.angle(c[lk])
    push = 0.5 * (S - C * np.cos(2.0 * phi))                  # locked: θ̇ ⊃ Kκ (S − C cos2φ)/2
    P_L = np.sum(np.cos(phi) ** 2) / N                         # drifters: b → b − Kκ q P_L
    w, q = omega[~lk], c[~lk].imag
    rt = np.sqrt(w ** 2 - b ** 2)
    dq_dw = (1.0 - np.abs(w) / rt) / b
    dq_db = -q / b + np.sign(w) / rt
    den = lk.sum() / (N * b) + np.sum(dq_dw) / N
    locked = K * np.sum(push) / (N * b) / den
    drift = -K * P_L * np.sum(q * dq_db) / N / den
    return dict(R=R, coef=locked + drift, locked=locked, drift=drift, S=S, C=C, P_L=P_L, L=lk.mean())


def _field_fixed_point(wt, R, S, W, kap, K, c0, n_it):
    """Per-oscillator static-field fixed point (damped). Returns c, locked mask, field modulus ρ."""
    c = c0.copy()
    for _ in range(n_it):
        F = K * (R + kap / 2j * (W * np.conj(c) - S * c))
        rho, al = np.abs(F), np.angle(F)
        lk = np.abs(wt) < rho
        cn = np.empty_like(c)
        cn[lk] = np.exp(1j * (al[lk] + np.arcsin(wt[lk] / rho[lk])))
        wd, rd = wt[~lk], rho[~lk]
        cn[~lk] = np.exp(1j * al[~lk]) * 1j * (wd - np.sign(wd) * np.sqrt(wd ** 2 - rd ** 2)) / rd
        c = 0.5 * c + 0.5 * cn
    return c, lk, rho


def quasistatic(omega, kap, K, n_outer=300, n_avg=100, R_tol=5e-4, Om_rtol=1e-2):
    """Self-consistent (R, Ω) of the static-field problem; per-oscillator mean frequency ⟨θ̇⟩ − Ω.

    On a finite frequency set, single oscillators at the locking edge can keep switching between the
    locked and the drifting solution, so convergence is judged on the macroscopic (R, Ω) only: they
    are averaged over the last `n_avg` iterations, and the solution is accepted if their spread is
    below `R_tol` and `Om_rtol`·|Ω|. Returns None otherwise (no stationary static-field solution)."""
    R, _, c = kuramoto_state(omega, K)
    Rs, Oms, ws = [], [], []
    try:
        for it in range(n_outer):
            S, W = np.mean(np.abs(c) ** 2), np.mean(c ** 2)
            Om = brentq(lambda O: _field_fixed_point(omega - O, R, S, W, kap, K, c, 60)[0].mean().imag,
                        -2.0, 2.0)
            c, lk, rho = _field_fixed_point(omega - Om, R, S, W, kap, K, c, 200)
            R = 0.5 * R + 0.5 * c.mean().real
            if it >= n_outer - n_avg:
                wt = omega - Om
                Rs.append(R); Oms.append(Om)
                ws.append(np.where(lk, 0.0, np.sign(wt) * np.sqrt(np.maximum(wt ** 2 - rho ** 2, 0.0))))
    except ValueError:
        return None
    Rs, Oms = np.array(Rs), np.array(Oms)
    if Rs.std() > R_tol or Oms.std() > Om_rtol * max(abs(Oms.mean()), 1e-3):
        return None
    return dict(R=Rs.mean(), Omega=Oms.mean(), w_eff=np.mean(ws, axis=0))


# ----------------------------------------------------------------------------- microscopic network
def micro(D, kap, cfg):
    """Euler simulation; Ω from a linear fit to the unwrapped mean phase over the final window."""
    N, K, g, dt = cfg["N"], cfg["K"], cfg["gamma"], cfg["dt"]
    mu = kap * g
    n0, n1 = int(cfg["T0"] / dt), int(cfg["Tw"] / dt)
    omega = base.lorentzian_quantiles(N, D)
    rng = np.random.default_rng(cfg["seed"])
    theta, A, _ = oai.init_state("sin", D, mu, dict(K=K, gamma=g, init="uniform"), omega, rng)
    theta, A = np.ascontiguousarray(theta, float), np.ascontiguousarray(A, float)
    e0 = np.empty(0)
    base._simulate_numba_z(theta, A, omega, K, mu, g, dt, n0, np.zeros(n0, np.bool_), 0.0,
                           e0, e0, e0, e0.astype(complex), e0.astype(complex))           # transient
    th_mid = theta.copy()
    Rs, Ab, V = np.empty(n1), np.empty(n1), np.empty(n1)
    Z1, Z2 = np.empty(n1, complex), np.empty(n1, complex)
    base._simulate_numba_z(theta, A, omega, K, mu, g, dt, n1, np.ones(n1, np.bool_), 0.0,
                           Rs, Ab, V, Z1, Z2)                                              # window
    t = np.arange(n1) * dt
    Om = np.polyfit(t, np.unwrap(np.angle(Z1)), 1)[0]
    w_eff = (theta - th_mid) / cfg["Tw"] - Om
    return dict(R=Rs.mean(), Omega=Om, w_eff=w_eff, locked=np.abs(w_eff) < cfg["lock_tol"])


# ----------------------------------------------------------------------------- main
def main(theory_only=False, cfg=CONFIG):
    deltas, kappas = np.array(cfg["deltas"]), np.array(cfg["kappas"])
    nD, nk = deltas.size, kappas.size
    out = dict(deltas=deltas, kappas=kappas, K=cfg["K"], gamma=cfg["gamma"], N=cfg["N"],
               profile=np.array(cfg["profile"]))
    fo = [first_order(base.lorentzian_quantiles(cfg["N_continuum"], D), cfg["K"]) for D in deltas]
    for k in ("coef", "locked", "drift", "R"):
        out[f"fo_{k}"] = np.array([f[k] for f in fo])
    qs_R, qs_Om = np.full((nD, nk), np.nan), np.full((nD, nk), np.nan)
    for a, D in enumerate(deltas):
        om = base.lorentzian_quantiles(cfg["N"], D)
        for b_, kap in enumerate(kappas):
            r = quasistatic(om, kap, cfg["K"])
            if r is not None:
                qs_R[a, b_], qs_Om[a, b_] = r["R"], r["Omega"]
            if (D, kap) == tuple(cfg["profile"]) and r is not None:
                out["omega_profile"], out["qs_w_eff"] = om, r["w_eff"]
            print(f"[theory] Δ={D} κ={kap}: " + ("no stationary solution" if r is None else
                  f"R={r['R']:.4f} Ω={r['Omega']:.5f} Ω/κ={r['Omega'] / kap:.4f}"), flush=True)
        print(f"[theory] Δ={D}: first-order Ω/κ = {fo[a]['coef']:.4f} "
              f"(locked {fo[a]['locked']:.4f}, drift {fo[a]['drift']:+.4f})", flush=True)
    out.update(qs_R=qs_R, qs_Omega=qs_Om)
    if not theory_only:
        mi_R, mi_Om = np.empty((nD, nk)), np.empty((nD, nk))
        mi_nlock = np.empty((nD, nk), int)
        for a, D in enumerate(deltas):
            for b_, kap in enumerate(kappas):
                t0 = time.time()
                r = micro(D, kap, cfg)
                mi_R[a, b_], mi_Om[a, b_], mi_nlock[a, b_] = r["R"], r["Omega"], r["locked"].sum()
                if (D, kap) == tuple(cfg["profile"]):
                    out["mi_w_eff"] = r["w_eff"]
                print(f"[micro] Δ={D} κ={kap}: R={r['R']:.4f} Ω={r['Omega']:.5f} "
                      f"Ω/κ={r['Omega'] / kap:.4f} n_lock={r['locked'].sum()} ({time.time() - t0:.1f}s)",
                      flush=True)
        out.update(mi_R=mi_R, mi_Omega=mi_Om, mi_nlock=mi_nlock)
    elif os.path.exists(cfg["out"]):                       # keep previous micro results
        old = np.load(cfg["out"])
        out.update({k: old[k] for k in old.files if k.startswith("mi_")})
    np.savez(cfg["out"], **out)
    print(f"saved {cfg['out']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", action="store_true", help="recompute the theory curves only")
    main(ap.parse_args().theory)
