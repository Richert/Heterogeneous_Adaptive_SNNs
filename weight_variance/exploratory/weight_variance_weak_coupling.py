"""
Replicates the earlier V(t) / C(t) validation figure.

This validates the variance-covariance closure (manuscript Eqs. 15, 36, 37)
IN ISOLATION, in the drift regime it is derived for:
  - phases free-run: theta_i(t) = omega_i t + phi_i  (NO feedback of A into theta)
  - the "simulation" integrates independent scalar weight filters, one per pair
  - regime Delta >> gamma, V(0) = 0
It is NOT a test of the full closed-loop mean-field system (Eqs. 8-10 + 38).
"""
import numpy as np
import matplotlib.pyplot as plt

# ---- parameters: the closure's home turf (fast forcing) ----
gamma, Delta, mu = 0.2, 1.0, 0.3      # Delta >> gamma
M      = 200_000                       # number of independent pairs
dt, T  = 0.005, 30.0
lam    = 2.0 * Delta                   # forcing decorrelation rate
sF2    = 0.5                           # Var(G) in the incoherent limit, R=0

rng = np.random.default_rng(0)
# beat frequencies ~ Lorentzian of HWHM 2*Delta (difference of two HWHM-Delta Lorentzians)
dij = 2.0 * Delta * np.tan(np.pi * (rng.random(M) - 0.5))
dij = np.clip(dij, -200, 200)
psi = rng.uniform(0, 2*np.pi, M)

A = np.ones(M)                          # V0 = 0, all weights at the relaxation target
ts, Vs, Cs = [], [], []
nsteps = int(T/dt)
for k in range(nsteps):
    t = k*dt
    F = np.cos(dij*t + psi)             # G_ij with free-running phases
    A = A + dt * (mu*F + gamma*(1.0 - A))   # independent scalar filters (Euler)
    if k % 40 == 0:
        ts.append(t); Vs.append(A.var())
        Cs.append(np.mean((A - A.mean()) * (F - F.mean())))
ts, Vs, Cs = map(np.array, (ts, Vs, Cs))

# ---- closed-form predictions ----
Cinf  = mu*sF2/(gamma+lam)              # Eq. 37 with R=0  ->  mu/2 / (gamma+2 Delta)
Vinf  = mu**2*sF2/(gamma*(gamma+lam))   # Eq. 40 with R=0
beta  = gamma + lam
# exact two-mode V(t) for V0=C0=0 (keeps the fast C build-up term)
Vexact = Vinf*(1-np.exp(-2*gamma*ts)) \
         - (2*mu*Cinf/(beta-2*gamma))*(np.exp(-2*gamma*ts)-np.exp(-beta*ts))
Cpred  = Cinf*(1-np.exp(-beta*ts))

print(f"V_inf: sim={Vs[-1]:.5f}  theory={Vinf:.5f}")
print(f"C_inf: sim={Cs[-1]:.5f}  theory={Cinf:.5f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.plot(ts, Vs, 'o', ms=3, color='#444', alpha=.5, label='simulation (free-running phases)')
a1.plot(ts, Vexact, '-', lw=2.2, color='#b5179e', label='closure (exact two-mode)')
a1.axhline(Vinf, ls=':', c='gray'); a1.set_xlabel('t'); a1.set_ylabel('V(t)')
a1.legend(fontsize=8); a1.set_title(r'Variance closure, $\Delta\gg\gamma$')
a2.plot(ts, Cs, 'o', ms=3, color='#444', alpha=.5, label='simulation')
a2.plot(ts, Cpred, '-', lw=2.2, color='#e8871e', label=r'$C_\infty(1-e^{-(\gamma+\lambda)t})$')
a2.axhline(Cinf, ls=':', c='gray'); a2.set_xlabel('t'); a2.set_ylabel('C(t)')
a2.legend(fontsize=8); a2.set_title('Covariance build-up')
plt.tight_layout(); plt.savefig('variance_closure_isolated.png', dpi=140)
print("saved variance_closure_isolated.png")