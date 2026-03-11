import numpy as np
from numba import njit
from scipy.stats import bernoulli, rv_discrete

def uniform(N: int, eta: float, Delta: float) -> np.ndarray:
    return eta + Delta*np.linspace(-0.5, 0.5, N)

def normalize(x):
    x = x - np.mean(x)
    return x / np.std(x)

def correlate(x, y):
    x = normalize(x)
    y = normalize(y)
    c = np.corrcoef(x, y)
    return c[0, 1]

@njit
def integrate_noise(x, inp, scale, tau):
    return x + scale * inp - x / tau

def generate_colored_noise(num_samples, tau, scale=1.0):
    """
    Generates Brownian noise by integrating white noise.

    Args:
        num_samples (int): The number of samples in the output Brownian noise.
        scale (float): A scaling factor for the noise amplitude.

    Returns:
        numpy.ndarray: An array containing the generated Brownian noise.
    """
    white_noise = np.random.randn(num_samples)
    x = 0.0
    colored_noise = np.zeros_like(white_noise)
    for sample in range(num_samples):
        x = integrate_noise(x, white_noise[sample], scale, tau)
        colored_noise[sample] = x
    return colored_noise

def integrate(y: np.ndarray, func, args, T, dt, dts):

    steps = int(T / dt)
    store_step = int(dts / dt)
    state_rec = []

    # solve ivp with Heun's method
    for step in range(steps):
        if step % store_step == 0:
            state_rec.append(y[:])
        rhs = func(step, y, *args)
        y_0 = y + dt * rhs
        y = y + (rhs + func(step, y_0, *args)) * dt/2

    return np.asarray(state_rec), y

def get_eigs(rates: np.ndarray, epsilon: float = 1e-12) -> tuple:

    rates_centered = np.zeros_like(rates)
    for i in range(rates.shape[1]):
        rates_centered[:, i] = rates[:, i] - np.mean(rates[:, i])
        rates_centered[:, i] /= (np.std(rates[:, i]) + epsilon)
    C = np.cov(rates_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx], C

def get_ff(rates: np.ndarray) -> np.ndarray:
    n = rates.shape[1]
    ff = np.zeros((n,))
    for i in range(n):
        ff[i] = np.var(rates[:, i]) / np.mean(rates[:, i])
    return ff

def dist(x: int, method: str = "inverse", zero_val: float = 1.0, inverse_pow: float = 1.0) -> float:
    if method == "inverse":
        return 1/x**inverse_pow if x > 0 else zero_val
    if method == "exp":
        return np.exp(-x) if x > 0 else zero_val
    else:
        raise ValueError("Invalid method.")

def chain_connectivity(N: int, p: float, spatial_distribution: rv_discrete, homogeneous_weights: bool = True
                       ) -> np.ndarray:
    """Generate a coupling matrix between nodes aligned on a circle.

    Parameters
    ----------
    N
        Number of nodes.
    p
        Connection probability.
    spatial_distribution
        Probability distribution defined over space. Will be used to draw indices of nodes from which each node in the
        circular network receives inputs.
    homogeneous_weights
        If true, all incoming weights to a node will have the same strength. Since incoming edges are drawn
        with replacement from the spatial distribution, this means that the actual connection probability is smaller or
        equal to p. If false, each drawn sample will contribute to the edge weights, such that the resulting edge
        strengths can be heterogeneous.

    Returns
    -------
    np.ndarray
        2D coupling matrix (N x N).
    """
    C = np.zeros((N, N))
    n_conns = int(N*p)
    for n in range(N):
        idxs = spatial_distribution.rvs(size=n_conns)
        signs = 1 * (bernoulli.rvs(p=0.5, loc=0, size=n_conns) > 0)
        signs[signs == 0] = -1
        conns = n + idxs*signs
        conns = conns[(conns >= 0) & (conns < N)]
        conns_unique = np.unique(conns)
        if homogeneous_weights:
            C[n, conns_unique] = 0.5
        else:
            for idx in conns_unique:
                C[n, idx] = np.random.rand()
    return C

def circular_connectivity(N: int, p: float, spatial_distribution: rv_discrete, homogeneous_weights: bool = True
                          ) -> np.ndarray:
    """Generate a coupling matrix between nodes aligned on a circle.

    Parameters
    ----------
    N
        Number of nodes.
    p
        Connection probability.
    spatial_distribution
        Probability distribution defined over space. Will be used to draw indices of nodes from which each node in the
        circular network receives inputs.
    homogeneous_weights
        If true, all incoming weights to a node will have the same strength. Since incoming edges are drawn
        with replacement from the spatial distribution, this means that the actual connection probability is smaller or
        equal to p. If false, each drawn sample will contribute to the edge weights, such that the resulting edge
        strengths can be heterogeneous.

    Returns
    -------
    np.ndarray
        2D coupling matrix (N x N).
    """
    C = np.zeros((N, N))
    n_conns = int(N*p)
    for n in range(N):
        idxs = spatial_distribution.rvs(size=n_conns)
        signs = 1 * (bernoulli.rvs(p=0.5, loc=0, size=n_conns) > 0)
        signs[signs == 0] = -1
        conns = _wrap(n + idxs*signs, N)
        conns_unique = np.unique(conns)
        if homogeneous_weights:
            C[n, conns_unique] = 0.5
        else:
            for idx in conns_unique:
                C[n, idx] = np.random.rand()
    return C

def _wrap(idxs: np.ndarray, N: int) -> np.ndarray:
    idxs[idxs < 0] = N+idxs[idxs < 0]
    idxs[idxs >= N] = idxs[idxs >= N] - N
    return idxs
