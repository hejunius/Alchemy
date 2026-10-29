"""Monotone semi-Lagrangian Bellman(-Isaacs) reference solver on a regular grid.

Implements the recursion of the companion paper (Eq. (94)) with the analytical
worst-case drift (saddle reduction) and closed-form trading-rate optimization
(soft thresholding).  State: z = log wealth, y = factor, pi = holdings.
"""
import numpy as np
from config import MarketConfig, DomainConfig
from market import Market


def interp_nd(V, axes, X):
    """Multilinear interpolation of V (ndarray, shape = tuple(len(a) for a in axes))
    at points X (N, d), sequentially along each axis.  Points outside the domain
    are clipped to the boundary (monotone, Dirichlet-style extrapolation)."""
    N, d = X.shape
    shape = V.shape
    Vc = V
    for j in range(d):
        xj = (X[:, j] - axes[j][0]) / (axes[j][-1] - axes[j][0])
        xj = np.clip(xj * (shape[j] - 1), 0.0, shape[j] - 1 - 1e-9)
        i0 = xj.astype(int)
        w1 = (xj - i0).reshape([N] + [1] * (d - 1 - j))
        if j == 0:
            a = Vc[i0]          # gather along the first (grid) axis
            b = Vc[i0 + 1]
        else:
            # Vc = (N, n_j, ..., n_{d-1}): per-sample gather along axis 1
            idx = (np.arange(N), i0) + (slice(None),) * (d - 1 - j)
            a = Vc[idx]
            b = Vc[(np.arange(N), i0 + 1) + (slice(None),) * (d - 1 - j)]
        Vc = (1 - w1) * a + w1 * b
    return Vc


class GridSolver:
    """Backward Bellman recursion on the regular grid of DomainConfig."""

    def __init__(self, mkt: Market, dom: DomainConfig):
        self.mkt = mkt
        self.dom = dom
        self.cfg = mkt.cfg
        n = mkt.cfg.n_assets
        g = dom.grid
        assert len(g) == 2 + n, "grid must be (nz, ny, npi...)"
        self.axes = [
            np.linspace(dom.z_lo, dom.z_hi, g[0]),
            np.linspace(dom.y_lo, dom.y_hi, g[1]),
        ] + [np.linspace(dom.pi_lo, dom.pi_hi, gj) for gj in g[2:]]
        self.dt = mkt.cfg.T / mkt.cfg.K
        self.V = [None] * (mkt.cfg.K + 1)
        self.policy = [None] * mkt.cfg.K   # optimal rates on the grid (Nnodes, n)
        # full outer-product grid: each array has shape grid_shape
        G = np.meshgrid(*self.axes, indexing="ij")
        self.grid_shape = tuple(len(a) for a in self.axes)
        self.Z_full, self.Y_full = G[0], G[1]
        self.P_full = G[2:]                     # list of pi grids
        # terminal value on the grid
        self.V[-1] = mkt.psi(G[0])

    def _gradients(self, V):
        """Grid gradients of the interpolant (central differences)."""
        grads = []
        for j in range(V.ndim):
            ax = self.axes[j]
            g = np.gradient(V, ax, axis=j, edge_order=2)
            grads.append(g)
        return grads

    def _state_grid(self):
        """Flattened grid states: shape (Nnodes, 2 + n)."""
        G = [self.Z_full, self.Y_full] + list(self.P_full)
        return np.stack([g.ravel() for g in G], axis=-1)

    def solve(self):
        """Run the backward recursion; returns None, fills self.V/self.policy."""
        mkt, dom, cfg = self.mkt, self.dom, self.cfg
        n = cfg.n_assets
        states = self._state_grid()
        z0, y0, pi0 = states[:, 0], states[:, 1], states[:, 2:]
        sqrt_dt = np.sqrt(self.dt)
        # quadrature: 2-point per shock; shocks = (wealth scalar, y)
        qs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)

        for k in reversed(range(cfg.K)):
            Vnext = self.V[k + 1]
            Gz, Gy = self._gradients(Vnext)[:2]
            Gpi = self._gradients(Vnext)[2:]          # list per asset
            EV = np.zeros_like(z0)
            EGpi = np.zeros_like(pi0)
            q = np.sqrt(np.maximum(mkt.q2(pi0), 1e-16))
            lam_wc = mkt.lam_wc(y0, pi0)
            for xi in qs:
                z1 = z0 + (cfg.r + np.einsum("ni,ni->n", pi0, lam_wc)
                           - 0.5 * q ** 2) * self.dt + sqrt_dt * q * xi[0]
                y1 = y0 + mkt.mu_y(y0) * self.dt + sqrt_dt * cfg.nu * xi[1]
                X1 = np.concatenate([z1[:, None], y1[:, None], pi0], axis=-1)
                EV += interp_nd(Vnext, self.axes, X1)
                EGpi += np.stack(
                    [interp_nd(g, self.axes, X1) for g in Gpi], axis=-1)
            EV /= len(qs)
            EGpi /= len(qs)
            v = mkt.optimal_rate(EGpi, z0)            # closed-form max over v
            v = np.clip(v, -dom.vmax, dom.vmax)
            # hold the portfolio inside the box
            v = np.clip(v, (dom.pi_lo - pi0) / self.dt, (dom.pi_hi - pi0) / self.dt)
            # value update: -K dt + E[V] + linear correction in v
            Vk = -cfg.lambda_tc * mkt.cost(v) * self.dt + EV \
                 + np.einsum("ni,ni->n", v, EGpi) * self.dt
            self.V[k] = Vk.reshape(Vnext.shape)
            self.policy[k] = v.reshape(pi0.shape[0], n) if False else v
        return self

    # ---- evaluation utilities ----
    def value_at(self, t, X):
        """Interpolate the value function at times t (scalar or (N,) array)
        and states X (N, d)."""
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        if scalar:                      # one time layer, all states
            k = min(int(t / self.dt), self.cfg.K)
            return interp_nd(self.V[k], self.axes, X)
        t_arr = np.atleast_1d(t)
        k_arr = np.minimum((t_arr / self.dt).astype(int), self.cfg.K)
        out = np.empty(len(t_arr))
        for k in np.unique(k_arr):
            m = k_arr == k
            out[m] = interp_nd(self.V[k], self.axes, X[m])
        return out

    def policy_at(self, t, X):
        """Interpolate the optimal trading rate at times t (scalar or (N,) array)."""
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        shape = tuple(len(a) for a in self.axes)
        n = self.mkt.cfg.n_assets
        if scalar:                      # one time layer, all states
            k = min(int(t / self.dt), self.cfg.K - 1)
            P = self.policy[k].reshape(shape + (n,))
            return np.stack([interp_nd(P[..., j], self.axes, X) for j in range(n)],
                            axis=-1)
        t_arr = np.atleast_1d(t)
        k_arr = np.minimum((t_arr / self.dt).astype(int), self.cfg.K - 1)
        out = np.empty((len(t_arr), n))
        for k in np.unique(k_arr):
            m = k_arr == k
            P = self.policy[k].reshape(shape + (n,))
            for j in range(n):
                out[m, j] = interp_nd(P[..., j], self.axes, X[m])
        return out


def simulate_paths(mkt: Market, dom: DomainConfig, policy_fn, rng: np.random.Generator):
    """Forward simulation of wealth/factor/holdings under a policy function
    policy_fn(t, z, y, pi) -> v.  Returns dict of arrays."""
    cfg = mkt.cfg
    dt = cfg.T / cfg.K
    n_steps = int(cfg.n_years * cfg.K)
    N = cfg.n_paths
    n = cfg.n_assets
    z = np.full(N, 0.0)                      # log wealth, X0 = 1
    y = np.zeros(N)
    pi = np.zeros((N, n))
    m = np.zeros(N)                          # regime (index)
    wealth_logs = np.empty((n_steps + 1, N))
    turnover = np.zeros(N)
    total_cost = np.zeros(N)
    wealth_logs[0] = z
    pi_star = mkt.merton_pi()

    for k in range(n_steps):
        t = (k % cfg.K) * dt
        # regime switching (hidden); factor mean follows regime
        switch = rng.random(N) < cfg.switch_rate * dt
        m = np.where(switch, 1 - m, m)
        m_mean = np.asarray(cfg.regime_means)[m.astype(int)]
        v = policy_fn(t, z, y, pi)
        v = np.clip(v, -dom.vmax, dom.vmax)
        v = np.clip(v, (dom.pi_lo - pi) / dt, (dom.pi_hi - pi) / dt)
        turnover += np.abs(v).sum(-1) * dt
        total_cost += cfg.lambda_tc * mkt.cost(v) * dt
        q = np.sqrt(np.maximum(mkt.q2(pi), 1e-16))
        lam_wc = mkt.lam_wc(y, pi)
        xi1, xi2 = rng.normal(size=(2, N))
        z = z + (cfg.r + np.einsum("ni,ni->n", pi, lam_wc) - 0.5 * q ** 2) * dt \
            + np.sqrt(dt) * q * xi1
        y = y + (-cfg.kappa * (y - m_mean)) * dt + cfg.nu * np.sqrt(dt) * xi2
        pi = pi + v * dt
        wealth_logs[k + 1] = z
    return dict(wealth=np.exp(wealth_logs), z=wealth_logs, turnover=turnover,
                cost=total_cost, pi_star=pi_star)
