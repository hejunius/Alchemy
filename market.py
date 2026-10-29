"""Market model: coefficients, CRRA utility, Merton closed form, policy helpers."""
import numpy as np
from config import MarketConfig


class Market:
    def __init__(self, cfg: MarketConfig):
        self.cfg = cfg
        self.r = cfg.r
        self.gamma = cfg.gamma
        n = cfg.n_assets
        self.mu_bar = np.asarray(cfg.mu_bar, dtype=float)
        self.B = np.asarray(cfg.B, dtype=float)
        sig = np.asarray(cfg.sigma, dtype=float)
        C = np.eye(n) * (1 - cfg.rho_asset) + cfg.rho_asset  # corr matrix
        self.Sigma = (sig[:, None] * C) * sig[None, :]       # D C D
        self.M = self.Sigma + cfg.M_reg * np.eye(n)          # ambiguity shape

    # ---- coefficient maps ----
    def mu(self, y):
        """mu(y) = mu_bar + B * y.  Scalar y -> (n,); array y of shape (N,)
        -> (N, n)."""
        y = np.asarray(y)
        if y.ndim == 0:
            return self.mu_bar + self.B * y
        return self.mu_bar[None, :] + np.outer(y, self.B)

    def lam(self, y):
        """Excess return vector lambda(y) = mu(y) - r * 1."""
        return self.mu(y) - self.r

    def lam_wc(self, y, pi):
        """Worst-case excess return under ellipsoidal drift ambiguity
        (companion paper Eq. (33)): pi' mu - eps_mu sqrt(pi' M pi)."""
        lam = self.lam(y)
        if self.cfg.eps_mu <= 0:
            return lam
        denom = np.sqrt(np.einsum("...i,...ij,...j->...", pi, self.M, pi))
        denom = np.expand_dims(np.maximum(denom, 1e-12), -1)
        return lam - self.cfg.eps_mu * (self.M @ pi.T).T / denom

    def q2(self, pi):
        """Squared volatility of log wealth: pi' Sigma pi."""
        return np.einsum("...i,...ij,...j->...", pi, self.Sigma, pi)

    def cost(self, v):
        """Running transaction cost c(v) = kappa1 |v|_1 + 0.5 kappa2 |v|^2."""
        return self.cfg.kappa1 * np.abs(v).sum(-1) + 0.5 * self.cfg.kappa2 * (v ** 2).sum(-1)

    def mu_y(self, y):
        return -self.cfg.kappa * (y - self.cfg.m_bar)

    # ---- utility ----
    def psi(self, z):
        """Terminal CRRA utility as a function of log wealth:
        Psi(z) = exp((1-gamma) z) / (1 - gamma)."""
        return np.exp((1 - self.gamma) * z) / (1 - self.gamma)

    # ---- closed forms ----
    def merton_pi(self):
        """Unconstrained Merton portfolio pi* = Sigma^{-1} lambda / gamma,
        evaluated at the baseline factor mean y = m_bar."""
        return np.linalg.solve(self.Sigma, self.lam(self.cfg.m_bar)) / self.gamma

    def merton_value(self, t, z):
        """Closed-form Merton value (no costs, no ambiguity, no factor):
        V(t,z) = exp((1-gamma) z)/(1-gamma) * exp((1-gamma) rho (T-t)),
        rho = r + |lambda|^2_{Sigma^{-1}} / (2 gamma)."""
        lam = self.lam(self.cfg.m_bar)
        rho = self.r + 0.5 / self.gamma * lam @ np.linalg.solve(self.Sigma, lam)
        return self.psi(z) * np.exp((1 - self.gamma) * rho * (self.cfg.T - t))

    # ---- policy helpers (running-cost formulation) ----
    @staticmethod
    def soft_threshold(g, tau, denom):
        """v* = shrink(g, tau) / denom, the maximizer of v'g - tau |v|_1
        - 0.5 denom |v|^2 (componentwise)."""
        return np.sign(g) * np.maximum(np.abs(g) - tau, 0.0) / denom

    def optimal_rate(self, E_Vpi, z):
        """Closed-form optimal trading rate given the expected value gradient
        E[V_pi]: v* = shrink(E[V_pi], lambda_tc kappa1) / (lambda_tc kappa2).
        (Wealth-independent in the running-cost formulation.)"""
        return self.soft_threshold(
            E_Vpi, self.cfg.lambda_tc * self.cfg.kappa1,
            self.cfg.lambda_tc * self.cfg.kappa2 + 1e-12)
