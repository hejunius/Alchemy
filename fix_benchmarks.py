"""Rerun the three analytical benchmarks with a damped fixed-point iteration
for the fixed-radius robust policy (the pure-substitution iteration oscillates
in a 2-cycle at high n).  Only simulations; the NN policy results stay as in
highdim_n*.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grid_solver import simulate_paths
from market import Market
from run_experiments import RESULTS, merton_policy_fn, pnl_summary
from run_scaling import market_for


def fixed_radius_policy_damped(mkt, eps, beta=0.5, iters=60):
    """Damped fixed point pi = Sigma^{-1} lam_wc(y, pi) / gamma."""
    Sinv = np.linalg.inv(mkt.Sigma)
    gamma = mkt.gamma
    dt_max = mkt.cfg.T / mkt.cfg.K

    def fn(t, z, y, pi):
        y = np.asarray(y).ravel()
        lam = mkt.mu_bar[None, :] + np.outer(y, mkt.B) - mkt.r
        p = np.tile(mkt.merton_pi(), (len(y), 1))
        for _ in range(iters):
            denom = np.sqrt(np.maximum(
                np.einsum("ni,ij,nj->n", p, mkt.M, p), 1e-12))[:, None]
            wc = lam - eps * (mkt.M @ p.T).T / denom
            p = (1 - beta) * p + beta * (wc @ Sinv.T / gamma)
        gap = p - pi
        return np.clip(gap / dt_max, -2.0, 2.0)
    return fn


def homothetic_policy(mkt, theta=1.0):
    Sinv = np.linalg.inv(mkt.Sigma)
    gamma = mkt.gamma + theta
    dt_max = mkt.cfg.T / mkt.cfg.K

    def fn(t, z, y, pi):
        y = np.asarray(y).ravel()
        lam = mkt.mu_bar[None, :] + np.outer(y, mkt.B) - mkt.r
        p = lam @ Sinv.T / gamma
        return np.clip((p - pi) / dt_max, -2.0, 2.0)
    return fn


def run(n):
    mkt_cfg = market_for(n)
    dom_cfg = __import__("config").DomainConfig()
    mkt = Market(mkt_cfg)
    mkt0 = Market(market_for(n, robust=False))
    g, K = mkt_cfg.gamma, mkt_cfg.K
    out = {"n": n}
    for meas, mk in (("worst_case", mkt), ("baseline", mkt0)):
        rng = np.random.default_rng(0)
        fr = simulate_paths(mk, dom_cfg,
                            fixed_radius_policy_damped(mk, 0.3), rng)
        rng2 = np.random.default_rng(0)
        hr = simulate_paths(mk, dom_cfg, homothetic_policy(mk), rng2)
        rng3 = np.random.default_rng(0)
        me = simulate_paths(mk, dom_cfg, merton_policy_fn(mk), rng3)
        out[meas] = dict(
            fixed_radius=pnl_summary(fr, g, K),
            homothetic=pnl_summary(hr, g, K),
            merton=pnl_summary(me, g, K))
    path = os.path.join(RESULTS, f"highdim_n{n}_bench.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    for meas in ("baseline", "worst_case"):
        m = out[meas]
        print(f"[n={n}] {meas}: merton U {m['merton']['mean_terminal_utility']:.3f}"
              f" dd {m['merton']['max_drawdown']:.3f} | fixedr U "
              f"{m['fixed_radius']['mean_terminal_utility']:.3e} dd "
              f"{m['fixed_radius']['max_drawdown']:.3f} | homoth U "
              f"{m['homothetic']['mean_terminal_utility']:.3e} dd "
              f"{m['homothetic']['max_drawdown']:.3f}", flush=True)
    return out


if __name__ == "__main__":
    for n in [int(x) for x in sys.argv[1:]]:
        run(n)
