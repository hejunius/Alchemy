"""Scaling driver: (a) certification rows for n=3,6 (grid reference feasible);
(b) high-dimensional neural-only campaign n=30,50,100 vs analytical benchmarks.

Usage:
    python run_scaling.py --n-cert 3 6        # certification rows
    python run_scaling.py --n-high 30 50 100   # neural-only high-dim campaign
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DomainConfig, MarketConfig, NNConfig
from market import Market
from grid_solver import GridSolver, simulate_paths
from nn_solver import SolverNN
from run_experiments import (RESULTS, merton_policy_fn, pnl_summary,
                             policy_from_grid, value_from_grid)

N_TRAIN_ITERS = 3000  # high-dim training iterations (n=2 stages use 4000)


def market_for(n, robust=True, rho=0.3):
    """n-asset market from a one-factor model with heterogeneous loadings."""
    rng = np.random.default_rng(n)  # deterministic per n
    mu_bar = tuple(0.04 + 0.04 * rng.random(n))
    B = tuple(0.5 + rng.random(n))
    sigma = tuple(0.15 + 0.20 * rng.random(n))
    cfg = MarketConfig()
    cfg.n_assets = n
    cfg.mu_bar, cfg.B, cfg.sigma = mu_bar, B, sigma
    cfg.rho_asset = rho
    cfg.kappa1, cfg.kappa2, cfg.lambda_tc = 0.05, 0.2, 1.0
    cfg.eps_mu = 0.3 if robust else 0.0
    cfg.M_reg = 0.05
    return cfg


def run_cert(n, grid):
    """Grid-certified run for n assets (robust stage-3 config)."""
    mkt_cfg = market_for(n)
    dom_cfg = DomainConfig()
    dom_cfg.grid = grid
    nn_cfg = NNConfig()
    nn_cfg.iterations = 4000
    mkt = Market(mkt_cfg)
    t0 = time.time()
    grid_solver = GridSolver(mkt, dom_cfg).solve()
    grid_time = time.time() - t0
    print(f"[n={n}] grid ({grid}) solved in {grid_time:.1f}s", flush=True)
    solver = SolverNN(mkt, dom_cfg, nn_cfg)
    solver.warm_start(value_from_grid(grid_solver), iters=nn_cfg.warm_iters)
    info = solver.train(ref_solver=grid_solver, verbose=True)
    cert = solver.eval_grid(grid_solver)
    rng = np.random.default_rng(0)
    sim_nn = simulate_paths(mkt, dom_cfg, solver.policy_fn(), rng)
    rng2 = np.random.default_rng(0)
    sim_grid = simulate_paths(mkt, dom_cfg, policy_from_grid(grid_solver), rng2)
    rng3 = np.random.default_rng(0)
    sim_merton = simulate_paths(mkt, dom_cfg, merton_policy_fn(mkt), rng3)
    out = dict(n=n, grid=grid, cert=cert, grid_time=grid_time,
               train_time=info["elapsed"], pnl=dict(
                   nn=pnl_summary(sim_nn, mkt_cfg.gamma, mkt_cfg.K),
                   grid=pnl_summary(sim_grid, mkt_cfg.gamma, mkt_cfg.K),
                   merton=pnl_summary(sim_merton, mkt_cfg.gamma, mkt_cfg.K)))
    path = os.path.join(RESULTS, f"cert_n{n}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[n={n}] done: L2 {cert['value_l2']:.4f} Linf {cert['value_linf']:.4f} "
          f"pol {cert['policy_err']:.4f} | nn util "
          f"{out['pnl']['nn']['mean_terminal_utility']:.4f} vs grid "
          f"{out['pnl']['grid']['mean_terminal_utility']:.4f}", flush=True)
    return out


def robust_fixed_radius_policy(mkt, eps):
    """Policy trading toward the myopic robust allocation with a FIXED radius
    eps: pi_rob = Sigma^{-1} lam_wc(y, pi_rob) / gamma (fixed-point)."""
    Sigma_inv = np.linalg.inv(mkt.Sigma)
    gamma = mkt.gamma
    dt_max = mkt.cfg.T / mkt.cfg.K

    def fn(t, z, y, pi):
        y = np.asarray(y).ravel()
        lam = mkt.mu_bar[None, :] + np.outer(y, mkt.B) - mkt.r
        pi_rob = np.tile(mkt.merton_pi(), (len(y), 1))
        for _ in range(15):  # fixed-point iteration for the robust allocation
            denom = np.sqrt(np.maximum(
                np.einsum("ni,ij,nj->n", pi_rob, mkt.M, pi_rob), 1e-12))[:, None]
            wc = lam - eps * (mkt.M @ pi_rob.T).T / denom
            pi_rob = wc @ Sigma_inv.T / gamma
        gap = pi_rob - pi
        return np.clip(gap / dt_max, -2.0, 2.0)
    return fn


def homothetic_policy(mkt, theta=1.0):
    """Maenhout-style homothetic robustness: effective risk aversion
    gamma + theta, trading toward the shrunk allocation."""
    Sigma_inv = np.linalg.inv(mkt.Sigma)
    gamma = mkt.gamma + theta
    dt_max = mkt.cfg.T / mkt.cfg.K

    def fn(t, z, y, pi):
        y = np.asarray(y).ravel()
        lam = mkt.mu_bar[None, :] + np.outer(y, mkt.B) - mkt.r
        pi_hr = lam @ Sigma_inv.T / gamma
        gap = pi_hr - pi
        return np.clip(gap / dt_max, -2.0, 2.0)
    return fn


def run_highdim(n):
    """Neural-only training (no reference) + benchmark PnL under both measures."""
    mkt_cfg = market_for(n)
    dom_cfg = DomainConfig()
    nn_cfg = NNConfig()
    nn_cfg.iterations = N_TRAIN_ITERS
    nn_cfg.warm_iters = 0          # no reference to warm-start from
    nn_cfg.w_res, nn_cfg.w_anchor = 1.0, 0.0   # pure HJBI residual
    nn_cfg.n_adv, nn_cfg.adv_lr, nn_cfg.adv_frac = 5, 1e-2, 0.5
    mkt = Market(mkt_cfg)
    print(f"\n=== high-dim n={n} (neural-only, adversarial residual) ===", flush=True)
    solver = SolverNN(mkt, dom_cfg, nn_cfg)
    info = solver.train(ref_solver=None, verbose=True)
    rng = np.random.default_rng(0)
    sim_nn = simulate_paths(mkt, dom_cfg, solver.policy_fn(), rng)
    rng2 = np.random.default_rng(0)
    sim_merton = simulate_paths(mkt, dom_cfg, merton_policy_fn(mkt), rng2)
    rng3 = np.random.default_rng(0)
    sim_fr = simulate_paths(mkt, dom_cfg,
                            robust_fixed_radius_policy(mkt, 0.3), rng3)
    rng4 = np.random.default_rng(0)
    sim_hr = simulate_paths(mkt, dom_cfg, homothetic_policy(mkt), rng4)
    # baseline-measure PnL: same paths, eps_mu = 0 market (drift without
    # worst-case tilt); policies identical (none depends on the measure)
    mkt0_cfg = market_for(n)
    mkt0_cfg.eps_mu = 0.0
    mkt0 = Market(mkt0_cfg)
    rng5 = np.random.default_rng(0)
    sim_nn0 = simulate_paths(mkt0, dom_cfg, solver.policy_fn(), rng5)
    rng6 = np.random.default_rng(0)
    sim_merton0 = simulate_paths(mkt0, dom_cfg, merton_policy_fn(mkt0), rng6)
    rng7 = np.random.default_rng(0)
    sim_fr0 = simulate_paths(mkt0, dom_cfg,
                             robust_fixed_radius_policy(mkt0, 0.3), rng7)
    rng8 = np.random.default_rng(0)
    sim_hr0 = simulate_paths(mkt0, dom_cfg, homothetic_policy(mkt0), rng8)
    g, K = mkt_cfg.gamma, mkt_cfg.K
    out = dict(n=n, train_time=info["elapsed"], input_dim=3 + n,
               final_residual=solver.history["residual"][-1]
               if solver.history["residual"] else None,
               worst_case=dict(
                   nn=pnl_summary(sim_nn, g, K),
                   merton=pnl_summary(sim_merton, g, K),
                   fixed_radius=pnl_summary(sim_fr, g, K),
                   homothetic=pnl_summary(sim_hr, g, K)),
               baseline=dict(
                   nn=pnl_summary(sim_nn0, g, K),
                   merton=pnl_summary(sim_merton0, g, K),
                   fixed_radius=pnl_summary(sim_fr0, g, K),
                   homothetic=pnl_summary(sim_hr0, g, K)))
    path = os.path.join(RESULTS, f"highdim_n{n}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    for meas in ("baseline", "worst_case"):
        print(f"[n={n}] {meas}: nn util "
              f"{out[meas]['nn']['mean_terminal_utility']:.4f} | merton "
              f"{out[meas]['merton']['mean_terminal_utility']:.4f} | fixed-r "
              f"{out[meas]['fixed_radius']['mean_terminal_utility']:.4f} | homoth "
              f"{out[meas]['homothetic']['mean_terminal_utility']:.4f}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cert", nargs="+", type=int, default=[])
    ap.add_argument("--n-high", nargs="+", type=int, default=[])
    args = ap.parse_args()
    # n=6: coarse per-asset axes (3 points each) so the 8-D interpolation
    # intermediates stay within RAM (dense 4-point axes OOM at ~5 GB/array)
    grids = {3: (12, 8, 8, 8, 8), 6: (8, 6, 3, 3, 3, 3, 3, 3)}
    for n in args.n_cert:
        try:
            run_cert(n, grids[n])
        except Exception as e:
            print(f"[n={n}] cert FAILED: {e!r}", flush=True)
    for n in args.n_high:
        try:
            run_highdim(n)
        except Exception as e:
            print(f"[n={n}] highdim FAILED: {e!r}", flush=True)
