"""Experiment driver: runs the three stages, certifies the neural solver
against references, and writes results (JSON + LaTeX table rows + figures).

Usage:
    python run_experiments.py --stage 1     # Merton smoke test
    python run_experiments.py --stage 2     # + transaction costs (grid ref)
    python run_experiments.py --stage 3     # + analytical worst-case drift
    python run_experiments.py --stage all   # all stages, writes summary table
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from config import stage_config
from market import Market
from grid_solver import GridSolver, simulate_paths
from nn_solver import SolverNN

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def policy_from_grid(solver: GridSolver):
    def fn(t, z, y, pi):
        X = np.concatenate([np.asarray(z).ravel()[:, None],
                            np.asarray(y).ravel()[:, None],
                            np.atleast_2d(pi)], axis=-1)
        return solver.policy_at(t, X)
    return fn


def value_from_grid(solver: GridSolver):
    """Warm-start target: interpolate the reference value on the grid."""
    def fn(t, z, y, pi):
        X = np.concatenate([np.asarray(z).ravel()[:, None],
                            np.asarray(y).ravel()[:, None],
                            np.atleast_2d(pi)], axis=-1)
        return solver.value_at(t, X)
    return fn


def merton_policy_fn(mkt: Market):
    """Policy that moves toward the Merton portfolio at maximum speed."""
    p_star = mkt.merton_pi()

    def fn(t, z, y, pi):
        gap = p_star - pi
        return np.clip(gap / (mkt.cfg.T / mkt.cfg.K), -2.0, 2.0)
    return fn


def pnl_summary(sim, gamma, K):
    """Summary stats of a forward simulation."""
    W, logW = sim["wealth"], sim["z"]
    n_steps_per_year = K
    n_years = W.shape[0] / n_steps_per_year
    rets = np.diff(logW, axis=0)
    ann_ret = (logW[-1].mean() - logW[0].mean()) / n_years
    ann_vol = rets.std() * np.sqrt(n_steps_per_year)
    sharpe = ann_ret / max(ann_vol, 1e-12)
    drawdowns = []
    for w in W.T:
        peak = np.maximum.accumulate(w)
        drawdowns.append(((peak - w) / peak).max())
    return dict(
        mean_terminal_wealth=float(W[-1].mean()),
        std_terminal_wealth=float(W[-1].std()),
        mean_terminal_utility=float(np.mean(
            (W[-1] ** (1 - gamma)) / (1 - gamma))),
        ann_ret=float(ann_ret), ann_vol=float(ann_vol), sharpe=float(sharpe),
        max_drawdown=float(np.mean(drawdowns)),
        turnover=float(sim["turnover"].mean()),
        cost=float(sim["cost"].mean()))


def run_stage(stage: int):
    mkt_cfg, dom_cfg, nn_cfg = stage_config(stage)
    mkt = Market(mkt_cfg)
    out = {"stage": stage,
           "config": {k: v for k, v in vars(mkt_cfg).items()
                      if k not in ("regime_means",)}}

    print(f"\n=== Stage {stage}: "
          f"{'Merton smoke test' if stage == 1 else 'transaction costs' if stage == 2 else 'robustness'} ===")

    if stage == 1:
        # ---- smoke test against the closed-form Merton value ----
        solver = SolverNN(mkt, dom_cfg, nn_cfg)
        rngs = [np.random.default_rng(i) for i in range(4)]
        B = 2048
        # NOTE: with a finite maximum trading rate, the constrained value agrees
        # with the Merton closed form only at the Merton portfolio pi* (the
        # reachable optimum).  We therefore test exactly there.
        t = rngs[0].random(B) * mkt_cfg.T
        z = rngs[1].random(B) * (dom_cfg.z_hi - dom_cfg.z_lo) + dom_cfg.z_lo
        y = np.zeros(B)
        pi = np.tile(mkt.merton_pi(), (B, 1))
        V_ref = mkt.merton_value(t, z)
        tt = torch.tensor(t, dtype=torch.float32)[:, None]
        zt = torch.tensor(z, dtype=torch.float32)[:, None]
        yt = torch.tensor(y, dtype=torch.float32)[:, None]
        pt = torch.tensor(pi, dtype=torch.float32)
        print(f"warm start ({nn_cfg.warm_iters} supervised iters at pi*)...")
        # the Merton closed form is the true value only at pi = pi* (there the
        # constrained agent does not trade), so warm-start exactly there
        solver.warm_start(lambda t, z, y, pi: mkt.merton_value(t, z),
                          iters=nn_cfg.warm_iters,
                          pi_fixed=mkt.merton_pi())
        print("training...")
        info = solver.train(verbose=True)
        with torch.no_grad():
            Vnn = solver.value(tt, zt, yt, pt).squeeze(-1).numpy()
        err = np.abs(Vnn - V_ref)
        # terminal sanity: the hard-terminal layer must reproduce Psi exactly
        with torch.no_grad():
            tT = torch.full_like(tt, mkt_cfg.T)
            VT = solver.value(tT, zt, yt, pt).squeeze(-1).numpy()
        term_err = np.abs(VT - mkt.psi(z)).max()
        out.update(dict(
            merton_l2=float(np.sqrt((err ** 2).mean()) / np.abs(V_ref).mean()),
            merton_linf=float(err.max() / np.abs(V_ref).mean()),
            terminal_err=float(term_err), n_test=B,
            train_time=info["elapsed"]))
        print(f"Merton smoke test: L2 err {out['merton_l2']:.4e}, "
              f"Linf err {out['merton_linf']:.4e}, "
              f"terminal err {out['terminal_err']:.2e}")
    else:
        # ---- grid reference + NN certification + PnL ----
        t0 = time.time()
        grid = GridSolver(mkt, dom_cfg).solve()
        grid_time = time.time() - t0
        print(f"grid reference solved in {grid_time:.1f}s")
        solver = SolverNN(mkt, dom_cfg, nn_cfg)
        print(f"warm start ({nn_cfg.warm_iters} supervised iters on grid)...")
        solver.warm_start(value_from_grid(grid), iters=nn_cfg.warm_iters)
        info = solver.train(ref_solver=grid, verbose=True)
        cert = solver.eval_grid(grid)
        out.update(dict(cert=cert, grid_time=grid_time,
                        train_time=info["elapsed"]))
        print(f"certification: value L2 {cert['value_l2']:.4e}, "
              f"Linf {cert['value_linf']:.4e}, policy err {cert['policy_err']:.4e}")

        # ---- forward PnL: NN policy vs grid policy vs Merton ----
        rng = np.random.default_rng(mkt_cfg.seed)
        sim_nn = simulate_paths(mkt, dom_cfg, solver.policy_fn(), rng)
        rng2 = np.random.default_rng(mkt_cfg.seed)
        sim_grid = simulate_paths(mkt, dom_cfg, policy_from_grid(grid), rng2)
        rng3 = np.random.default_rng(mkt_cfg.seed)
        sim_merton = simulate_paths(mkt, dom_cfg, merton_policy_fn(mkt), rng3)
        out["pnl"] = dict(
            nn=pnl_summary(sim_nn, mkt_cfg.gamma, mkt_cfg.K),
            grid=pnl_summary(sim_grid, mkt_cfg.gamma, mkt_cfg.K),
            merton=pnl_summary(sim_merton, mkt_cfg.gamma, mkt_cfg.K))

        # ---- no-trade region figure ----
        dom = dom_cfg
        # slice at the true optimum (V-argmax on the grid at z=0, y=0), which
        # differs from the myopic Merton pi* under regime-switching factors
        V0g = grid.V[0].reshape(grid.grid_shape)
        zi = np.argmin(np.abs(grid.axes[0] - 0.0))
        yi = np.argmin(np.abs(grid.axes[1] - 0.0))
        i1, i2 = np.unravel_index(np.argmax(V0g[zi, yi]), V0g[zi, yi].shape)
        pi_opt = (grid.axes[2][i1], grid.axes[3][i2])
        p1 = np.linspace(dom.pi_lo, dom.pi_hi, 200)
        zf = np.zeros_like(p1); yf = np.zeros_like(p1)
        p2 = np.full_like(p1, pi_opt[1])
        Xpi = np.stack([p1, p2], axis=-1)
        v_g = policy_from_grid(grid)(0.0, zf, yf, Xpi)[:, 0]
        v_n = solver.policy_fn()(0.0, zf, yf, Xpi)[:, 0]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(p1, v_g, "k-", lw=2, label="grid reference")
        ax.plot(p1, v_n, "r--", lw=1.5, label="neural policy")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel(r"$\pi_1$"); ax.set_ylabel(r"$v_1$ (trading rate)")
        ax.legend(); ax.set_title(f"Stage {stage}: no-trade region (slice $y=0$, "
                                  rf"$\pi_2={pi_opt[1]:.2f}$)")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, f"stage{stage}_no_trade.png"), dpi=150)
        plt.close(fig)

        # ---- wealth path figure ----
        fig, ax = plt.subplots(figsize=(7, 4))
        tt = np.arange(sim_nn["wealth"].shape[0]) / mkt_cfg.K
        for W, style in [(sim_nn["wealth"], "r-"),
                         (sim_grid["wealth"], "k--"),
                         (sim_merton["wealth"], "b:")]:
            ax.plot(tt, W[:, :30], style, alpha=0.25)
        ax.plot(tt, sim_nn["wealth"].mean(axis=1), "r-", lw=2, label="NN policy")
        ax.plot(tt, sim_grid["wealth"].mean(axis=1), "k--", lw=2, label="grid policy")
        ax.plot(tt, sim_merton["wealth"].mean(axis=1), "b:", lw=2, label="Merton")
        ax.set_xlabel("years"); ax.set_ylabel("wealth (mean over paths)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, f"stage{stage}_wealth.png"), dpi=150)
        plt.close(fig)

    path = os.path.join(RESULTS, f"stage{stage}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"results written to {path}")
    return out


def write_summary_table(results: dict):
    """LaTeX rows for the paper's Table 1 (low-dim certification)."""
    rows = []
    for stage in (2, 3):
        if stage not in results:
            continue
        c = results[stage]["cert"]
        p = results[stage]["pnl"]
        rows.append(
            f"2 & {c['value_l2']:.4f} & {c['value_linf']:.4f} "
            f"& {c['policy_err']:.4f} & {p['grid']['mean_terminal_utility']:.4f} "
            f"/ {p['nn']['mean_terminal_utility']:.4f} \\\\")
    with open(os.path.join(RESULTS, "certification_table.tex"), "w") as f:
        f.write("% Auto-generated by run_experiments.py -- copy into "
                "sections/06-experiments.tex (Table 1)\n")
        f.write("% columns: n | L2 value err | Linf value err | policy err "
                "| terminal utility (grid / NN)\n")
        for r in rows:
            f.write(r + "\n")
    print("summary LaTeX rows written to results/certification_table.tex")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["1", "2", "3", "all"])
    args = ap.parse_args()
    stages = [1, 2, 3] if args.stage == "all" else [int(args.stage)]
    all_res = {}
    for s in stages:
        all_res[s] = run_stage(s)
    if args.stage == "all":
        write_summary_table(all_res)
