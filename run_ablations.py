"""Ablation driver: four variants of the Stage-3 configuration.

   base_repro          : reproduce Stage 3 (sanity check of this driver)
   adv_on              : adversarial L-infinity inner loop enabled (n_adv=5)
   no_warmstart        : supervised warm start disabled
   standalone_residual : anchor off, pure HJBI residual (w_anchor=0, w_res=1)

   Each variant reuses the Stage-3 grid reference and the PnL simulation;
   results are written to results/ablation_<name>.json .
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import stage_config
from market import Market
from grid_solver import GridSolver, simulate_paths
from nn_solver import SolverNN
from run_experiments import (RESULTS, merton_policy_fn, pnl_summary,
                             policy_from_grid, value_from_grid)


def run_variant(name, mutate):
    mkt_cfg, dom_cfg, nn_cfg = stage_config(3)
    mutate(nn_cfg)
    mkt = Market(mkt_cfg)
    out = {"variant": name,
           "config": {k: v for k, v in vars(mkt_cfg).items()
                      if k not in ("regime_means",)},
           "nn": {k: v for k, v in vars(nn_cfg).items()}}

    print(f"\n=== ablation [{name}] ===", flush=True)
    t0 = time.time()
    grid = GridSolver(mkt, dom_cfg).solve()
    grid_time = time.time() - t0
    print(f"[{name}] grid reference solved in {grid_time:.1f}s", flush=True)

    solver = SolverNN(mkt, dom_cfg, nn_cfg)
    if nn_cfg.warm_iters > 0:
        print(f"[{name}] warm start ({nn_cfg.warm_iters} iters on grid)...",
              flush=True)
        solver.warm_start(value_from_grid(grid), iters=nn_cfg.warm_iters)
    info = solver.train(ref_solver=grid, verbose=True)
    cert = solver.eval_grid(grid)

    rng = np.random.default_rng(mkt_cfg.seed)
    sim_nn = simulate_paths(mkt, dom_cfg, solver.policy_fn(), rng)
    out.update(cert=cert, grid_time=grid_time, train_time=info["elapsed"],
               pnl=pnl_summary(sim_nn, mkt_cfg.gamma, mkt_cfg.K))

    path = os.path.join(RESULTS, f"ablation_{name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[{name}] done: value_l2={cert['value_l2']:.4f} "
          f"linf={cert['value_linf']:.4f} policy={cert['policy_err']:.4f} "
          f"util={out['pnl']['mean_terminal_utility']:.4f} "
          f"dd={out['pnl']['max_drawdown']:.4f}", flush=True)
    return out


if __name__ == "__main__":
    variants = [
        ("base_repro", lambda c: None),
        ("adv_on", lambda c: setattr(c, "n_adv", 5)),
        ("no_warmstart", lambda c: setattr(c, "warm_iters", 0)),
        ("standalone_residual",
         lambda c: (setattr(c, "w_anchor", 0.0), setattr(c, "w_res", 1.0))),
    ]
    for name, mut in variants:
        try:
            run_variant(name, mut)
        except Exception as e:  # keep going on the remaining variants
            print(f"[{name}] FAILED: {e!r}", flush=True)
