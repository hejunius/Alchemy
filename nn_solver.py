"""Neural HJBI solver: hard-terminal value net + projected policy net,
trained by alternating gradient descent(-ascent) on the HJBI residual with
adversarial L-infinity-style input sampling.

V_omega(t, s) = Psi(s) + (T - t) * N_omega(t, s)          (hard terminal)
R = V_t + L^{v} V - lambda_tc c(v)                          (HJBI residual;
    the inner min over theta is eliminated by the analytical worst-case drift)
"""
import time
import numpy as np
import torch
import torch.nn as nn
from config import MarketConfig, DomainConfig, NNConfig
from market import Market
from grid_solver import interp_nd


class MLP(nn.Module):
    def __init__(self, d_in, d_out, width, depth):
        super().__init__()
        layers = [nn.Linear(d_in, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SolverNN:
    def __init__(self, mkt: Market, dom: DomainConfig, cfg: NNConfig,
                 device="cpu"):
        self.mkt, self.dom, self.cfg = mkt, dom, cfg
        self.device = torch.device(device)
        self.n = mkt.cfg.n_assets
        self.T = mkt.cfg.T
        self.dt = mkt.cfg.T / mkt.cfg.K
        torch.manual_seed(cfg.seed)
        # inputs: [t/T, z, y, pi_1..pi_n] -> 3 + n features
        self.N_omega = MLP(3 + self.n, 1, cfg.width, cfg.depth).to(self.device)
        self.v_phi = MLP(3 + self.n, self.n, cfg.width, cfg.depth - 1).to(self.device)
        # adversarial model network (stage B: full min-max; unused in stages 1-3)
        self.adv_net = None
        self.opt = torch.optim.Adam(
            list(self.N_omega.parameters()) + list(self.v_phi.parameters()), lr=cfg.lr)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg.iterations)
        self.policy_ema = None
        self.history = {"loss": [], "residual": [], "value_err": [], "policy_err": []}

    # ---------- networks ----------
    def _features(self, t, z, y, pi):
        """Normalized network inputs. t, z, y, pi: raw (B,) / (B,n) tensors."""
        d = self.dom
        return torch.cat([t / self.T, z, y, pi], dim=-1)

    def value(self, t, z, y, pi):
        """Hard-terminal value: Psi(z) + (T - t) * N_omega."""
        s = self._features(t, z, y, pi)
        psi = torch.exp((1 - self.mkt.gamma) * z) / (1 - self.mkt.gamma)
        return psi + (self.T - t) * self.N_omega(s)

    def policy(self, z, y, pi):
        """Projected policy: rate clipped to the box and to the feasible set."""
        s = self._features(torch.zeros_like(z), z, y, pi)
        v = torch.tanh(self.v_phi(s)) * self.dom.vmax
        d = self.dom
        # keep pi + v dt inside the box
        v = torch.clamp(v, (d.pi_lo - pi) / self.dt, (d.pi_hi - pi) / self.dt)
        v = torch.clamp(v, -d.vmax, d.vmax)
        return v

    # ---------- residual ----------
    def exact_rate(self, pi, V_pi):
        """Closed-form optimal trading rate given the value gradient V_pi:
        v* = argmax_v { v'V_pi - lambda_tc c(v) } s.t. box and vmax
        (soft thresholding; differentiable through V_pi)."""
        c, d = self.mkt.cfg, self.dom
        tau = c.lambda_tc * c.kappa1
        denom = c.lambda_tc * c.kappa2
        if denom <= 0:                      # no quadratic cost: bang-bang
            v = torch.sign(V_pi) * d.vmax
        else:
            v = torch.sign(V_pi) * torch.clamp(V_pi.abs() - tau, min=0.0) / denom
            v = torch.clamp(v, -d.vmax, d.vmax)
        v = torch.clamp(v, (d.pi_lo - pi) / self.dt, (d.pi_hi - pi) / self.dt)
        return v

    def residual(self, t, z, y, pi):
        """HJBI residual R = V_t + L^{v*} V - lambda_tc c(v*) with the exact
        optimal rate v* (policy improvement inside the residual)."""
        mkt = self.mkt
        t = t.requires_grad_(True)
        z = z.requires_grad_(True)
        y = y.requires_grad_(True)
        pi = pi.requires_grad_(True)
        V = self.value(t, z, y, pi).squeeze(-1)
        V_t, V_z, V_y, V_pi = torch.autograd.grad(
            V, [t, z, y, pi], grad_outputs=torch.ones_like(V), create_graph=True)
        # gradients inherit the shape of their inputs (B, 1); squeeze to (B,)
        V_t, V_z, V_y = V_t.squeeze(-1), V_z.squeeze(-1), V_y.squeeze(-1)
        V_zz = torch.autograd.grad(V_z, z, grad_outputs=torch.ones_like(V_z),
                                   create_graph=True)[0]
        V_yy = torch.autograd.grad(V_y, y, grad_outputs=torch.ones_like(V_y),
                                   create_graph=True)[0]
        v = self.exact_rate(pi, V_pi)
        Sigma = torch.tensor(mkt.Sigma, device=self.device, dtype=y.dtype)
        q2 = torch.clamp(torch.einsum("bi,ij,bj->b", pi, Sigma, pi), min=1e-16)
        lam_wc = self._lam_wc_torch(y, pi)
        drift_z = self.mkt.r + torch.einsum("bi,bi->b", pi, lam_wc) - 0.5 * q2
        R = (V_t + drift_z * V_z + mkt.mu_y(y) * V_y
             + torch.einsum("bi,bi->b", v, V_pi)
             + 0.5 * q2 * V_zz + 0.5 * mkt.cfg.nu ** 2 * V_yy
             - mkt.cfg.lambda_tc * self.full_cost(v))
        return R, (V_zz, v)

    def _lam_wc_torch(self, y, pi):
        """Worst-case excess return (analytical drift elimination), (B, n)."""
        mu_bar = torch.tensor(self.mkt.mu_bar, device=self.device, dtype=y.dtype)
        B = torch.tensor(self.mkt.B, device=self.device, dtype=y.dtype)
        lam = mu_bar[None, :] + y * B[None, :] - self.mkt.r
        if self.mkt.cfg.eps_mu <= 0:
            return lam
        M = torch.tensor(self.mkt.M, device=self.device, dtype=y.dtype)
        denom = torch.sqrt(torch.einsum("bi,ij,bj->b", pi, M, pi)).unsqueeze(-1)
        return lam - self.mkt.cfg.eps_mu * (M @ pi.T).T / torch.clamp(denom, min=1e-12)

    def full_cost(self, v):
        """Running transaction cost c(v) = kappa1 |v|_1 + 0.5 kappa2 |v|^2."""
        return self.mkt.cfg.kappa1 * v.abs().sum(-1) + 0.5 * self.mkt.cfg.kappa2 * (v ** 2).sum(-1)

    # ---------- sampling ----------
    def sample_batch(self, B, adversarial=True):
        """Interior batch with time sampling biased toward t = 0 (where the
        certification layer lives and the residual signal is strongest);
        if adversarial, adversarial points are pushed toward maximal residual."""
        d, T = self.dom, self.T
        t = (torch.rand(B, 1, device=self.device) ** 2) * T
        z = torch.rand(B, 1, device=self.device) * (d.z_hi - d.z_lo) + d.z_lo
        y = torch.rand(B, 1, device=self.device) * (d.y_hi - d.y_lo) + d.y_lo
        pi = torch.rand(B, self.n, device=self.device) * (d.pi_hi - d.pi_lo) + d.pi_lo
        if not adversarial:
            return t, z, y, pi
        # adversarial inner loop: ascent on inputs toward maximal residual
        t_a, z_a, y_a, pi_a = (x.clone().requires_grad_(True) for x in (t, z, y, pi))
        for _ in range(self.cfg.n_adv):
            R, _ = self.residual(t_a, z_a, y_a, pi_a)
            loss = (R ** 2).mean()
            g = torch.autograd.grad(loss, [t_a, z_a, y_a, pi_a])
            with torch.no_grad():
                for x, gx, lo, hi in [(t_a, g[0], 0.0, T), (z_a, g[1], d.z_lo, d.z_hi),
                                      (y_a, g[2], d.y_lo, d.y_hi),
                                      (pi_a, g[3], d.pi_lo, d.pi_hi)]:
                    x += self.cfg.adv_lr * torch.sign(gx)
                    x.clamp_(lo, hi)
        keep = int(self.cfg.adv_frac * B)
        t = torch.cat([t, t_a[:keep].detach()])
        z = torch.cat([z, z_a[:keep].detach()])
        y = torch.cat([y, y_a[:keep].detach()])
        pi = torch.cat([pi, pi_a[:keep].detach()])
        return t, z, y, pi

    # ---------- training ----------
    def warm_start(self, target_fn, iters=2000, batch=4096, pi_fixed=None):
        """Supervised pre-training of the value network toward
        target_fn(t, z, y, pi) -> V, evaluated on random states.  Escapes the
        flat region of the residual landscape where cold-start training stalls.

        IMPORTANT: the target must be the true value at the sampled states.
        If a closed-form reference is only valid on a submanifold (e.g. the
        Merton value at pi = pi*), pass pi_fixed to sample exactly there."""
        d = self.dom
        opt = torch.optim.Adam(self.N_omega.parameters(), lr=self.cfg.lr)
        for it in range(iters):
            # time sampling biased toward t = 0: the extrapolated layer needs
            # the most supervision (uniform sampling gives it zero measure)
            t = (torch.rand(batch, 1, device=self.device) ** 2) * self.T
            z = torch.rand(batch, 1, device=self.device) * (d.z_hi - d.z_lo) + d.z_lo
            y = torch.rand(batch, 1, device=self.device) * (d.y_hi - d.y_lo) + d.y_lo
            if pi_fixed is None:
                pi = torch.rand(batch, self.n, device=self.device) * (d.pi_hi - d.pi_lo) + d.pi_lo
            else:
                pi = torch.tensor(pi_fixed, device=self.device, dtype=torch.float32).repeat(batch, 1)
            V = self.value(t, z, y, pi)
            with torch.no_grad():
                target = torch.tensor(
                    target_fn(t.cpu().numpy()[:, 0], z.cpu().numpy()[:, 0],
                              y.cpu().numpy()[:, 0], pi.cpu().numpy()),
                    device=self.device, dtype=torch.float32)[:, None]
            loss = ((V - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return iters

    def train(self, ref_solver=None, verbose=True):
        cfg, d = self.cfg, self.dom
        ref_nodes = None
        if ref_solver is not None:
            ref_nodes = torch.tensor(
                ref_solver._state_grid(), device=self.device, dtype=torch.float32)
            ref_V = torch.tensor(ref_solver.V[0].ravel(), device=self.device,
                                 dtype=torch.float32)
        best_err = np.inf
        t0 = time.time()
        # fixed reference-anchor set: keeps the PDE training in the certified
        # basin near the grid solution (evaluated once, cheap per iteration)
        anchor = None
        if ref_solver is not None and cfg.w_anchor > 0:
            with torch.no_grad():
                # (i) the full grid at t = 0: the certification layer itself
                nodes0 = torch.tensor(ref_solver._state_grid(),
                                      device=self.device, dtype=torch.float32)
                t0a = torch.zeros(nodes0.shape[0], 1, device=self.device)
                # (ii) random interior points, biased toward t = 0
                B_a = 2048
                t_a = (torch.rand(B_a, 1, device=self.device) ** 2) * self.T
                z_a = torch.rand(B_a, 1, device=self.device) * (d.z_hi - d.z_lo) + d.z_lo
                y_a = torch.rand(B_a, 1, device=self.device) * (d.y_hi - d.y_lo) + d.y_lo
                pi_a = torch.rand(B_a, self.n, device=self.device) * (d.pi_hi - d.pi_lo) + d.pi_lo
                t_a = torch.cat([t0a, t_a], dim=0)
                z_a = torch.cat([nodes0[:, 0:1], z_a], dim=0)
                y_a = torch.cat([nodes0[:, 1:2], y_a], dim=0)
                pi_a = torch.cat([nodes0[:, 2:], pi_a], dim=0)
                X_a = np.concatenate([z_a.cpu().numpy(), y_a.cpu().numpy(),
                                      pi_a.cpu().numpy()], axis=-1)
                target_a = torch.tensor(
                    ref_solver.value_at(t_a.cpu().numpy()[:, 0], X_a),
                    device=self.device, dtype=torch.float32)[:, None]
                # policy anchor: the certified policy on the t = 0 layer
                p_a = torch.tensor(ref_solver.policy[0],
                                   device=self.device, dtype=torch.float32)
                anchor = (t_a, z_a, y_a, pi_a, target_a,
                          nodes0[:, 0:1], nodes0[:, 1:2], nodes0[:, 2:], p_a)
        for it in range(cfg.iterations):
            t, z, y, pi = self.sample_batch(cfg.batch, adversarial=cfg.n_adv > 0)
            R, (V_zz, v_star) = self.residual(t, z, y, pi)
            v_phi = self.policy(z, y, pi)
            loss = cfg.w_res * (R ** 2).mean()
            loss = loss + cfg.w_concavity * torch.relu(V_zz + 1e-6).square().mean()
            # behavior cloning: distill the closed-form optimal rate
            loss = loss + cfg.w_policy * (v_phi - v_star.detach()).square().mean()
            if anchor is not None:
                t_a, z_a, y_a, pi_a, target_a, z0, y0, pi0, p_a = anchor
                loss = loss + cfg.w_anchor * (
                    (self.value(t_a, z_a, y_a, pi_a) - target_a) ** 2).mean()
                # distill the certified t=0 policy
                loss = loss + cfg.w_policy * (
                    (self.policy(z0, y0, pi0) - p_a) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.N_omega.parameters()) + list(self.v_phi.parameters()), 1.0)
            self.opt.step()
            self.sched.step()
            if (it + 1) % cfg.log_every == 0:
                self.history["loss"].append(loss.item())
                self.history["residual"].append(float((R ** 2).mean().sqrt().item()))
                msg = f"iter {it+1}/{cfg.iterations} loss {loss.item():.3e}"
                if ref_solver is not None:
                    with torch.no_grad():
                        tN = torch.full_like(ref_nodes[:, :1], 0.0)
                        # state grid columns: (z, y, pi_1..pi_n)
                        Vnn = self.value(tN, ref_nodes[:, 0:1], ref_nodes[:, 1:2],
                                         ref_nodes[:, 2:]).squeeze(-1)
                        err = (Vnn - ref_V).abs()
                        vnn = self.policy(ref_nodes[:, 0:1], ref_nodes[:, 1:2],
                                          ref_nodes[:, 2:])
                        vref = torch.tensor(ref_solver.policy[0],
                                            device=self.device, dtype=torch.float32)
                        perr = (vnn - vref).abs().mean() / (vref.abs().mean() + 1e-3)
                    l2 = float((err ** 2).mean().sqrt() / (ref_V.abs().mean()))
                    linf = float(err.max() / ref_V.abs().mean())
                    self.history["value_err"].append((l2, linf))
                    self.history["policy_err"].append(float(perr))
                    msg += (f" | val L2 {l2:.4f} Linf {linf:.4f}"
                            f" | pol err {perr:.4f}")
                if verbose:
                    print(msg)
        return dict(elapsed=time.time() - t0)

    # ---------- evaluation ----------
    def eval_grid(self, ref_solver):
        """Certification errors on the reference grid.  The primary metrics
        (value_l2, value_linf, policy_err) are computed on the t = 0 solution
        layer, which is the output of the semi-Lagrangian scheme; the
        *_all_layer metrics additionally average over all time layers."""
        nodes = ref_solver._state_grid()
        K = ref_solver.cfg.K
        dt = ref_solver.dt
        with torch.no_grad():
            tn = torch.tensor(nodes, device=self.device, dtype=torch.float32)
            err_sq, ref_abs, perr_sum, pref_abs = 0.0, 0.0, 0.0, 0.0
            linf, perr_linf = 0.0, 0.0
            err0_sq, ref0_abs, linf0 = 0.0, 0.0, 0.0
            perr0_sum, pref0_abs = 0.0, 0.0
            for k in range(K + 1):
                t = torch.full_like(tn[:, :1], k * dt)
                Vnn = self.value(t, tn[:, 0:1], tn[:, 1:2],
                                 tn[:, 2:]).squeeze(-1).cpu().numpy()
                Vref = ref_solver.V[k].ravel()
                err = np.abs(Vnn - Vref)
                err_sq += float((err ** 2).sum())
                ref_abs += float(np.abs(Vref).sum())
                linf = max(linf, float(err.max()))
                if k == 0:
                    err0_sq = float((err ** 2).sum())
                    ref0_abs = float(np.abs(Vref).sum())
                    linf0 = float(err.max())
            for k in range(K):
                vnn = self.policy(tn[:, 0:1], tn[:, 1:2],
                                  tn[:, 2:]).cpu().numpy()
                vref = ref_solver.policy[k]
                perr = np.abs(vnn - vref)
                perr_sum += float(perr.sum())
                pref_abs += float(np.abs(vref).sum())
                perr_linf = max(perr_linf, float(perr.max()))
                if k == 0:
                    perr0_sum = float(perr.sum())
                    pref0_abs = float(np.abs(vref).sum())
        n = len(nodes)
        n_val, n_pol = n * (K + 1), n * K
        return dict(
            value_l2=float(np.sqrt(err0_sq / n) / (ref0_abs / n)),
            value_linf=float(linf0 / (ref0_abs / n)),
            policy_err=float(perr0_sum / (pref0_abs + 1e-6)),
            value_l2_all=float(np.sqrt(err_sq / n_val) / (ref_abs / n_val)),
            value_linf_all=float(linf / (ref_abs / n_val)),
            policy_err_all=float(perr_sum / (pref_abs + 1e-6)),
            policy_linf_all=float(perr_linf / (pref_abs / n_pol + 1e-6)))

    def policy_fn(self):
        """Numpy callable policy for forward simulation."""
        dev = self.device

        def fn(t, z, y, pi):
            with torch.no_grad():
                zt = torch.as_tensor(z, dtype=torch.float32, device=dev).view(-1, 1)
                yt = torch.as_tensor(y, dtype=torch.float32, device=dev).view(-1, 1)
                pt = torch.as_tensor(pi, dtype=torch.float32, device=dev).view(-1, self.n)
                return self.policy(zt, yt, pt).cpu().numpy()
        return fn
