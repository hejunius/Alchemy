"""Configuration for the neural HJBI solver prototype."""
from dataclasses import dataclass, field


@dataclass
class MarketConfig:
    """Market, utility, and cost parameters.

    Notes on units: time is measured in years; pi are portfolio weights on the
    risky assets (risk-free weight is 1 - sum(pi)); v = d pi / dt is the trading
    rate in weight units per year; transaction costs enter as the running cost
    lambda_tc * c(v) with c(v) = kappa1 * |v|_1 + 0.5 * kappa2 * |v|^2
    (the K-term of the companion paper, Eq. (26)).
    """
    T: float = 1.0            # horizon (years)
    K: int = 24               # time steps of the discrete reference scheme
    r: float = 0.01           # risk-free rate
    gamma: float = 3.0        # CRRA risk aversion
    n_assets: int = 2

    # drift: mu(y) = mu_bar + B * y, where y is the 1-D factor (OU, two-regime
    # mean in *simulation*; the PDE uses the baseline mean m_bar)
    mu_bar: tuple = (0.06, 0.08)
    B: tuple = (1.0, 1.5)
    # covariance: Sigma = D C D, D = diag(sigma), C with off-diagonal rho_asset
    sigma: tuple = (0.20, 0.30)
    rho_asset: float = 0.5

    # factor y: dy = -kappa (y - m) dt + nu dW^y (independent of asset shocks)
    kappa: float = 2.0
    m_bar: float = 0.0
    nu: float = 0.3

    # transaction costs (running-cost formulation)
    kappa1: float = 0.005
    kappa2: float = 0.2
    lambda_tc: float = 1.0

    # drift ambiguity (ellipsoidal, worst case closed form):
    # inf_{pi' mu} = pi' mu_hat - eps_mu sqrt(pi' M pi),  M = Sigma + M_reg I
    eps_mu: float = 0.0       # 0 = no robustness (Stage 2), >0 = Stage 3
    M_reg: float = 0.05

    # simulation (PnL evaluation)
    n_years: int = 4
    switch_rate: float = 2.0        # regime switch intensity (calm <-> stress)
    regime_means: tuple = (-0.5, 0.5)
    n_paths: int = 200
    seed: int = 0


@dataclass
class DomainConfig:
    """State domain of the PDE.  z = log wealth."""
    z_lo: float = -0.6
    z_hi: float = 0.6
    y_lo: float = -1.0
    y_hi: float = 1.0
    pi_lo: float = -0.5
    pi_hi: float = 1.5
    vmax: float = 2.0
    # reference grid sizes (z, y, pi_1, ..., pi_n)
    grid: tuple = (16, 10, 12, 12)


@dataclass
class NNConfig:
    width: int = 128
    depth: int = 3
    lr: float = 1e-3
    iterations: int = 3000
    warm_iters: int = 1000    # supervised pre-training iterations
    batch: int = 4096
    n_adv: int = 0            # adversarial inner-loop steps on residual inputs
                              # (0 = plain uniform sampling; ablation later)
    adv_lr: float = 1e-2      # step size of the input-space ascent
    adv_frac: float = 0.5     # fraction of batch kept as adversarial points
    w_boundary: float = 1e-2  # spatial-boundary loss weight
    w_concavity: float = 1e-3 # penalty on V_zz > 0 (CRRA concavity in z)
    w_policy: float = 1e-1    # behavior-cloning weight toward the exact rate
    w_res: float = 0.1        # weight of the HJBI residual term (the anchor
                              # leads in low-dim certification; the residual
                              # enforces smooth off-grid generalization)
    w_anchor: float = 1.0     # reference-anchoring weight (fixed anchor set
                              # drawn from the grid solution; keeps the PDE
                              # training in the certified basin)
    policy_ema: float = 0.99
    log_every: int = 200
    seed: int = 0


def stage_config(stage: int) -> tuple[MarketConfig, DomainConfig, NNConfig]:
    """Stage presets: 1 = Merton smoke test, 2 = + transaction costs,
    3 = + analytical worst-case drift robustness."""
    mkt, dom, nn = MarketConfig(), DomainConfig(), NNConfig()
    if stage == 1:
        # constant-parameter Merton market (B = 0, kappa = 0, nu = 0): the
        # factor y is irrelevant and the Merton closed form is the exact value
        # at every state.  Smooth quadratic cost only: v* = V_pi/(lambda*k2)
        # is Lipschitz, the value function is smooth (representable by tanh
        # nets), and at pi = pi* no trading occurs.
        mkt.kappa1 = 0.0
        mkt.kappa2 = 0.1
        mkt.B = (0.0, 0.0)
        mkt.kappa = 0.0
        mkt.nu = 0.0
        nn.iterations = 4000
    elif stage == 2:
        mkt.kappa1 = 0.05   # L1 cost large enough for a visible no-trade region
        nn.iterations = 4000
    elif stage == 3:
        mkt.kappa1 = 0.05
        mkt.eps_mu = 0.3
        nn.iterations = 4000
    else:
        raise ValueError(f"unknown stage {stage}")
    return mkt, dom, nn
