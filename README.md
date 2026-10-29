# 神经 HJBI 求解器原型（论文 Part II 的实验代码）

对论文《A Deep Learning Solver for Learning-Aware Robust Portfolio Optimization》
的实验实现：硬终端价值网络 + 投影策略网络 + 对抗 L∞ 残差采样，逐阶段对照参照解验证。

## 三阶段设计（每阶段都有参照解）

| 阶段 | 模型 | 参照解 | 验证内容 |
|---|---|---|---|
| 1 | Merton（无成本、无因子、无鲁棒） | **闭式解** | 价值函数 L2/L∞ 误差 |
| 2 | + 比例/二次交易成本（无交易区出现） | **单调半拉格朗日网格**（Eq. 94） | 价值/策略误差 + 前向 PnL |
| 3 | + 椭圆漂移模糊（解析最坏漂移，Eq. 33） | 半拉格朗日网格 | 鲁棒性下的认证 |

## 运行

```bash
pip install -r requirements.txt
python run_experiments.py --stage 1     # 冒烟测试（~分钟级）
python run_experiments.py --stage 2     # 交易成本认证（~10 分钟）
python run_experiments.py --stage 3     # 鲁棒认证（~10 分钟）
python run_experiments.py --stage all   # 全部 + 生成论文 Table 1 的 LaTeX 行
```

输出到 `results/`：
- `stage{N}.json` — 数值结果（价值误差、策略误差、PnL 汇总、耗时）
- `stage{N}_no_trade.png` — 无交易区（NN 策略 vs 网格参照）
- `stage{N}_wealth.png` — 财富路径（NN / 网格 / Merton）
- `certification_table.tex` — 直接复制进论文 06-experiments.tex 的 Table 1 行

## 代码结构

```
config.py          # MarketConfig / DomainConfig / NNConfig + 三阶段预设
market.py          # 市场系数、CRRA 效用、Merton 闭式解、软阈值最优速率
grid_solver.py     # 半拉格朗日 Bellman(-Isaacs) 递归（向量化多线性插值 + 正交积分）
nn_solver.py       # 硬终端值网络 + 投影策略网络 + 对抗输入采样 + Adam 训练
run_experiments.py # 实验编排、PnL 前向模拟、图表与 LaTeX 输出
```

## 与论文方程/算法的对应

| 代码 | 论文 |
|---|---|
| `SolverNN.value` 硬终端参数化 | Eq. (105)：V_ω = Ψ + (T−t)N_ω |
| `SolverNN.residual` | Eq. (106)：HJBI 残差 |
| `sample_batch` 内环输入上升 | 4.2 节对抗 L∞ 残差采样（Algorithm 1） |
| `Market.optimal_rate` 软阈值 | 论文 Eq. (96) 解析最坏漂移 + 成本结构的闭式最优速率 |
| `GridSolver.solve` | Eq. (94) 单调 Bellman–Isaacs 递归（Theorem 14 的参照实现） |
| `simulate_paths` 两状态市场 | Part I Section 6 的 calm/stress 合成市场 |

## 已知简化（与 Part I 的差异，后续迭代补齐）

1. **成本单通道**：交易成本仅作运行成本（K 项），未同时写入财富动力学
   （Part I Eq. (18) 是双通道）；`config.py` 中可切换
2. **信息状态冻结**：网格参照把学习状态 (θ̂, P) 冻结在稳态值（否则网格维度爆炸），
   NN 输入预留了学习状态通道
3. **协方差模糊**：Stage 3 只做漂移模糊（解析消去）；完整 min-max 需要对抗网络
   θ_ψ（Algorithm 1 的 GDA 上升支，`SolverNN.adv_net` 接口已预留）
4. **ρ=0**：因子冲击与资产冲击不相关（论文的交叉项 L^{XY,θ} 待加）
5. CPU 运行；GPU/MPS 可用 `SolverNN(..., device="cuda")` 切换

## 复现协议

随机种子固定（`MarketConfig.seed`）；前向 PnL 三组策略共用同一路径生成器种子，
保证同一市场路径上公平对比（对应论文"identical paths"协议）。
