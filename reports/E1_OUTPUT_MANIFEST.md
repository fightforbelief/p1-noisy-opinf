# E1 output manifest

运行时 Git HEAD：`d6834cc92cf5f7e7220d472941688a4f4ba30c12`。

绝对路径前缀：`C:\Users\hanqi\Desktop\projects\p1-noisy-opinf\`

`results/` 和 `*.npz` 都在 `.gitignore` 中。下面结果文件不受 Git 跟踪。写出 commit 不会把它们放进版本库。`figures/fig_e1_bias_theory_vs_sim.png` 不在忽略规则里，由 E1 代码提交跟踪。

配置对应：

- `results/e1/`：S1，r=8，rho ∈ {0.02, 0.05, 0.1, 0.2}，n_train ∈ {10, 30, 100}，OLS，replicate 0–199。算子键为 `true`、`lim_rho{rho_index}_n{n_train}`、`rep_rho{rho_index}_n{n_train}_r{replicate:03d}`。rho_index 分别是 2、3、4、5。
- `results/e1_pilot/`：同一系统，rho=0.05，n_train=30，replicate 0–9。这是先导，不是正式统计结果。
- 图使用正式 `summary.csv` 的 A、H 偏差范数。

`summary.csv` 的哈希是修正引号之后、用归档算子重算的文件。`runs.csv`、`operators.npz` 和两份 `meta.json` 没有重写。

| 相对路径 | 字节 | SHA-256 | Git |
|---|---:|---|---|
| `results/e1/meta.json` | 893 | `c74af312a196556e3c8012e04b3f3dc2c3fe73e41897e058d08e4a1e65f994e9` | 不跟踪 |
| `results/e1/operators.npz` | 7592782 | `8bdfba610049f834061b27f26c03c458b135faa5312b69c1d3e80134707dee68` | 不跟踪 |
| `results/e1/runs.csv` | 718323 | `299b0062b9b2254de1c3b63cd6787412c6443f5a70502c7faa2522727e8754af` | 不跟踪 |
| `results/e1/summary.csv` | 11444 | `7257f25d0522a097cd022dece867e1cfd7a04f0a07df9e8c4cb3c71ee1e093a3` | 不跟踪 |
| `results/e1_pilot/meta.json` | 3135 | `a7918359b8573531ce33958fc732897d4019ab46f3754a67c3472a09615d880b` | 不跟踪 |
| `results/e1_pilot/operators.npz` | 37738 | `75d152db5aec4ab09255e66fa4803e5ba4efa5ad0198579ce3f2b179088f12a7` | 不跟踪 |
| `results/e1_pilot/runs.csv` | 3355 | `c705f6529c0cea5b5a1c413b3a34fcb1a7eb1a9ba815d9b766c1ee296e2f391e` | 不跟踪 |
| `results/e1_pilot/summary.csv` | 1644 | `948208d4583f312b3eaf3624d5d9505e5f6ff07a30a8c342fb490c5fd46992d8` | 不跟踪 |
| `figures/fig_e1_bias_theory_vs_sim.png` | 106345 | `857b6bc490405b69b99f726e29b180959e7eb086e523adf881894140f75152d2` | 由本次代码提交跟踪 |
