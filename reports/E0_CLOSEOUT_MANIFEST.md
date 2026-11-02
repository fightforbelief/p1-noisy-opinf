# E0 closeout manifest

核对时间的 Git HEAD 是 `d6834cc92cf5f7e7220d472941688a4f4ba30c12`，分支 `master`，工作区干净。两套结果都在 `.gitignore` 的 `results/` 下，不受 Git 跟踪。本清单不改写任何 E0 数值。

绝对路径前缀：`C:\Users\hanqi\Desktop\projects\p1-noisy-opinf\`

## results/e0

这次运行写入的 `summary.json` 里 `pass` 为 false，文件内 `git_head` 为 `ae168aaef3a4e8f6aba13daa57a9809177673d0f`。S2 `mc_psd` 在 rho=0 的 `err_O_rel` 仍是 `0.0030837080887182624`。

| 相对路径 | 字节 | SHA-256 |
|---|---:|---|
| `results/e0/meta.json` | 800 | `b596af93b6bfefc3a7a3ed1e5ebe4fa2bc4df4071939f272a020a2cc7aca0dcc` |
| `results/e0/solver_acceptance.json` | 2105 | `cb3d6601a4f0ba6a4b36b901d23e363e472fca135845d204fdbed3a1a84c82ac` |
| `results/e0/s1_true_operator.json` | 1173 | `2e3bba6324fbed6be58cdc04fe0162d2b02f9921d059069808ae8aa14c105daf` |
| `results/e0/zero_noise.json` | 3769 | `f267f5a08549cb90ec201c8f38cab291860e3c670e4ac73b2b9e068fcf64c41f` |
| `results/e0/s2_sigma2.json` | 2080 | `0a4d580e58a07175c5f3c2ab59e7de0929c7e3738e422af924d390f4dd4f0cd7` |
| `results/e0/summary.json` | 1010 | `3251504f1bc645ed64d4beb959ba17ea0f033a7d735a11d9b2525a665c8a99df` |

## results/e0_retry1

`summary.json` 的 `pass` 为 true，文件内 `git_head` 为 `3b8f7d5db2ab927b6e2da6e4c4b1f095cb50f827`。S2 在 rho=0、n_train=10、r=8 的 MC 与 MC-PSD 都是 `6.375601524926835e-12`，`n_clipped=0`。

| 相对路径 | 字节 | SHA-256 |
|---|---:|---|
| `results/e0_retry1/meta.json` | 824 | `4cfc8860855fa0da84f32fe345c6c95f69548d08a6c9a8ea6572891627efc9f3` |
| `results/e0_retry1/solver_acceptance.json` | 2129 | `0467dec85498a6ad0f42c2289a6889c57354c224911adef2071555be4bff65c6` |
| `results/e0_retry1/s1_true_operator.json` | 1197 | `371121a2683a12591d01be9e52843331aa2ffd306e96b8f169697781ada9f20f` |
| `results/e0_retry1/zero_noise.json` | 4098 | `43e43b7aef7322abeb0636c0c21186bca942e7c072b21b0e175af0d15d278c8e` |
| `results/e0_retry1/s2_sigma2.json` | 2104 | `4ce552de5b86134bf62ee2e57135b8bdf39db535253e2f7d4660ef2659300be6` |
| `results/e0_retry1/summary.json` | 1033 | `a0b96c7f1d486c0c9e0b55ced1b57b1b5d50be2e94c83ad23f6c7852f0e4969a` |

这些哈希与 `reports/PROGRESS.md` 里 E0 修订重跑一节的记录一致。进入 E1 之前又逐份重算了一次，12 个摘要全部相同。
