# E2 output manifest

数值运行时的 Git HEAD 是 `9bc73f190ba92c11d9b969348fd02cc15e6e9639`，分支 `master`。该 HEAD 之上的 E2 代码当时尚未提交。打包提交发生在运行之后，提交信息不含这些结果文件。

绝对路径前缀：`C:\Users\hanqi\Desktop\projects\p1-noisy-opinf\`

`results/` 被 `.gitignore` 忽略。下面的 csv、json、npz 不受 Git 跟踪。五张 png 在 `figures/`，可以进入提交。

环境：Python 3.12.10，机器 `hanqing-win`，`cpu_count=16`。依赖为 joblib 1.6.0、matplotlib 3.11.2、numpy 2.5.4、pandas 3.1.0、pytest 9.2.0、pyyaml 6.0.3、scipy 1.18.1、tqdm 4.70.1。

正式运行墙钟 `1400.5735625999514` 秒，`meta.json` 的 `pass` 为 true，`rows` 为 12600。

## results/e2

| 相对路径 | 字节 | SHA-256 | Git |
|---|---:|---|---|
| `results/e2/meta.json` | 1361 | `c1212827c430f428a97cd1c09b574b7bc80c952760e3d0c2552e84bced93625d` | 未跟踪 |
| `results/e2/runs.csv` | 5478022 | `b3d129340f55087d299a1e8c3d503e4919fac4faef16e7ad7cfd5094880f81ca` | 未跟踪 |
| `results/e2/summary.csv` | 169568 | `38ffc883157f5979d3140400d83942b3bd9c72b0adbeeed8e711ca02677f2efe` | 未跟踪 |
| `results/e2/ridge.csv` | 1943436 | `a066c3084dd732081036b8dcdb448ceaf36754e6eb14608441ef8d63926513b2` | 未跟踪 |
| `results/e2/failures.json` | 4 | `a5338d955b09046ec0b16f3a9625b7955c763aae07dc722e474e6078745f932f` | 未跟踪 |
| `results/e2/operators.npz` | 39773542 | `4023d9a41b61936659856877492549dedadbee6565fe9ca1851ecf7a274463e5` | 未跟踪 |

`failures.json` 的四个字节是空列表 `[]`。

## figures

| 相对路径 | 字节 | SHA-256 | Git |
|---|---:|---|---|
| `figures/fig_e2_bias_sd.png` | 216742 | `359af8fcc7998a09b808018373b63859afd96b2c5152116fe9501c3cac5b4531` | 可跟踪 |
| `figures/fig_e2_rmse_vs_n.png` | 60536 | `32ca6fcc6bcbe8444caa8e679b6766d0b93f6fa2e25e7919710bafb84fe58c79` | 可跟踪 |
| `figures/fig_e2_specrad.png` | 89497 | `9b7fa5da7b146a303d2448505a04ab65a852bb17a4d5a2ec746e3b88944e18b9` | 可跟踪 |
| `figures/fig_e2_ep.png` | 134348 | `f68a04ddf581dc787fe551c74ec0ba22d30d09658d9e97472d4eff0c0e11338b` | 可跟踪 |
| `figures/fig_e2_rollout.png` | 259960 | `8b32a97b90e5fda4302cb42a3169eb6687472496b6e51ffd6c112124b7269691` | 可跟踪 |

## results/e2_pilot

先导通过。两次重复耗时 `6.8555055998731405` 秒。按重复外推 `6169.955039885826` 秒，按配对数加权 `9597.707839822397` 秒。

| 相对路径 | 字节 | SHA-256 | Git |
|---|---:|---|---|
| `results/e2_pilot/meta.json` | 1448 | `d688c12433f30069b307158397dd6209e5d3de4984c33cc19e0da2b3a9f930c8` | 未跟踪 |
| `results/e2_pilot/runs.csv` | 6585 | `9066f5fccdc5108469c653d737b3f4506a81d1d0cabb0a6af288d75759b79f11` | 未跟踪 |
| `results/e2_pilot/summary.csv` | 11058 | `5e4ccffb958c5ab1894e6b3ac18b2a13876ca9cd11161c197e63f09fff670744` | 未跟踪 |
| `results/e2_pilot/ridge.csv` | 2264 | `2429682c2a1a28477732eea8328136af440a02449681910004f8d96c264c56ef` | 未跟踪 |
| `results/e2_pilot/failures.json` | 4 | `a5338d955b09046ec0b16f3a9625b7955c763aae07dc722e474e6078745f932f` | 未跟踪 |
| `results/e2_pilot/operators.npz` | 47322 | `536bf777e7ebea84bcf97f1cfd4640e92c65e4544d2523d432d19859a57c8c16` | 未跟踪 |

运行结束后重新核对：`results/e0/summary.json` 仍是 `3251504f1bc645ed64d4beb959ba17ea0f033a7d735a11d9b2525a665c8a99df`，`results/e0_retry1/summary.json` 仍是 `a0b96c7f1d486c0c9e0b55ced1b57b1b5d50be2e94c83ad23f6c7852f0e4969a`，`results/e1/summary.csv` 仍是 `7257f25d0522a097cd022dece867e1cfd7a04f0a07df9e8c4cb3c71ee1e093a3`。
