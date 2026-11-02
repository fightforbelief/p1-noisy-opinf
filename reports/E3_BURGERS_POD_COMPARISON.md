# E3 Burgers POD 七估计器比较

状态：`E3_EXPERIMENT_COMPLETE_WITH_SCIENTIFIC_DISCREPANCY`

正式网格、归档、四张图、测试和独立复算都已完成。下面的数字来自 `results/e3/` 的归档，不是事先填入的预期。E3 的参考是各基上的 `O_clean`，不是 S1 的 `O_true`。没有启动 E4–E8。

## 1. 本机与 Git

- 项目根目录：`C:\Users\hanqi\Desktop\Projects\Research_Portfolio_2026\p1-noisy-opinf`
- 机器：`hanqing-win`
- 分支：`master`
- E2 冻结时的 HEAD：`d3d9d858d1f6167dcf04851e67736c157c4bc685`
- E3 正式运行前的提交：`ed0006a4d3d37e81091c86d443423cc2be266903`，说明是 `feat: prepare reproducible E3 experiment`
- 正式命令：`scripts\e3_compare.py run`
- 运行记录中的 HEAD 与上面这个提交一致。开始和结束时，执行文件的 SHA-256 没有变化。
- Python 3.12.10。numpy 2.5.4，scipy 1.18.1，pandas 3.1.0，matplotlib 3.11.2。
- 工作区在运行开始时不是绝对干净：`manifests/pre_implementation_baseline.json` 有路径迁移修改，`uploads/` 未跟踪。这两项没有进入 E3 提交，也没有被当成需要删除的文件。

## 2. E2 审计

接受 `E2_AUDIT_PASS_WITH_LIMITATIONS`。E2 冻结为 S1 精确二次系统上的证据。没有重跑 E2，也没有改 E2 的数据、代码、图和统计结论。

遗留限制保持为历史事实：

- `RUNTIME_SOURCE_SNAPSHOT_NOT_FULLY_VERIFIED`
- `PER_TRAJECTORY_RECALCULATION_NOT_AVAILABLE`
- IV weak identification：`NOT_IDENTIFIED`

E2 审计报告已进入 pre-run 提交：`reports/E2_POSTRUN_AUDIT_2026-11-02.md`。

## 3. 运行版本

正式运行使用的源码哈希与当前工作区一致：

| 文件 | SHA-256 |
|---|---|
| `configs/e3.yaml` | `ee240194eea9675300d801d352403a6bff1b371a544ba9237ea6fa994e81010f` |
| `src/noisyopinf/e3.py` | `b8f71d5277745f056fdab0b57d318b4c7f3f63fdeac49cce0b9bf7be141570e6` |
| `src/noisyopinf/estimators.py` | `a4716e630850e93bafc0acb55197ad4e7dfef0758c8ba4c06a32ccccc0e381c4` |
| `src/noisyopinf/theory.py` | `264c777fb42388a298ceda6a2fd600fc55ac43076beb4f1d46af0ede16975a13` |

`estimators.py` 和 `theory.py` 的哈希与 E2 审计时相同。已验收的数学实现没有改。

S2 参数哈希：`219c52f59203ad7fb468cec9e59f4e47d58f632105a0056e54cef5d252ba4dfa`。

## 4. 网格

S2 Burgers，r=8，n_train=30，n_val=20，n_test=50，预测 200 步，另记 h=10 和 h=50。rho 为 0、0.01、0.02、0.05、0.1、0.2。basis 为 clean 与 noisy。R=20。experiment_id=3。噪声键是 `(2, 4, 3, rho_index, replicate, traj)`，验证轨迹下标加 100。

每个 rho × basis × replicate 有 15 行：oracle、ols、ridge 各 1 行，sigma2 来源记为 `na`；mc、mc_psd、iv、iv2 各有 true、est_trailing、est_gd 三行。

正式结果 `runs.csv` 有 3600 行，全部 `status=ok`。`failures.json` 是空数组。来源计数是 true、est_trailing、est_gd 各 960 行，na 720 行。

先导在 `results/e3_pilot/`，rho=0.05，replicate 0 和 1，60 行，失败 0。墙钟 17.28277450008318 秒，单次 `(rho, replicate)` 平均 16.967052199994214 秒，外推全网格 135.7364175999537 秒。正式运行实际用了 302.1770353000611 秒。两者都低于 4 小时，R 保持 20，没有去掉 est_gd，也没有删 rho 或估计器。

pytest：115 passed，0 failed。其中包含原有 106 项和新增 9 项。默认的 `pytest-of-hanqi` 临时目录在这台机器上不可访问，所以这次使用 `--basetemp %LOCALAPPDATA%\e3-pytest-tmp`。这是环境权限，不是测试失败。

## 5. 两种 POD 基和参考算子

clean basis 由 30 条干净训练全阶轨迹一次 SVD 得到，所有 rho 和重复共用。noisy basis 由该次重复的全阶带噪训练轨迹得到。两边的 `O_clean` 都是把干净训练轨迹投到该基之后做 OLS。Oracle 用干净投影，其余六个估计器用带噪投影。sigma2 来源不重建基、不更换训练噪声、不更换参考。

审计核对了：

- 基的列正交误差在 1e-15 量级。
- clean basis 的哈希在全部重复上相同，并且等于缓存里的 `V_clean`。
- noisy basis 的哈希随重复改变，并且不等于 clean basis。
- 同一基、同一次重复的 15 个估计器共用一个参考哈希。
- rho=0 时 clean 与 noisy 的基、参考和阈值一致到记录精度。rho>0 时两边的参考哈希不同，算子矩阵没有跨基相减。
- noisy basis 上，重复之间的参考不同，因此 `bias_O_rel` 和 `sd_O_rel` 记为 NaN。这是参考系变了，不是估计失败。clean basis 的参考跨重复相同，偏差和标准差只在这个基上报告。

发散阈值是该基上干净投影训练轨迹的 `1e3 * max ||q||_∞`。倍数沿用 E2 的 1e3。clean basis 的阈值对全部 rho 和重复都是 `13167.147110481143`。noisy basis 的阈值随重复变化，范围是 `13161.335766485108` 到 `13175.442407245675`。同一个 `(rho, basis, replicate)` 里，七个估计器和三种 sigma2 来源使用同一个阈值。

## 6. 三种 σ²

σ² 在全阶带噪训练快照上估计，N=256，K_tot=6030，beta=256/6030<1，两种方法的 applicable 都是 true。没有用 true 替换估计值。

rho=0 时 true 为 0。trailing median 的估计是 `2.165803e-32`，Gavish–Donoho 是 `1.099842e-21`。相对误差按约定是 NaN，没有改写成 0。这两种非零的数值估计没有让需要 σ² 的估计器失败；MC 在三种来源下的 `err_O_rel` 中位数都是 `5.592886e-14`。

rho>0 时两种估计都偏高。相对误差中位数：

| rho | trailing | Gavish–Donoho |
|---|---:|---:|
| 0.01 | 0.036939 | 0.045807 |
| 0.02 | 0.032065 | 0.041272 |
| 0.05 | 0.027597 | 0.033894 |
| 0.1 | 0.022646 | 0.029312 |
| 0.2 | 0.018125 | 0.023393 |

trailing median 在每一档都更接近 true。这个优势没有变成明显更好的算子。noisy basis 上，MC 的 est_trailing 相对 true 的配对中位数在 rho=0.05 是 `2.118219e-05`，在 rho=0.2 是 `3.683574e-04`。rho=0.2 的 bootstrap 区间是 `[2.650527e-04, 5.100585e-04]`，整段为正，所以 est_trailing 比 true 略差，但差值比 MC 本身的 `err_O_rel`（0.02273）小两个数量级。IV 和 IV2 的三种来源在记录精度内几乎重合，rho=0.2 的来源配对差约 1e-10。

## 7. 七个估计器

下表是 noisy basis 上的 `err_O_rel` 中位数。需要 σ² 的方法用 est_trailing，其余用 na。配对差是该方法减去同一重复的 OLS，bootstrap 是 2000 次、`rng(0, 77)`。

| rho | oracle | ols | ridge | mc | mc_psd | iv | iv2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.00845778 | 5.59289e-14 | 2.51504e-12 | 0.000680154 | 0.000930351 |
| 0.01 | 0 | 0.000462477 | 0.0085291 | 0.000345523 | 0.000345523 | 0.000746545 | 0.00101011 |
| 0.02 | 0 | 0.00136353 | 0.00878743 | 0.00064828 | 0.00064828 | 0.000977307 | 0.00115976 |
| 0.05 | 0 | 0.00738635 | 0.0116217 | 0.00201324 | 0.00201324 | 0.00232285 | 0.00220681 |
| 0.1 | 0 | 0.0278851 | 0.0313127 | 0.00640021 | 0.00640021 | 0.00626531 | 0.00472478 |
| 0.2 | 0 | 0.0914012 | 0.0830942 | 0.0227309 | 0.0227309 | 0.0221881 | 0.0144788 |

rho≥0.02 时，MC、IV、IV2 相对 OLS 的 bootstrap 区间整段为负。rho=0.05 的配对中位数是 MC `-0.0054879`、IV `-0.00513358`、IV2 `-0.00515848`。rho=0.01 时只有 MC 整段优于 OLS（配对 `-0.000116519`，区间 `[-0.00012955, -9.91273e-05]`）；IV 和 IV2 整段差于 OLS。

误差主要在二次块。rho=0.2 的 `err_H_rel` 中位数是 OLS 2.79412、MC 1.21453、IV 1.22125、IV2 0.770255，而 `err_A_rel` 分别是 0.0782214、0.00739866、0.00691097、0.00460474。

clean basis 上可以计算偏差。rho=0.2、est_trailing 或 na：

| 估计器 | bias_O_rel | sd_O_rel |
|---|---:|---:|
| ols | 0.0898168 | 0.0162964 |
| mc | 0.00503466 | 0.0231807 |
| iv | 0.00477412 | 0.0219366 |
| iv2 | 0.00400919 | 0.0142904 |
| ridge | 0.08224 | 0.0113846 |

MC、IV、IV2 把相对 `O_clean` 的偏差降到 OLS 的大约二十分之一，但没有降到 0。标准差没有出现 E2 在 n_train=10 上那种几十到几百的放大。rho=0.05 时 OLS 与 MC 的 `sd_O_rel` 分别是 0.0019444 和 0.00198999。

rho=0 的 IV 误差 0.000680154 不是公式被判错的证据。此时 OLS 与 `O_clean` 一致，IV 仍丢掉端点，而且 POD 坐标里的 Burgers 不是精确二次系统。

## 8. Ridge

每个 `(rho, basis, replicate)` 只选一次 λ。13 个验证分数都在 `ridge.csv`，共 3120 行。选择用带噪验证轨迹，并列取更大的 λ。独立复算用保存的分数重新执行 `larger_lambda_index`，与归档选择一致。

rho≤0.05 时，clean 和 noisy 的选中 λ 都是 0.001。rho=0.1 的中位数是 0.01，范围 0.001 到 0.01。rho=0.2 的中位数是 0.055，范围 0.01 到 0.1。

## 9. 全阶预测、发散和投影下限

预测从干净测试初值出发，在该基的降阶坐标中迭代，再重构回全阶，与干净全阶测试轨迹比较。发散记为 inf，没有使用有限惩罚。

noisy basis、est_trailing 或 na 的 `rollout_med` 中位数和发散比例中位数：

| rho | oracle | ols | ridge | mc / mc_psd | iv | iv2 | 发散 oracle, ols, ridge, mc, iv, iv2 |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.134232 | 0.134232 | 0.118321 | 0.134232 | 0.135997 | 0.136291 | 0.02, 0.02, 0, 0.02, 0.02, 0.02 |
| 0.05 | 0.134338 | 0.131516 | 0.126288 | 0.13508 | 0.138485 | 0.139167 | 0.02, 0.02, 0, 0.02, 0.02, 0.02 |
| 0.1 | 0.134338 | 0.183098 | 0.192523 | 0.147128 | 0.148226 | 0.146905 | 0.02, 0.10, 0, 0.04, 0.02, 0.04 |
| 0.2 | 0.134768 | 0.397238 | 0.349525 | 0.267817 | 0.246192 | 0.196258 | 0.02, 0.24, 0.05, 0.17, 0.12, 0.08 |

图上的 projection floor 是各重复把全部干净测试快照堆在一起之后的投影误差，再取中位数。noisy basis 上它从 rho=0 的 0.132764 升到 rho=0.2 的 0.133074。Oracle 的全视界中位数贴着这条线，约为 0.134。这是闭合误差，不是估计器没有拟合到 `O_clean`。

这条池化下限不是每条轨迹的下限。rho=0、clean、replicate 0、oracle 的逐轨迹投影误差中位数是 0.10330631966675169，而该重复的全视界误差中位数是 0.1342323196314346。Ridge 在低噪声时的全视界中位数约 0.118，落在这两个不同汇总之间。不能把 0.133 理解成每个方法都必须达到、也不能理解成任何方法都不能低于它的那一个数。

逐轨迹归档 `rollout_details.npz` 有全部 3600×50 条测试轨迹的全视界、h10、h50、发散、首次发散步、阈值和投影误差。独立复算用这些数重算了每次重复的中位数、有限均值和发散比例，与 `runs.csv` 一致。没有重跑 200 步预测。

## 10. 四张图

- `figures/fig_e3_operator_err.png`，SHA-256 `8f606e7ab28465b3a1063a4617fee0a02208fed21901d55bd677dec017730cae`，85071 字节
- `figures/fig_e3_rollout.png`，SHA-256 `353a26fefce0a08e5e18a351a3b9f61333aa69d7ce2409e07a5c5db5e1c0a413`，137954 字节
- `figures/fig_e3_basis_effect.png`，SHA-256 `1ff2654fee4011201ea954f8bc122cd1d6bcce246a9fdc4c7ee34fa3763711ee`，69041 字节
- `figures/fig_e3_sigma_source.png`，SHA-256 `bb423b2ce4d36a7d72ca2272eece5909537fd8b22dc67b534d1222f1d851b1cd`，92882 字节

算子图的纵轴是对数。oracle 和 OLS 在 rho=0 的相对误差是 0，不能画在对数轴上，因此该点不在图中；表里保留 0。发散没有出现 inf，所以预测图没有 inf 标注。σ² 相对误差在 rho=0 是 NaN，因此右图没有 rho=0 的点。

## 11. 十二个科学问题

1. MC、IV、IV2 能否降低相对 `O_clean` 的算子误差？rho≥0.02 可以，配对区间整段低于 OLS。rho=0.01 只有 MC 可以，IV 和 IV2 更差。谁也没有把误差降到 oracle 的 0。
2. S1 的小样本不稳定是否还在？在 S2 的 n_train=30 上没有。clean basis 的 `sd_O_rel` 最大约 0.023，MC-PSD 没有出现 E2 那种中位数几百的失败。这里只有一个样本量，不能把“随 n 消失”写成已观察到的事实。
3. noisy basis 影响多大？对 `err_O_rel` 和全视界预测都很小。rho=0.2 的 MC 中位数在 clean 是 0.0227569，在 noisy 是 0.0227309。两边各用各的 `O_clean`，没有把两个坐标里的矩阵相减。
4. σ² 来源如何改变 MC/IV？MC 有可检出但很小的差别，rho=0.2 时估计方差比 true 差大约 0.00037。IV 和 IV2 几乎不变。
5. 谁的方差估计更准？trailing median。它在每个 rho>0 的相对误差中位数都低于 Gavish–Donoho，而且两者都偏高。
6. 方差更准是否带来更好的算子？在这次网格里没有实质好处。来源造成的算子误差差比噪声水平造成的误差小很多。
7. MC-PSD 何时截断？`n_clipped` 在 clean 和 noisy、全部 rho 和重复上都是 0。MC 与 MC-PSD 的误差中位数相同。这次没有截断带来的额外偏差。
8. IV2 是否比 IV 更稳？rho≥0.05 时 IV2 的算子误差、全视界误差和发散比例都更低。rho≤0.02 时 IV2 更差，因为它丢掉更多端点，而噪声还不足以补偿这个损失。
9. 闭合误差是否限制预测？是。Oracle 已经是 `O_clean`，全视界中位数仍约为 0.134，与池化投影误差同一量级。参考算子的能量违反中位数本身就是 0.04424，不是 0。
10. 更低的算子误差是否对应更低的全阶预测误差？只在离开投影带之后大致如此。rho=0.2 时 IV2 的算子误差和预测误差都最低。rho≤0.05 时各方法的预测都挤在 0.12 到 0.14，算子误差的排序不能决定预测排序。Ridge 的算子误差大于 OLS，低噪声时的预测中位数却更低，发散也更少。
11. 有没有算子误差小但预测仍发散的方法？有。Oracle 的算子误差是 0，发散比例中位数仍是 0.02。rho=0.2 的 MC 算子误差 0.02273，低于 Ridge 的 0.08309，但发散比例是 0.17，高于 Ridge 的 0.05。
12. 哪些结论从 S1 转到了 S2？转到 S2 的是：中等和高噪声下 MC、IV、IV2 比 OLS 更接近无噪声 OLS；高噪声下 IV2 优于 IV；OLS 的谱半径略偏小，二次块比线性块更脆。没有转到 S2 的是：误差的量级、E2 的排名、n=10 那种方差爆炸、MC-PSD 截断，以及“算子误差很大但预测几乎不变”。S2 上算子误差整体小一个数量级以上，预测则被约 0.13 的投影误差托住。

## 12. H3 与 H4

H3 说 OLS 让 A 的谱半径偏小，MC/IV 能纠正。noisy basis、est_trailing 上，参考谱半径中位数是 0.998023。OLS 的 `specrad_err` 在 rho=0.2 是 `-0.00172346`，方向符合“偏小”，但只占半径的大约千分之二。MC 在同一格点是 `+0.000226136`，IV2 是 `+7.31994e-05`。它们的绝对谱误差小于 OLS，但落在参考的另一侧，不是把半径恢复到 `O_clean`。H3 作为可观的纠正没有得到支持。

H4 说 OLS 的 H 更破坏能量结构，MC/IV 破坏较少。`O_clean` 自己的 `ep_violation` 中位数约为 0.04424。rho=0.2 时 OLS 是 0.217601，MC 是 0.130618，IV 是 0.111307，IV2 是 0.0908322。高噪声下 MC/IV 的违反小于 OLS，也仍大于参考。rho=0.01 时 OLS 是 0.0437313，略低于参考 0.0442463，所以“OLS 违反更大”并不是每一档都成立。

这两条是 E3 对应的原假设。实验本身完整，假设没有按原表述成立，因此状态是 `E3_EXPERIMENT_COMPLETE_WITH_SCIENTIFIC_DISCREPANCY`。

## 13. 异常和未识别的机制

- 没有失败拟合，没有 NaN 算子，没有 inf 全视界中位数。
- rho=0 的 σ² 相对误差是 NaN。这是 true=0 且估计值非 0 的既定语义。
- noisy basis 的跨重复偏差和标准差是 NaN，因为每次重复的 `O_clean` 不同。机制是坐标改变，不是未识别的数值故障。
- Oracle 仍有 2% 的测试轨迹发散。归档里有首次发散步，本次没有再给它指定一个闭合机制。
- Ridge 在低噪声时预测更好、算子更差。本次只记录这个分离，不把它解释成新的正则理论。
- E2 的 IV weak identification 没有在 E3 上重开。E3 的 IV 误差相对 `O_clean`，不能用来判断 S1 的工具变量是否弱识别。

## 14. 独立复算与测试

`scripts/e3_compare.py audit` 对正式目录返回 `AUDIT_PASS`。复算使用保存的算子、基、训练和测试状态、逐轨迹误差、ridge 分数和 σ² 数值，重算了算子误差、能量违反、投影下限、汇总分位数、clean basis 的偏差和标准差、两组 bootstrap、ridge 选择、逐轨迹预测汇总，以及 σ² 相对误差。没有重拟合，也没有重跑 rollout。

## 15. 结果清单

正式结果目录：`C:\Users\hanqi\Desktop\Projects\Research_Portfolio_2026\p1-noisy-opinf\results\e3`

这些文件被 `.gitignore` 忽略，没有放进 Git。哈希来自 `results/e3/output_manifest.json`。

| 相对路径 | 字节 | SHA-256 |
|---|---:|---|
| `results/e3/runs.csv` | 1634909 | `212f9d27b201d70ab3e5803afc8d88d6f313c3fc9cc27735de13b8a41858c4db` |
| `results/e3/summary.csv` | 260178 | `55de8af435495db0222e3c689f1db1623af61c451405b10914d3d6a086ea0c2d` |
| `results/e3/operators.npz` | 11678109 | `47b058442acd534872b04777548d905e636179da3b3032ad22206f39a679814b` |
| `results/e3/bases.npz` | 1920194 | `8e23511c5b501d43cd85cf101ed83b8a6e5a00ebcda9e6fdf81211f8c7e44baa` |
| `results/e3/rollout_details.npz` | 2953072 | `67f9ba095cd3ad2e083c06a938e7894b4e5c019431550576eb257c9d980ed099` |
| `results/e3/sigma2_estimation.csv` | 22993 | `1801d192c413241a069d078baee03d175c6b6e79aa3a3312740c555d121f5448` |
| `results/e3/ridge.csv` | 216199 | `397cb3849f1806f9d1e8c8306d29503bda324e8987747a28f8c0f9f06d261de5` |
| `results/e3/basis_audit.csv` | 53436 | `58db5439f4cc82ae95a834e6acc4c9bc511612cb6037bb363bc30ff19a0f591d` |
| `results/e3/meta.json` | 1585 | `81267a208c38a9c0fd6f2096752c008f464a3a45de1859200a64418e03bd3938` |
| `results/e3/failures.json` | 4 | `a5338d955b09046ec0b16f3a9625b7955c763aae07dc722e474e6078745f932f` |
| `results/e3/runtime_start.json` | 3442 | `847b8017d0e83de3391662fe18e36ab9cebe6a6c0d138483ede4798c6bdacfc9` |
| `results/e3/runtime_end.json` | 3440 | `fad3f789aa02509909b9d58bc65f0f7b626264e53839270ece3705a2eb225aff` |
| `results/e3/rollout_details.codes.json` | 291 | `418238354800e61d52986d8c04e63b6fe149685894c1fe9ddf11dfd66ff4491b` |

先导在 `results/e3_pilot/`。状态缓存是 `data/e3_s2_states.npz`。

## 16. 提交边界

pre-run 提交包含 E3 配置、源码、脚本、测试、先导说明和 E2 审计报告。结果提交只包含本报告、输出清单、四张图和 `PROGRESS.md` 的 E3 节。

没有修改 `estimators.py`、`theory.py`、`features.py`、`noise.py`、`seeds.py`、`pod.py`、物理求解器、MC-PSD 的零噪声特例、IV 定义、AR(1)、κ、MP 校准或 E2 的阈值倍数。

没有启动 E4、E5、E6、E7、E8。没有 push。
