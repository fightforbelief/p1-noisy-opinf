# E4 最终科学审计与冻结

状态：`E4_AUDIT_PASS_WITH_SCIENTIFIC_DISCREPANCY`

H6 判断：`H6_PARTIALLY_SUPPORTED`

本轮只读 `results/e4/`、`data/e4_s2_states.npz` 和已提交源码。没有重跑 E4，没有改估计器，没有启动 E5–E8，也没有覆盖原始结果。归档完整。逐条轨迹的全视界滚动误差没有低于该轨迹自己的投影误差。汇总表里 22/120 个 `rollout < proj_floor` 来自两种不同的汇总，不是逐条下界被突破。

H6 的第一句在预注册的误差曲线汇总上成立：四个估计器的 clean-validation 最优秩都不随 rho 上升。第二句不成立：四个预注册的 κ 都没有在全部估计器和全部 rho 上把这个最优秩预测到 ±2 个模态以内。rho=0.01 和 rho=0.02 的 `r_snr=16` 是 `RIGHT_CENSORED_AT_RANK_16`，不能当成精确阈值。IV2 高秩发散的机制是 `MECHANISM_NOT_IDENTIFIED`。

`r*_test` 在全文中都是 `DESCRIPTIVE_ONLY`。它不参与部署秩的选择。

## 1. Executive Summary

1. 原始 E4 归档与运行时记录的 SHA-256 一致。1200 条运行的状态都是 `ok`，唯一键 1200，失败列表为空。
2. `proj_floor` 是 50 条干净测试轨迹叠在一起之后的 Frobenius 相对误差，记为池化投影参考。`rollout_med` 是先对每条轨迹算全视界误差、再取中位数。二者使用同一组 noisy POD 前缀、同一批干净测试轨迹、同一时间窗和同一个相对 Frobenius 范数，但汇总顺序不同。
3. 用 `bases.npz` 的秩前缀和 `data/e4_s2_states.npz` 的干净测试轨迹独立重算后，60000 条轨迹记录里，有限滚动误差低于该轨迹自身投影误差的次数是 0。因此 E4 不标记为 `E4_AUDIT_BLOCKED`。
4. 22/120 个汇总格的滚动中位数低于池化参考，全部出现在 rho≤0.05。这与逐条下界相容：逐条投影误差的中位数低于池化参考，滚动中位数可以落在两者之间。
5. noisy POD 只保存了 16 列。rho=0.01 和 rho=0.02 上，四个 κ 的 16 个已保存模态全部通过阈值，10 次重复都是如此。这里的 16 是右删失，不是恰好等于 16 的估计。
6. 从原始 `lam_*` 和对应重复的 `est_trailing` 重算后，replicate 级平均绝对秩差与 ±2 命中数是：κ=1 为 5.98 和 24/200；κ=4 为 5.00 和 24/200；κ=16 为 3.74 和 31/200；κ=64 为 2.26 和 106/200。κ=64 在 rho=0.05、0.1、0.2 上的平均绝对差是 2.25、1.75、0.05。κ=64 只是这四个预注册常数里最接近的一个，不是已经验证的通用最优常数。
7. Clean validation 有 13/20 个组合 10 次重复选同一秩；noisy validation 有 11/20。Ridge、rho=0.2 的 noisy validation 误差曲线选 r=16，10 次里只有 5 次相同，重复秩的 lower central median 是 12。这是两个不同的统计量。
8. 相对差距小于 5% 的 “close” 标记不在 `configs/e4.yaml` 和运行前协议里。它是事后探索指标。
9. MC 在 r=16 的五个 rho 上，校正 Gram 的最小特征值都为负，`n_clipped=0`，条件数约 1e11–7e12，`err_H` 约 45–187，谱半径离开参考，`diverged_frac=1`，滚动中位数为 inf。这是数值不定性与发散同时出现的证据，还不足以把条件数说成唯一原因。
10. E4 归档没有保存 IV2 的工具变量矩阵、一阶段残差、一阶段 F、奇异值或交叉矩。`instrument_cond` 和 `first_stage_r2_median` 在拟合时计算过，但没有写入运行行。机制保持 `MECHANISM_NOT_IDENTIFIED`。

## 2. E4 实验设计和运行版本

设计来自 `p1-noisy-opinf-docs/04_实验设计.md` 的 E4，以及 `02_数学与估计器规格.md` 第 8 节。系统是 S2，`n_train=30`，`n_val=20`，`n_test=50`，基只使用 noisy，`r∈{4,6,8,10,12,16}`，`rho∈{0.01,0.02,0.05,0.1,0.2}`，估计器是 `ols`、`ridge`、`mc`、`iv2`，σ² 只用 `est_trailing`，R=10。一次全阶噪声、一次截到 16 的 SVD，候选秩取该基的前 r 列。每个秩有自己的干净投影 OLS 参考，不跨秩相减算子。

三个秩分开保存：

- `r*_val`：干净验证初值对干净验证真解。它是 oracle，不能直接部署。
- `r*_noisyval`：带噪验证初值对带噪验证状态。它不使用干净验证轨迹。
- `r*_test`：由已保存的测试误差描述，标记 `DESCRIPTIVE_ONLY`。

运行前写进 `configs/e4.yaml` 和 `src/noisyopinf/e4.py` 的汇总是：每个重复先对验证轨迹取中位数，再对 10 个重复中位数取中位数，然后选最小的秩；严格并列取较小秩。SNR 图的汇总是非空原始 `r_snr` 的 lower central median：偶数个时取中间两个里较小的那个。Ridge 的 λ 仍是 13 个 `10^k`（k 从 −10 到 2），并列取更大的 λ。

| 对象 | 值 |
|---|---|
| 当前审计开始时的分支 / HEAD | `master` / `c5ec0761080e3ae70c8d46bccbefa40bd8187e24` |
| E4 运行前提交 | `8f90c7e6c85252a606767c1920b44d6902caf1ab`，`feat: prepare reproducible E4 rank experiment` |
| 结果记录提交 | `311f04281a378921ac06d34b19a39b3a0ba4c51c` |
| 上一轮审计提交 | `c5ec0761080e3ae70c8d46bccbefa40bd8187e24` |
| `meta.json` 的 `git_head` | `8f90c7e6c85252a606767c1920b44d6902caf1ab` |
| 网格墙钟 | 280.15501830005087 秒 |
| 参数哈希 | `219c52f59203ad7fb468cec9e59f4e47d58f632105a0056e54cef5d252ba4dfa` |
| `pass` / 行数 | true / 1200 |

当前源码字节与运行时记录一致：

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `configs/e4.yaml` | 699 | `cadcb90634b575cdbaaab58cbcd94c90107461df26c8905f510dd3c6035e52c4` |
| `src/noisyopinf/e4.py` | 21240 | `e54cb1fcbcb80a8ed1fa8ecb7911079da067f5620e8ecaf5c316b5e3b7ea5109` |
| `scripts/e4_rank_tradeoff.py` | 43118 | `17653d712401713d9d4df36ca372c19bb70cbe6756f4f953e3cce0e315eb23b2` |
| `tests/test_e4.py` | 5195 | `5463977a1d65dc263802357afe7ee9ae0bb31afef2eb5388842e68cb784a4d4c` |

`runtime_start.json` 与 `runtime_end.json` 的源码哈希没有漂移。估计器文件本轮没有修改。

## 3. 原始数据完整性

核对时重新计算了 `reports/e4_audit/source_hashes.json` 里 16 个原始文件的 SHA-256 和字节数，不一致的个数是 0。`output_manifest.json` 自身的哈希是 `adc23174d07ffb3bfe38121f280968d78c92084d82c88251f86d9dd06404ebc7`（2539 字节）；它列出那 16 个数据文件，不列出自己。

| 对象 | 预期 | 实际 |
|---|---:|---|
| `runs.csv` | 1200 | 1200，唯一键 `(rho, r, estimator, replicate)` 1200，状态全部 `ok` |
| `summary.csv` | 120 | 120 |
| `rank_selection.csv` | 60 | 60 |
| `rank_curves.csv` | 1200 | 1200 |
| `conditioning.csv` | 120 | 120 |
| `rank_gaps.csv` | 60 | 60 |
| `rank_stability.csv` | 60 | 60 |
| `snr_comparison.csv` | 80 | 80 |
| `snr_rank.csv` | 3200 | 3200 |
| `ridge.csv` | 3900 | 3900 |
| `sigma2_estimation.csv` | 50 | 50 |
| `failures.json` | 空 | `[]`，4 字节 |

`runs.csv` 里 `rollout_med=inf` 的行是 199，`rollout_med` 为 NaN 的行是 0。`summary.csv` 里滚动中位数为 inf 的格子是 21。算子数组里没有非有限算子。这些 inf 保留为 inf，没有改写成大数。

缺失配置数是 0。每个 `(rho, replicate)` 都有一条 `est_trailing`。OLS 和 Ridge 的 `sigma2_source` 是 `na`，不注入 σ²。MC 和 IV2 使用同一条重复上的 `est_trailing`。

## 4. Projection floor 口径审计

代码位置：

- 池化参考：`metrics.proj_floor`，`||V V^T u - u||_F / ||u||_F`。E4 在 `e4._one_rank` 里对 `np.vstack(clean_test)` 调用一次，再把这个数复制到该秩的每一行和每一条轨迹明细。
- 全视界滚动误差：`metrics.rollout_err_h`。预测系数乘 `V.T` 后，与同一条全阶真轨迹的同一窗口做相对 Frobenius。全视界使用整条轨迹。汇总先在轨迹上取中位数，inf 保留为 inf，再跨重复取中位数。

八项对照：

| 问题 | 结论 |
|---|---|
| 使用的 POD 基 | 相同。都是该 `(rho, replicate)` 的 noisy POD 前 r 列 |
| 测试轨迹 | 相同。都是干净测试轨迹 |
| 时间范围 | 相同。都是整条轨迹 |
| 归一化分母 | 相同。都是该比较对象自身的 Frobenius 范数 |
| 先池化还是先按轨迹 | 不同。`proj_floor` 先把 50 条轨迹叠成一个矩阵；`rollout` 先算每条轨迹 |
| 是否逐重复再汇总 | 相同的后半步。每个重复有自己的基和自己的数，summary 再取 10 次中位数 |
| 明细里的 `proj_floor` | 是池化参考的重复拷贝，不是该轨迹自己的投影误差 |
| 逐条下界是否被突破 | 否。容差 `1e-8 * max(1, |floor|, |rollout|)` 下，违反次数为 0 |

因此要分开四个量：

- 逐条投影下界：同一条轨迹、同一组正交基、同一窗口、同一个相对 Frobenius 范数下，`||V q - u||` 不小于 `||V V^T u - u||`。
- 池化投影参考：50 条轨迹叠在一起后的同一个公式。它是这个池化矩阵上的投影残差，不是每条轨迹误差的下界。
- 重复级投影误差：每个重复的池化参考，或每个重复内逐条投影误差的中位数。二者数值不同。
- 全视界滚动中位数：每条轨迹的全视界误差（发散记 inf）的中位数，再对重复取中位数。

独立复算把每个秩前缀重新作用到每条干净测试轨迹上，没有重新积分。120 个配置的池化中位数都与 `summary.csv` 的 `proj_floor_median` 一致，滚动中位数也都与 `rollout_med_median` 一致。22 个 `rollout_median < pooled_floor_median` 的格子全部在 rho≤0.05，逐条违反数全是 0。

一个代表格是 Ridge、rho=0.01、r=16：池化参考中位数 0.036260，逐条投影误差中位数的中位数 0.017995，滚动中位数 0.031658。滚动误差低于池化参考，同时高于该配置典型轨迹自己的投影误差。

rho=0.05 的池化参考中位数仍是：r=4 为 0.331306，r=6 为 0.186104，r=8 为 0.132786，r=10 为 0.095209，r=12 为 0.069076，r=16 为 0.036459。其他 rho 只在小数第四位以后有变化。下文把这个数称为池化投影参考。图 `fig_e4_heatmap.png` 和 `fig_e4_rstar.png` 的标题没有把这个数写成逐条下界，所以本轮不改图像素。热图白圈仍是误差曲线选出的 `r*_val`，不是在测试集上拟合出来的。

派生表：`reports/e4_audit/projection_floor_audit.csv`，120 行。

## 5. Rank selection 稳定性

预注册的选秩是：先把 10 条重复的轨迹中位数收成一条误差曲线，再在候选秩上取最小。另一个统计量是：每个重复独立取 argmin，再看这 10 个秩的 lower central median 或普通中位数。本轮两列都保存，不把它们当成同一个量。

Clean validation 13/20 个组合 10 次完全一致。Noisy validation 11/20 个组合 10 次完全一致。误差曲线秩与重复秩的 lower central median 不同的组合：clean validation 3 个，noisy validation 2 个。

Clean-validation 误差曲线秩，按 rho 从 0.01 到 0.2：

| 估计器 | 秩 |
|---|---|
| OLS | 12, 10, 8, 8, 6 |
| Ridge | 16, 16, 12, 10, 6 |
| MC | 12, 10, 8, 8, 6 |
| IV2 | 12, 12, 10, 8, 6 |

Noisy-validation 误差曲线秩：

| 估计器 | 秩 |
|---|---|
| OLS | 12, 10, 8, 6, 6 |
| Ridge | 16, 16, 12, 12, 16 |
| MC | 12, 10, 8, 8, 6 |
| IV2 | 12, 12, 10, 8, 6 |

描述性测试秩：OLS 为 12, 12, 10, 8, 6；Ridge 为 16, 16, 16, 16, 6；MC 为 12, 12, 10, 8, 6；IV2 为 12, 12, 10, 8, 8。

### A. Ridge，rho=0.2

Noisy validation 的 10 次选择是 r=6 一次、r=10 三次、r=12 一次、r=16 五次。误差曲线选 r=16，因为 r=16 的中位验证误差是 0.398926，r=12 是 0.400023，相对差距 0.002751。重复秩的 lower central median 是 12，10 个数的普通中位数是 14。只有 5/10 次与误差曲线的 r=16 相同。没有删除任何重复。Clean validation 有 9/10 次选 r=6，另一次选 r=8。描述性测试同样是 9/10 次 r=6、一次 r=8。

### B. Ridge，rho=0.1

Noisy validation 的 10 次是 r=10 四次、r=12 三次、r=16 三次。误差曲线选 r=12，第二名是 r=10，两条中位曲线的差是 `6.398686961817113e-05`，相对差距 `0.00028562884207254967`，与先前 `rank_gaps.csv` 的 0.000286 一致。众数是 r=10，不是 r=12；与误差曲线相同的重复只有 3/10。重复秩的 lower central median 和普通中位数在这个格子上都是 12，所以“中位数秩”和“误差曲线秩”碰巧相同，但 10 次选择本身并不集中。

Clean validation 的误差曲线选 r=10，重复秩的 lower central median 是 12。次数是 r=10 四次、r=12 四次、r=16 两次。相对差距 0.002181。

### C. OLS，rho=0.1

Clean validation 在 r=6 和 r=8 上各有 5 次。误差曲线选 r=8：r=8 的中位误差是 0.239033，r=6 是 0.239943，相对差距 0.003807。重复秩的 lower central median 是 6，普通中位数是 7。10 次里只有 5 次与误差曲线一致。没有异常到需要剔除的重复；这是一个接近并列的分裂。Noisy validation 则是 8 次 r=6、2 次 r=8，误差曲线选 r=6。

### D. 两个汇总

误差曲线先聚合再选秩，重复秩先选秩再取中位数。Ridge、rho=0.2 的 noisy validation 把这个差别显出来：前者是 16，后者的 lower central median 是 12。IV2、rho=0.02 的 validation 和 noisy validation 都是 5 次 r=10、5 次 r=12；误差曲线选 12，lower central median 是 10。OLS、rho=0.1 的 clean validation 也是这一类。

派生表：`reports/e4_audit/rank_stability_final.csv`，60 行。

## 6. Rank gap 分析

相对差距定义为 `(第二名 − 最优) / |最优|`。严格并列在 60 个比较里都是 1，也就是没有两个候选秩的聚合分数完全相等。

“close” 使用 0.05 这个阈值。`configs/e4.yaml` 没有 `relative_gap` 或 `close`。`reports/E4_PRERUN.md` 也没有把 0.05 写成运行前规则。因此 `close_exploratory_5pct` 的状态是 `EXPLORATORY_NOT_PREREGISTERED`。它不改变已经保存的选秩。

按这个探索阈值：clean validation 5/20，noisy validation 5/20，描述性测试 6/20。验证侧的接近格子包括 IV2 rho=0.02（相对差距 0.0471）、Ridge rho=0.05（validation 0.0297，noisy validation 0.00431）、IV2 rho=0.05（0.0454 和 0.0338）、OLS rho=0.1（0.00381 和 0.0453）、Ridge rho=0.1（0.00218 和 0.000286）、Ridge rho=0.2 的 noisy validation（0.00275）。

派生表：`reports/e4_audit/rank_gap_final.csv`，60 行。

## 7. SNR 四个 κ 的完整比较

定义保持为 `λ_i = mean_k q_{k,i}^2`，`r_snr(κ) = max {i : λ_i ≥ κ σ²}`，κ 为 1、4、16、64，σ² 为同一重复的 `est_trailing`。本轮从 `bases.npz` 的 `lam_rho*_rep*` 和 `sigma2_estimation.csv` 重算，不修改 κ。空集在这批数据里没有出现。

因为每个 `(rho, κ)` 的 10 次重复给出同一个 `r_snr`，lower central median 与普通中位数在这批数据上相同。这个巧合只说明本次样本，不说明两条规则永远一致。lower central median 写在运行前提交 `8f90c7e` 的配置和 `SNR_PLOT_AGGREGATE` 里，仍按预注册规则报告。

聚合 `r_snr`，按 rho 从 0.01 到 0.2：

| κ | 聚合秩 | 其中右删失的 rho |
|---|---|---|
| 1 | 16, 16, 16, 16, 15 | 0.01, 0.02, 0.05, 0.1 |
| 4 | 16, 16, 16, 15, 11 | 0.01, 0.02, 0.05 |
| 16 | 16, 16, 15, 11, 9 | 0.01, 0.02 |
| 64 | 16, 16, 11, 9, 6 | 0.01, 0.02 |

15、11、9 不在候选集合 `{4,6,8,10,12,16}` 里，也没有被吸附到最近的候选秩。

相对每个重复自己的 clean-validation 选秩，四个估计器、五个 rho、十次重复共 200 个差：

| κ | 平均绝对差 | 绝对差中位数的格平均 | ±2 的重复数 |
|---|---:|---:|---:|
| 1 | 5.98 | 与格内中位数一致到表中精度 | 24/200 |
| 4 | 5.00 | 同上 | 24/200 |
| 16 | 3.74 | 同上 | 31/200 |
| 64 | 2.26 | 同上 | 106/200 |

κ=64 在 rho=0.05、0.1、0.2 上、对四个估计器平均后的 replicate 级平均绝对差是 2.25、1.75、0.05。低噪声上的 16 受第 8 节的删失限制，不能把低噪声的零差距解读成规则已经命中。

对聚合 `r*_val` 的差距，κ=64 在 20 个估计器–rho 格里有 12 个绝对值不超过 2。超过 2 的格子包括 rho=0.01 的 OLS、MC、IV2（都是 +4），rho=0.02 的 OLS 和 MC（+6）以及 IV2（+4），rho=0.05 的 OLS 和 MC（+3）。这些低噪声正差距还是删失后的下界：若未保存的更高模态也通过阈值，真实差距只会更大。

四个 κ 都报告。κ=64 在这四个数里平均绝对差最小，这是对本档案的描述，不是把 64 选成通用常数。

派生表：`reports/e4_audit/snr_reconciliation.csv`（80 行）和 `reports/e4_audit/snr_replicate_detail.csv`（800 行，每个重复、每个 κ、每个估计器一行）。

## 8. SNR rank 上限和删失分析

`bases.npz` 里每个 `v_rho*_rep*` 的列数是 16，对应的 `lam_*` 也是 16 个数。E4 的 `r_max` 就是 16。归档里没有第 17 个及以后的基向量、奇异值或 λ。不能从现有数组恢复更多模态，也没有做外推。

因此：

- 16 个已保存模态全部满足 `λ_i ≥ κ σ²` 时，结果标记 `RIGHT_CENSORED_AT_RANK_16`。它表示阈值在可观察范围内没有被突破，真实的 max 下标大于或等于 16。
- `r_snr<16` 时，标记 `OBSERVED_WITHIN_16`。它是已保存 16 个模态中的最大通过下标。λ 来自干净训练轨迹在 noisy 基上的坐标能量，不自动继承 noisy 快照奇异值的单调性，所以这个数不对未保存模态做断言。

rho=0.01 和 rho=0.02 上，四个 κ、四个估计器、10 次重复全部是 `RIGHT_CENSORED_AT_RANK_16`。rho=0.05 上 κ=1 和 κ=4 仍是删失；κ=16 的聚合值是 15，κ=64 是 11，这两个落在已保存模态内部。rho=0.1 上只有 κ=1 删失。rho=0.2 上四个 κ 的聚合值都小于 16。

## 9. MC 高秩病态分析

MC 的 Gram 来自 `estimators._mc_moments`：用 Hermite 单项式和做噪声校正，不是未校正的 `Z^T Z`。条件数是 `np.linalg.cond(gram)`。最小特征值是对称化 Gram 的 `eigvalsh` 的第一个值。非 PSD 的 MC 走 `_solve_moments`，`n_clipped` 固定写成 0，然后直接 `np.linalg.solve`，不把负特征值截断。PSD 路径才会截断；E4 没有跑 MC-PSD。

r=16、五个 rho 的重复中位数：

| rho | Gram 条件数 | 最小特征值 | err_O | err_A | err_H | 谱半径 | 参考谱半径 | 发散比例 | 首次发散步 | 滚动中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.01 | 7.12e12 | −6.24e-8 | 10.73 | 1.13 | 45.54 | 1.070 | 0.998 | 1 | 7 | inf |
| 0.02 | 2.53e12 | −1.31e-6 | 12.65 | 1.19 | 53.90 | 1.277 | 0.998 | 1 | 6 | inf |
| 0.05 | 8.75e11 | −2.06e-5 | 23.18 | 2.38 | 99.22 | 2.720 | 0.998 | 1 | 5 | inf |
| 0.1 | 4.48e11 | −1.08e-4 | 40.66 | 5.41 | 169.97 | 2.567 | 0.998 | 1 | 4 | inf |
| 0.2 | 1.08e11 | −4.28e-4 | 43.11 | 3.74 | 187.33 | 2.708 | 0.998 | 1 | 4 | inf |

同一估计器在更低秩上的趋势说明，负特征值和立即发散不是从 r=8 就开始的。rho=0.01 时，r=8、10、12 的最小特征值仍为正（约 5.0e-4、1.9e-5、1.1e-6），`err_H` 是 0.018、0.050、0.273，滚动中位数有限，首次发散步的中位数在 67–71。到 r=16，最小特征值变成负的，`err_H` 跳到 45.5，首次发散步变成 7，滚动中位数变成 inf。rho=0.2 上，这个转折更早：r=10 的最小特征值仍为正，但 `err_H` 已经是 17.9、发散比例是 1、首次步是 17；r=12 和 r=16 的最小特征值为负，首次步是 5 和 4。

可以肯定的数值事实是：E4 的 MC 使用未截断的校正 Gram；r=16 的五个 rho 上这个 Gram 不是正定的；求解没有裁剪负特征值；算子的 H 块误差和谱半径同时离开参考；测试滚动全部发散。条件数、负特征值、算子误差和首次发散步在 r=8 到 r=16 上一起变差，这是相关性。OLS 在 r=16 的 Gram 条件数同样很大（rho=0.01 约 2.85e11，rho=0.2 约 1.33e9），但 OLS 不保存最小特征值，谱半径仍在参考附近（约 0.998 对 0.998，rho=0.2 约 0.996），发散比例约 0.42–0.51，并不是五个 rho 全部立即发散。所以大条件数本身还没有被证明是 MC 发散的唯一原因。特征数在 r=16 是 153，OLS/MC 的 `n_pairs` 是 6000，不是行数少于列数造成的欠定。

## 10. IV2 发散机制及未识别项

IV2 在 r=16 的五个 rho 上，滚动中位数都是 inf，发散比例从 0.57 到 0.86，首次发散步的中位数从 27 降到 13。`err_O` 约 0.97–2.54，`err_H` 约 4.14–10.52。谱半径仍在 0.998–1.000，参考约 0.998。这和 MC 在 r=16 上谱半径跳到 1.07–2.72 的形态不同。

`estimators._fit_iv` 会计算 `instrument_cond` 和 `first_stage_r2_median`。`e4._one_estimator` 只把 `gram_min_eig`、`gram_cond`、`n_clipped` 写入运行行，所以 IV2 的这两个诊断没有进入 `runs.csv`。`operators.npz` 只存估计算子和参考算子。`rollout_details.npz` 只有误差、发散标志、阈值和池化参考。`bases.npz` 只有 POD 基、奇异值和 λ。归档中没有工具变量矩阵、一阶段残差、一阶段 F、工具变量奇异值或相关交叉矩。

按本轮规则，E4 的 IV2 高秩发散标记为 `MECHANISM_NOT_IDENTIFIED`。没有用 E2 的 IV 诊断代替，也没有重新拟合。

## 11. Ridge 稳定性分析

`ridge.csv` 的 300 个 `(rho, r, replicate)` 组都有 13 个分数。用预注册的 `larger_lambda_index` 从这 13 个分数复选 λ，与 `runs.csv` 里的 `lam` 不一致的组数是 0。每个重复在每个秩上只选一次 λ。分数来自 `_ridge`：带噪降阶训练拟合，带噪降阶验证打分，不使用测试轨迹。

30 个 Ridge 汇总格的滚动中位数都有限。发散比例不是全 0。最大是 rho=0.2、r=6 的 0.12；rho=0.2、r=8 是 0.10。rho=0.2 的高秩滚动中位数约 0.356–0.358，明显高于 rho=0.01、r=16 的 0.03166。

λ 的重复中位数随秩和噪声变化，但不是每一行都随秩增大：

| rho | r=4 | r=8 | r=16 |
|---|---:|---:|---:|
| 0.01 | 0.01 | 0.001 | 0.0001 |
| 0.05 | 0.01 | 0.001 | 0.001 |
| 0.2 | 0.001 | 0.01 | 0.1 |

rho=0.1、r=4 的中位数是 0.1，同组最小值是 1e-10，重复之间差很大。高噪声、高秩上选中的 λ 远离 1e-10，正则化是起作用的。与此同时，r=16 的 `err_H` 停在大约 0.99–1.06，而 MC 的 `err_H` 是几十到一百多；Ridge 的滚动中位数保持有限。这是本网格上的描述：正则化较大的算子偏差可以和有限的长时预测同时出现。它不是一条普适的稳定化定理。

## 12. POD 截断误差、算子误差与 rollout 的权衡

r=4 的池化投影参考约 0.331，四个估计器和五个 rho 的滚动中位数约 0.367–0.385。Ridge 在 rho=0.01、r=4 的 `err_O` 只有约 0.00048。滚动误差紧贴池化参考，算子误差很小，所以 r=4 的主要限制是 POD 截断。这里的比较对象是池化参考，不是逐条下界；逐条投影误差的中位数约 0.310，滚动中位数仍高于它。

中等秩在部分格子上比两端更平衡。rho=0.05 时，OLS 和 MC 的 clean-validation 选 r=8，池化参考约 0.133，OLS 滚动中位数约 0.130。这个 0.130 低于池化参考，原因是第 4 节的汇总差异；它没有低于逐条下界。rho=0.01 时，继续增大秩仍能降低 Ridge 的滚动误差，r=16 达到 0.0317。rho=0.2 时，四个估计器的 clean-validation 都回到 r=6，高秩滚动误差更大或直接发散。

MC 和 IV2 在高秩上失去低秩时的预测表现，同时 H 块误差变大、发散比例升高。对 MC，第 9 节还有不定 Gram 和未截断求解这些数值事实。对 IV2，只能说算子误差和发散同时变大，机制未识别。

Ridge 在若干高秩格子上用更大的算子误差换到了仍然有限的预测。rho=0.2、r=16 的 `err_O` 约 0.462，滚动中位数约 0.357，而且有限。同一 rho 的 r=6 滚动中位数约 0.334，算子误差约 0.013。高秩并没有带来更好的预测，只是没有像 MC 那样变成 inf。

最优秩明显依赖估计器和噪声。低噪声下 Ridge 的 oracle 秩停在 16，OLS、MC、IV2 停在 10 或 12。rho=0.2 时四个估计器的 oracle 秩都是 6。Noisy validation 只有 Ridge 在最后一档重新升到 16，而且第 5 节说明这个 16 不稳定。

## 13. H6 最终判断

`01_研究问题与假设.md` 的原文是：

> H6：最优秩 \(r^*\) 随 \(\sigma\) 增大而不增；信噪比规则在 ±2 个模态内预测 \(r^*\)。

RQ5 问的是：预测误差最小的秩是否随噪声下降，以及 \(r^*(\sigma)\approx\max\{i:\lambda_i\ge\kappa\sigma^2\}\) 能否预测它。`02` 第 8 节要求四个 κ 都报告，不能事后只留最好的 κ。`04` 的 E4 把 clean-validation 秩定义为 oracle，并要求另报 noisy-validation 秩和描述性测试秩。

逐项判断：

1. 最优秩是否随噪声增加而下降。按预注册误差曲线，四个估计器的 `r*_val` 都不随 rho 上升，相邻 rho 有平台。描述性 `r*_test` 也不上升。Noisy validation 里 OLS、MC、IV2 不上升；Ridge 在 rho=0.2 从 12 升到 16。若干相邻步的相对差距小于 5%，R=10 不能把每一小步都说成精确的单调定律。第一句对 oracle 秩成立，对可部署的 noisy validation 不成立。
2. SNR 规则能否预测 clean-validation 最优秩。不能按 H6 的 ±2 要求成立。四个 κ 的 200 个重复绝对差里，落在 ±2 的分别是 24、24、31、106。κ=64 最好，仍然有 94/200 超出 ±2，而且低噪声的正差距被秩上限删失，真实差距可能更大。
3. SNR 规则能否作为实际可行的秩选择方法。不能。它不使用验证轨迹，也不区分估计器。低噪声下它被 16 截断；高噪声下它接近 OLS/MC/IV2 的 oracle 秩，同时 Ridge 的可部署秩在 rho=0.2 并不跟随它。
4. κ 是否存在与估计器无关的有效范围。在这四个预注册值里没有。同一个 κ 不能同时贴住低噪声的 Ridge（oracle 秩 16）和 OLS/MC（oracle 秩 12 或 10），而低噪声上的表面重合来自删失。
5. Ridge 是否构成明显例外。是。它的 oracle 秩在低噪声保持 16，下降得更晚；它的 noisy validation 在 rho=0.2 不稳定地选到 16，与 clean validation 和描述性测试的 r=6 分开。
6. 最大秩 16 是否造成 SNR 删失。是。rho=0.01 和 0.02 的四个 κ，以及更高 rho 上的较小 κ，都是 `RIGHT_CENSORED_AT_RANK_16`。

综合这两句原文，判断是 `H6_PARTIALLY_SUPPORTED`。第一句在预注册的 oracle 汇总上成立。第二句在四个预注册 κ 上都不作为整体成立。这个结论不依赖把 κ=64 提升为最优常数。

## 14. 科学局限

- 候选秩止于 16，SNR 在低噪声被右删失。未保存模态不能补。
- R=10，若干验证曲线的相对差距远小于 0.05。误差曲线秩和重复秩中位数可以不同。
- 5% close 阈值是探索性的。
- 池化投影参考不是逐条滚动误差的下界。
- MC 的不定 Gram、大条件数和发散同时出现；OLS 也有大条件数，所以因果还没有唯一化。
- IV2 的一阶段强度没有进入归档。
- Ridge 的有限滚动是这个网格上的观察。
- `r*_test` 只作描述。用它报告验证方案的性能会把选择和评价放在同一批轨迹上。
- E3 固定 r=8 只在中等噪声附近接近 OLS/MC 的 oracle 秩；低噪声下它截断了还能受益的更高秩，高噪声下它高于本轮选出的 r=6。

## 15. SHA-256 文件清单

原始 16 个文件与 `reports/e4_audit/source_hashes.json` 一致，本轮复算不一致数为 0。清单中的关键项：

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `results/e4/runs.csv` | 536468 | `27b6635b154c5ce21def1c5eea77d974e5178804d00a3ec88d5ba5a9e0203147` |
| `results/e4/summary.csv` | 181588 | `4196bd52b620d27d048b4e3e38541f63e64c7705c0a2ab705791c3e4b0aeb8cb` |
| `results/e4/rank_selection.csv` | 3624 | `517532638f2295736e637a371de890a17a573a9ca331f964746ca855f1ba537e` |
| `results/e4/rollout_details.npz` | 1213510 | `71fc968ee8bed70b567ff22aa8d17d26f33f0494274ee12c0484c277025bc2ee` |
| `results/e4/bases.npz` | 1703269 | `b0d2a3dade6d08a6b650ecd09bcb4be5a07f99133847bb99838057cc2856a2ff` |
| `results/e4/operators.npz` | 9583231 | `993ea1082615fdf4ce066096b83143f49d878da8589279f5d445d7a06bea473b` |
| `results/e4/output_manifest.json` | 2539 | `adc23174d07ffb3bfe38121f280968d78c92084d82c88251f86d9dd06404ebc7` |
| `figures/fig_e4_heatmap.png` | 99035 | `9124cf941661a9d035b4034420279e6c6e662b9c4c6d62133387546eb1b3385f` |
| `figures/fig_e4_rstar.png` | 287314 | `b6152566d0a3efd52e2b4895edda21a3e28d57526ba4dafc3c39ca1069b847fd` |

本轮新增派生文件：

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `reports/e4_audit/projection_floor_audit.csv` | 13993 | `e806f9ab75c74f006f0afb99522d3cb5c5b40e83dc2ecd8dc554dfad8c55507f` |
| `reports/e4_audit/snr_reconciliation.csv` | 5491 | `1d2ff5d147f0cc922b73ceba0738a47b50920f81811656047e0efb7588101ecb` |
| `reports/e4_audit/snr_replicate_detail.csv` | 58215 | `c32d9b3c8e658885eae9e3bbc5bda41ef8ff1899ac9d90429fac5b0d831282a9` |
| `reports/e4_audit/rank_stability_final.csv` | 3270 | `80ff4cd9378164d8771a560903a91766541b2912307bbfa7753339805ed8bb21` |
| `reports/e4_audit/rank_gap_final.csv` | 8208 | `0ee034f6651e2d0a65ffd98ad0d4ba58be5d0dc8f2ba9fa5e8f5a19b36c1a25e` |
| `scripts/e4_final_reconciliation.py` | 18072 | `7a91e60995f963c4a7a0d874e34a741543b4dbb84489b6059ad52646999a12ca` |

本报告随提交写入，写入时不知道该提交自己的哈希。

## 16. 独立复算和测试记录

`scripts/e4_final_reconciliation.py` 只读归档。它重算池化参考、逐条投影误差、SNR 最大下标、删失标记、误差曲线秩、重复秩中位数和 Ridge 的 λ。打印结果是：`POOLED_BELOW 22`，`OWN_VIOLATIONS 0`，`RIDGE_BAD 0`，四个 κ 的平均绝对差和 ±2 计数与上表一致。带 `--reuse` 时跳过已经写好的投影和 SNR 明细，只重算后面的汇总；完整路径会重新读取 npz。

本轮没有改测试和估计器。pytest 使用 `.venv\Scripts\python.exe` 和 `--basetemp %LOCALAPPDATA%\e4-pytest-tmp`，结果是 123 passed、0 failed，29.00 秒。`.pytest_cache` 出现 Permission denied 警告，不计为失败。

## 17. 最终 Git 状态

审计开始时分支是 `master`，HEAD 是 `c5ec0761080e3ae70c8d46bccbefa40bd8187e24`。工作区里原有的 `manifests/pre_implementation_baseline.json` 修改和未跟踪的 `uploads/` 保持不动，不进入本次提交。`results/` 被忽略，不提交。没有 `git reset`、`git clean`、`git add -A` 或 push。

本次只暂存本报告、`reports/PROGRESS.md`、`scripts/e4_final_reconciliation.py` 和五张新审计表。提交说明是 `docs: finalize E4 scientific audit`。

## 18. 下一阶段建议

E4 可以按 `E4_AUDIT_PASS_WITH_SCIENTIFIC_DISCREPANCY` 冻结。建议负责人决定是否接受 `H6_PARTIALLY_SUPPORTED`，以及是否在以后的实验里把秩上限提高到 16 以上后再讨论 SNR 低噪声段。那将是新的实验，不是本档案的修补。

IV2 若要识别发散机制，需要在新运行里保存一阶段矩阵和强度；现有 E4 字节不能补出这个诊断。MC 的不定 Gram 已经记录，若要检验“截断负特征值之后发散是否消失”，那是 MC-PSD 或新算法的问题，本轮不改 MC。

本报告到此停止。不启动 E5，不修改估计器，不增加新的模拟。
