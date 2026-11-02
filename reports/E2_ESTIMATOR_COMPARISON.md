# E2：S1 精确二次系统上的七估计器比较

状态：`E2_EXPERIMENT_COMPLETE_WITH_SCIENTIFIC_DISCREPANCY`

数值运行使用的 Git HEAD 是 `9bc73f190ba92c11d9b969348fd02cc15e6e9639`。E2 代码在运行时尚未提交。打包提交在运行和审计之后，提交信息是 `feat: run E2 exact-system estimator comparison`。结果 csv、json、npz 位于被忽略的 `results/`，该提交不包含研究数据。机器是 `HANQING-WIN`，Python 3.12.10，16 核。依赖：numpy 2.5.4、scipy 1.18.1、pandas 3.1.0、matplotlib 3.11.2、pyyaml 6.0.3、pytest 9.2.0、joblib 1.6.0、tqdm 4.70.1。

绝对路径前缀：`C:\Users\hanqi\Desktop\projects\p1-noisy-opinf\`

## 1. E1 完整性

E1 没有重跑，数值、图和统计没有改。运行前核对：

- HEAD 为 `9bc73f190ba92c11d9b969348fd02cc15e6e9639`，`git status --short` 为空。
- `results/e1/` 与 `results/e1_pilot/` 的八个结果文件，以及 `figures/fig_e1_bias_theory_vs_sim.png`，SHA-256 与 `reports/E1_OUTPUT_MANIFEST.md` 一致。
- `results/e0/summary.json` 为 `3251504f1bc645ed64d4beb959ba17ea0f033a7d735a11d9b2525a665c8a99df`（1010 字节，`pass=false`）。
- `results/e0_retry1/summary.json` 为 `a0b96c7f1d486c0c9e0b55ced1b57b1b5d50be2e94c83ad23f6c7852f0e4969a`（1033 字节，`pass=true`）。
- 运行结束后这三个摘要哈希未变。

E1 仍是 12 个配置、2400 次 OLS、0 失败。`theory_gap_rel` 约 0.00773–0.04701。n_train=10 的有限样本偏离大于 Monte Carlo 均值标准误的典型尺度，n_train=100 回到相近尺度。这个差距留在 E1，没有用来改 E2 的估计器。

## 2. 共同发散阈值

旧口径：每个估计器用自己的训练数组调用 `threshold_from_train`。oracle 的训练数组是干净轨迹，其余是带噪轨迹，发散线可以不同。

新口径从 E2 起生效，只用于评估。每个 `n_train`：

`B(n_train) = 1e3 * max ||q_k||_∞`，最大值只取该样本量的干净训练前缀。

实测三个值：

| n_train | B |
|---:|---:|
| 10 | 963.9619697494801 |
| 30 | 1189.4190516808758 |
| 100 | 1570.4767311563792 |

同一 `n_train` 下，七个估计器、六个 rho、100 次重复使用同一个 B。验证集、测试集和预测轨迹不参与。倍数仍是 1e3。发散误差仍是 inf，发散比例仍按 50 条测试轨迹计算。oracle 仍拟合干净训练轨迹；其余六个估计器仍共用该重复的带噪训练轨迹。`rollout.py` 的一步预测没有改。smoke 仍对每个估计器用它自己的训练轨迹取阈值。`02` 第 7 节只增加了这段说明。历史 E0、E1 不重算。

pytest 覆盖了：七个估计器阈值相同、阈值不随 rho 或 replicate 改变、不同 `n_train` 分开计算、oracle 与 OLS 的训练数据分流、inf 中位数保持 inf、任一 NaN 使分位数变为 NaN。这些测试在正式运行前通过。

## 3. 先导与预算

先导固定为 S1、n_train=30、rho=0.05、replicate 0 和 1、七个估计器，目录 `results/e2_pilot/`。`pass=true`，`problems=[]`。14 行运行、7 行汇总、26 行 Ridge 记录（2×13）。失败列表为空。

核对过的内容：七个估计器都跑了；oracle 与在干净前缀上重拟合的结果一致；OLS 与在共享带噪训练集上重拟合的结果一致；阈值都是 1189.4190516808758；参考是 `O_true`；IV 每条轨迹少 2 个端点、IV2 少 3 个端点，配对不跨轨迹；Ridge 的 λ 与 `select_ridge_lambda` 一致，该函数没有测试集参数；MC、MC-PSD、IV、IV2 的 `sigma2_source` 为 `true`，oracle、OLS、Ridge 为 `na`；同一重复内 sigma2 相同；算子可从 npz 读回；OLS 的全视界、h=10、h=50 中位数可独立重算；汇总中位数与原始行一致。

两次重复耗时 6.8555055998731405 秒。按 6×3×100 外推 6169.955039885826 秒。按配对数加权（乘以 140/90）为 9597.707839822397 秒，低于 4 小时。R 保持 100。正式墙钟后来是 1400.5735625999514 秒，短于这个上界。

## 4. 完成情况

18 个配置 × 7 个估计器 × 100 次重复 = 12600 行，`runs.csv` 与 `meta.json` 都是 12600。汇总 126 行。Ridge 记录 23400 行（1800 次选择 × 13 个 λ），不计入 12600。失败拟合 0。`failures.json` 为 `[]`。

每个配置都有日志行：rho ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2}，n_train ∈ {10, 30, 100}。没有缺配置，没有补种子，没有删行。

运行过程中有两次启动被交互式控制台的 Ctrl+C 关掉（计划任务上次结果 `-1073741510`，即 `STATUS_CONTROL_C_EXIT`）。那两次都停在写出结果文件之前，`results/e2` 里只有空目录，已删除后重跑。成功的这一次在独立控制台里跑完，是目录里的唯一一套结果。

## 5. 七个估计器

误差相对 `O_true`。表内是 100 次重复的中位数或 `bias_rel` / `sd_rel`。oracle 在全部配置上的 `err_O_rel` 中位数约为 2e-14 到 1e-13，是干净数据上的 OLS 与真算子之间的数值差。

`err_O_rel` 中位数：

| rho | n | oracle | ols | ridge | mc | mc_psd | iv | iv2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 5.22e-14 | 5.22e-14 | 1.54e-08 | 6.01e-12 | 6.01e-12 | 9.58e-12 | 4.63e-13 |
| 0.01 | 10 | 5.22e-14 | 0.353 | 0.134 | 0.109 | 0.109 | 0.120 | 0.105 |
| 0.05 | 10 | 5.22e-14 | 1.93 | 0.265 | 5.32 | 5.91 | 4.50 | 1.73 |
| 0.1 | 10 | 5.22e-14 | 2.53 | 0.372 | 14.7 | 63.1 | 20.0 | 3.16 |
| 0.2 | 10 | 5.22e-14 | 2.55 | 0.512 | 24.5 | 408 | 39.3 | 4.74 |
| 0.1 | 30 | 2.44e-14 | 0.692 | 0.301 | 0.145 | 0.145 | 0.151 | 0.111 |
| 0.1 | 100 | 1.17e-13 | 0.216 | 0.216 | 0.0253 | 0.0253 | 0.0257 | 0.0214 |
| 0.2 | 100 | 1.17e-13 | 0.591 | 0.468 | 0.0825 | 0.0825 | 0.0855 | 0.0562 |

`bias_A_rel`，rho=0.1：n=10 时 OLS 0.358，Ridge 0.267，MC 0.445，MC-PSD 0.754，IV 0.479，IV2 0.0965。n=100 时 OLS 0.0810，Ridge 0.0810，MC 0.000292，MC-PSD 0.000292，IV 0.000282，IV2 0.000311。

`bias_H_rel`，rho=0.1：n=10 时 OLS 20.6，Ridge 2.23，MC 145，MC-PSD 94.4，IV 104，IV2 13.5。n=100 时 OLS 1.71，Ridge 1.71，MC 0.0200，MC-PSD 0.0200，IV 0.0197，IV2 0.0195。

`sd_O_rel`，rho=0.1：n=10 时 OLS 0.727，Ridge 0.0146，MC 141，MC-PSD 64.5，IV 118，IV2 2.78。n=100 时 OLS、Ridge、MC、MC-PSD 都在 0.025 附近，IV2 是 0.0215。

rho=0 时，除 Ridge 外的六个估计器相对 `O_true` 的中位数都小于 1e-8，MC 与 MC-PSD 逐位相同。Ridge 在 n=10 为 1.54e-08，规格不要求它等于 oracle。

`||c_true||` 约为 9.07e-16。rho=0.1、n=30 的 `err_c_abs` 中位数：OLS 7.13e-4，Ridge 6.73e-4，MC 1.10e-4，IV2 8.86e-5。对应的 `err_c_scaled` 是 6.4e-3、6.1e-3、9.9e-4、8.0e-4。绝对误差和按状态均方根缩放的误差都是小量。

记录了 Gram 条件数的 7200 行全部有限。最大值为 `6905587492.349244`，出现在 MC、rho=0.05、n_train=10、replicate 97。该次拟合仍然有限。

## 6. 配对比较

配对差是同一 `(rho, n_train, replicate)` 上的 `err_O_rel(method) - err_O_rel(ols)`。Bootstrap 使用已有的一条 `rng(0, 77)`，按 `expand_cases` 再按估计器顺序，对 2000 次重复中位数取 2.5% 与 97.5%。OLS 的差是 0，也消耗这条随机流。没有为每个比较另开子流。

rho=0.1 的配对中位数和 95% 区间：

| n | 估计器 | 中位数 | 下端 | 上端 |
|---:|---|---:|---:|---:|
| 10 | ridge | -2.156 | -2.184 | -2.137 |
| 10 | mc | 12.18 | 9.95 | 14.89 |
| 10 | mc_psd | 60.59 | 55.68 | 66.86 |
| 10 | iv | 17.47 | 12.18 | 20.97 |
| 10 | iv2 | 0.636 | 0.505 | 0.697 |
| 30 | ridge | -0.392 | -0.395 | -0.389 |
| 30 | mc | -0.546 | -0.553 | -0.543 |
| 30 | iv | -0.541 | -0.548 | -0.537 |
| 30 | iv2 | -0.582 | -0.586 | -0.578 |
| 100 | ridge | -1.11e-10 | -1.12e-10 | -1.09e-10 |
| 100 | mc | -0.192 | -0.193 | -0.190 |
| 100 | iv | -0.190 | -0.191 | -0.189 |
| 100 | iv2 | -0.194 | -0.195 | -0.194 |

n=100、rho≤0.1 时 Ridge 与 OLS 的配对差在 1e-10 或更小，λ 选择落到了网格的小端附近。区间都不含 0 的那些比较，方向在这 100 次重复里是稳定的：n=10 时 MC、MC-PSD、IV、IV2 比 OLS 差；n=30 和 100 时 MC、IV、IV2 比 OLS 好。

## 7. 预测与发散

测试初值是干净的。发散误差为 inf，inf 进入中位数。全部分位数在样本含 inf 时仍用排序定义，避免 `np.quantile` 把 inf 变成 NaN。

`rollout_med` 中位数：

| rho | n | ols | ridge | mc | iv2 |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 10 | 0.748 | 0.194 | 0.207 | 0.193 |
| 0.02 | 10 | inf | 0.327 | inf | 1.05 |
| 0.05 | 10 | inf | 0.497 | inf | inf |
| 0.1 | 30 | 0.514 | 0.486 | 0.259 | 0.165 |
| 0.2 | 30 | 0.806 | 0.783 | inf | 0.702 |
| 0.1 | 100 | 0.385 | 0.385 | 0.0456 | 0.0382 |
| 0.2 | 100 | 0.749 | 0.742 | 0.159 | 0.0983 |

`diverged_frac` 中位数，n=10：rho=0.01 时 OLS 0.44、Ridge 0.06，MC 与 IV 为 0。rho=0.05 时 OLS 0.86、Ridge 0.28、MC 与 IV 为 1、IV2 为 0.92。rho=0.2 时 OLS 0.73、Ridge 0.12、MC、MC-PSD、IV、IV2 都是 1。

n=100、rho=0.2：OLS 0.14、Ridge 0.12，MC、MC-PSD、IV、IV2 为 0。

oracle 在全部配置的发散比例都是 0。

## 8. 五张图

rho 轴是线性的，包含 0。纵轴若遇 inf，不画成有限数，图题位置标注 inf，并在日志打印 `ROLLOUT_INF`。本次没有 `RMSE_NOT_POSITIVE` 或能量图的非正值省略。七个估计器都在图中。

| 图 | 绝对路径 |
|---|---|
| `fig_e2_bias_sd` | `C:\Users\hanqi\Desktop\projects\p1-noisy-opinf\figures\fig_e2_bias_sd.png` |
| `fig_e2_rmse_vs_n` | `C:\Users\hanqi\Desktop\projects\p1-noisy-opinf\figures\fig_e2_rmse_vs_n.png` |
| `fig_e2_specrad` | `C:\Users\hanqi\Desktop\projects\p1-noisy-opinf\figures\fig_e2_specrad.png` |
| `fig_e2_ep` | `C:\Users\hanqi\Desktop\projects\p1-noisy-opinf\figures\fig_e2_ep.png` |
| `fig_e2_rollout` | `C:\Users\hanqi\Desktop\projects\p1-noisy-opinf\figures\fig_e2_rollout.png` |

哈希见 `reports/E2_OUTPUT_MANIFEST.md`。

## 9. 异常与病态矩

失败拟合 0。没有非有限算子。没有把 inf 换成惩罚分。

Ridge 的 1800 次选择里，418 次至少有一个 λ 的验证分数非有限，0 次是 13 个 λ 全部非有限。这些 λ 的分数都留在 `ridge.csv`。测试集没有参与选 λ。

最大 Gram 条件数 `6.905587492349244e9` 来自 MC、rho=0.05、n_train=10、replicate 97，算子仍然有限。n_train=10 上 MC、MC-PSD、IV 的 `sd_O_rel` 可以到几十到几百，与这个病态尺度同向。

被标成 `ROLLOUT_INF` 的汇总中位数是：n=10 时，OLS 在 rho=0.02、0.05、0.1、0.2；MC 与 MC-PSD 在这四档；IV 在这四档；IV2 在 rho=0.05、0.1、0.2。n=30、rho=0.2 时 MC、MC-PSD、IV 也是 inf。

## 10. 对 H2–H4 的判断

这些是对这张网格的描述，不是定理。

H2。n_train=30 和 100 上，MC、MC-PSD、IV、IV2 的 A 与 H 偏差随样本量下降，并且小于同 rho 的 OLS。OLS 的偏差随 rho 增大；n=100、rho=0.2 时 `bias_A_rel` 仍是 0.237，`bias_H_rel` 仍是 4.61。n_train=10 且 rho≥0.05 时，MC、MC-PSD 和 IV 的偏差与方差一起放大，看不到“偏差趋向 0”。H2 在中等样本和大样本上成立，在 n_train=10 的高噪声格点上不成立。

H3。`specrad_err` 定义为 `ρ(A_hat) - ρ(A_true)`。OLS 在全部正 rho 上为负：谱半径偏小。rho=0.1 时，n=10、30、100 分别是 -0.0121、-0.00681、-0.00538。n≥30 时 MC 与 IV 的这个差接近 0，符号为正，过冲远小于 OLS 的负偏差。n=10、rho=0.2 时 MC 的谱半径误差是 +0.512，MC-PSD 是 +37.2。OLS 的伪耗散在这张网格上是稳定现象；MC/IV 的修正只在样本量够用时把谱半径拉回真值附近。

H4。n≥30 时，MC、IV、IV2 的能量违反度低于 OLS，并随 n 下降。n=100、rho=0.1：OLS 0.261，MC 0.0230，IV2 0.0188。n=10 时各带噪估计器的能量违反度都在约 0.09 到 0.43，MC 略低于 OLS，但 MC 的发散比例可以是 1，而 OLS 在 rho=0.2 是 0.73。能量违反度下降，并不自动给出稳定的 300 步预测。n=10 的发散更接近 H 的巨大方差和 Gram 病态。

另外三件和假设直接有关的事：MC-PSD 在 `sigma2=0` 时与 MC 相同，正噪声下多数格点也与 MC 相同；它没有降低 RMSE。n=10、rho=0.2 时它的 `err_O_rel` 中位数是 408，高于 MC 的 24.5。Ridge 在 n=10 把长期预测留在有限值，算子误差也小于 OLS；在 n=100、rho≤0.1 时 Ridge 与 OLS 几乎重合，算子误差和预测都差于 MC。算子误差低的估计器，在发散还没出现时预测也更好；一旦中位数本身是 inf，算子误差的排序不再代表长期预测。

## 11. SHA-256

正式结果、先导和图的路径、字节数和 SHA-256 在 `reports/E2_OUTPUT_MANIFEST.md`。Git 不跟踪 `results/e2` 与 `results/e2_pilot`。

## 12. pytest

正式运行前：

`.\.venv\Scripts\python.exe -m pytest --tb=short -q`

结果：106 passed，0 failed，25.10 秒。原有 94 个测试都在。新增测试覆盖 12600 行展开、嵌套前缀的索引约定、验证噪声偏移与复现、oracle/OLS 数据分流、共同阈值、Ridge 与现有选择器一致且签名不含测试集、S1 零噪声下 OLS/MC/MC-PSD/IV/IV2 相对 oracle 小于 1e-8、配对按 replicate 对齐、失败行保留 NaN、inf 中位数、已有目录拒绝覆盖、smoke 仍用各自训练数据的阈值。测试没有跑 12600 次拟合。

## 13. Git

数值运行 HEAD：`9bc73f190ba92c11d9b969348fd02cc15e6e9639`。

打包提交在本报告写入之后执行，信息为 `feat: run E2 exact-system estimator comparison`。提交后的哈希写在交给审阅的 uploads 文本里，不通过 amend 塞回这个提交。没有 push。

## 14. 工作区

运行开始时工作区干净。运行结束时，相对 `9bc73f1` 的改动是：`02` 第 7 节一段说明，新增 `configs/e2.yaml`、`src/noisyopinf/e2.py`、`scripts/e2_compare.py`、`tests/test_e2.py`、五张图、本报告、清单和 `PROGRESS.md` 的 E2 一节。`results/`、`data/`、`logs/` 未跟踪。

## 15. 受保护实现

没有修改 `estimators.py`、`theory.py`、`features.py`、`noise.py`、`seeds.py`、`pod.py`、`rollout.py` 的预测方程、S1/S2/S3、MC-PSD 的零噪声约定、IV/IV2、AR(1) Delta、κ、MP 校准、E1 与两组 E0 结果、04 的网格和通过阈值。

运行后 `theory.py` 的 SHA-256 仍是 `264c777fb42388a298ceda6a2fd600fc55ac43076beb4f1d46af0ede16975a13`。`estimators.py` 仍是 `a4716e630850e93bafc0acb55197ad4e7dfef0758c8ba4c06a32ccccc0e381c4`。

## 16. E3–E8

没有启动。

## 17. 下一步

交给负责人看这张网格。n_train=10 上 MC、MC-PSD 和 IV 的方差与全轨迹发散，是这次比较里最需要单独读的部分。在审阅结束前不改阈值、不改估计器、不开始 E3。
