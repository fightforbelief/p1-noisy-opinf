# PROGRESS

## Stage 1 系统与数据基础设施（2026-10-30）

状态：`STAGE1_SYSTEMS_AND_DATA_COMPLETE`

这不是正式 E0。没有启动 E0–E8。

### A. 环境

- 远程机器：`hanqing-win`（用户 `hanqi`）
- 项目目录：`C:\Users\hanqi\Desktop\projects\p1-noisy-opinf`
- 基础 Python：`C:\Users\hanqi\AppData\Local\Programs\Python\Python312\python.exe`
- Python 版本：3.12.10
- 虚拟环境：`C:\Users\hanqi\Desktop\projects\p1-noisy-opinf\.venv\Scripts\python.exe`（同一个 3.12.10）
- CPU：8 核 / 16 线程；`env_check` 读到的 `cpu_count` 为 16
- 内存：31.1 GB
- 建目录时 C 盘剩余：1025.59 GB

只升级了本项目 `.venv` 里的 pip，升级后为 26.2.2。没有改全局 Python、系统 PATH，也没有安装 PyTorch、CUDA、JAX。

### B. 依赖

`requirements.txt`：

- numpy==2.5.4
- scipy==1.18.1
- pandas==3.1.0
- matplotlib==3.11.2
- pyyaml==6.0.3
- pytest==9.2.0
- joblib==1.6.0
- tqdm==4.70.1

安装后这八个直接依赖的实际版本与上面一致。`pip check`：`No broken requirements found.`

`pip freeze` 在 `reports/pip_freeze.txt`。除直接依赖外，冻结列表里还有这些传递依赖：cloudpickle==3.1.2，colorama==0.4.6，contourpy==1.4.0，cycler==0.12.1，fonttools==4.68.0，iniconfig==2.3.1，kiwisolver==1.5.1，packaging==26.3，pillow==13.0.0，pluggy==1.7.0，Pygments==2.21.0，pyparsing==3.3.3，python-dateutil==3.0.0，PyYAML==6.0.3，six==1.17.0，tzdata==2026.6。

第一次安装即成功，没有改用非官方镜像。

### C. Git

初始提交：`a8207287d8f003d481a3ec8b5c1b4958149d2bb5`  
验收与本汇报之前的最新代码提交：`7e3c9968bd48fc4e6e6177b6db8d2c9a6afd1713`

中间提交：

- `4700903721e274baf5522758817fa395a5d96068` 忽略 macOS AppleDouble 文件
- `f641952` 从树里删除已经进入基线的 AppleDouble 文件
- `cbabc82` 固定并安装虚拟环境依赖
- `b2061c08dbdbe2072d7d6b1474b70c15a37f069e` S1、ETDRK4、S2、S3 及系统测试
- `7e3c9968bd48fc4e6e6177b6db8d2c9a6afd1713` 数据缓存与 POD

本汇报与 `reports/stage1_acceptance.json`、`scripts/stage1_acceptance.py` 在随后的提交里。没有 push。没有执行 `git add -A`。`.venv/`、`data/`、`logs/` 没有提交。

### D. 实现

- S1 Galerkin–Burgers：已完成。显式 Euler，`rng(1, 9)` 上的 OLS 真算子。
- 通用 ETDRK4：已完成。围道点 32，2/3 去混叠。
- S2 Burgers：已完成。参数保持 `N=256`，`nu=0.01`，`dt=2.5e-4`，幅度 0.5。
- S3 KdV–Burgers：已完成。参数保持 `N=512`，`nu=0.005`，`delta=1e-5`，`dt=1e-4`，幅度 0.5。
- `data.py`：已完成。按 index 前缀生成，npz 缓存带参数哈希，`joblib` loky。
- `pod.py`：已完成。不中心化 SVD；`trailing_median` 与 `gavish_donoho` 都按 02 第 5 节实现。

没有修改已经接受的 `theory.py`、`features.py`、`estimators.py`、`noise.py`、`seeds.py`、`tests/test_theory.py`。

### E. 数值验证

数值来自远程运行的 `reports/stage1_acceptance.json`。五条训练初值，index 0–4。没有改参数。

- S1 真算子残差最大绝对值：`1.582067810090848e-14`（阈值 1e-11）
- S1 常数项范数：`9.07314923274536e-16`（阈值 1e-12）
- S2 分辨率相对 L2 最大值：`9.756859013484725e-09`（阈值 1e-4）
- S3 分辨率相对 L2 最大值：`0.0004904044747538911`（阈值 1e-3）
- S3 终态范数比中位数：`0.46985669661102464`（阈值 0.2）
- S2 POD 秩（99% / 99.9% / 99.99%）：6 / 10 / 14
- S3 POD 秩（99% / 99.9% / 99.99%）：11 / 18 / 27

同一次运行里，S2 终态范数比中位数 0.31765046655008006，`max|u|` 1.3978808182734976。S3 `max|u|` 1.479198692437421。S2 与 S3 的 99% 能量秩不同。

文档作者在 Linux 上的五条轨迹参考值是：S2 收敛误差 8.9e-7、范数比 0.35、`max|u|` 1.61、POD 秩 8/13/18；S3 收敛误差 3.0e-4、范数比 0.44、`max|u|` 1.70、POD 秩 17/33/49。本次 S2 分辨率误差比该参考值更小，S3 分辨率误差约 1.6 倍，范数比和 `max|u|` 在同一量级。ETDRK4 阶测试已通过，因此没有改 N、dt 或初值幅度。

### F. 测试

在虚拟环境中执行 `python -m pytest --tb=line -q`：

- 通过：21
- 失败：1
- 耗时：11.69 秒

失败：`tests/test_data_pod.py::test_sigma2_trailing`。N=128、K=2000、秩 5、σ=0.3 时，`trailing_median` 给出 0.07165561351841215，真实 σ²=0.09，相对误差 0.2038，没有达到 05 的 5%。估计量按 02 第 5 节取 `i > N/2` 的特征值中位数；同一矩阵上全部特征值的中位数相对误差约 0.33%，但规格写的不是这个口径，所以没有改公式，也没有放宽断言。

系统测试（含 ETDRK4 线性精确解、四阶误差比、Burgers 均值守恒、S1 正交与能量）已通过。S1 嵌套前缀、缓存哈希、S2/S3 单条轨迹形状也已通过。

### G. 文件保护

远程 SHA-256 与实现前一致：

- `theory.py` `e762cd190ddd9c92cdf01262a032037189783323f2b199ee39783d3a429cb499`
- `features.py` `24dd2453330d880f416f7783de2b6b4b02cdc5b7c51f37b116c5388fd7da7153`
- `estimators.py` `ba2e5f473275c70037f2344cca7fbe95317da972074962159650b48896f57704`
- `noise.py` `ea39b097255081046cb0a953578f780f8a5d06be70f78bda919eefe39671c6de`
- `seeds.py` `ac3a246b35fc718f5eaaa01afcecdf0434f01f3b0637175ddf99a19e4d9b8473`
- `tests/test_theory.py` `189b6ff5ad7531a6a35e86fde5567ba22da90385979efad8cc4c6f883c654379`

### H. 最终状态

`STAGE1_SYSTEMS_AND_DATA_COMPLETE`

S1、ETDRK4、S2、S3、数据生成和 POD 都已在远程实现并运行。分辨率、终态范数比和 S1 真算子阈值均通过，参数未改。唯一未通过的检查是上面的 σ² 尾部中位数测试，原因是 02 的公式与 05 的 5% 阈值不一致。

### 下一阶段

还没有做，也不在本阶段启动：

- 其余估计器：ols、ridge、mc_psd、iv、iv2、oracle
- `add_white`、`sigma_from_rho`
- `ols_limit`、线性稀释公式和其余理论函数
- rollout、metrics、结果写入
- 05 里尚未覆盖的特征、Hermite、估计器和冒烟测试
- 正式 E0，然后才是 E1–E8
- E8 在 φ=0 时 `theory_gap_rel` 分母接近 0 的口径仍未决定

## Stage 2 核心估计器与理论 — 2026-10-30 16:31

**远程 git commit：**

- `140c5ce20233a5d726b942e3698973d4f3721344`：修正 `trailing_median` 的 MP 25% 分位数校准。
- `089a91e579107b0d8456369032bdaf8af39ed92c`：Stage 2 估计器、噪声模块、理论函数和测试。

本汇报在上述两个提交之后单独提交。

**做了什么：**

1. `trailing_median` 改为下半谱中位数除以 Marchenko–Pastur 分布的 25% 分位数。下半谱的取法没有改。Gavish–Donoho 仍使用 MP 中位数。
2. 实现 `ols`、`ridge`、`mc_psd`、`iv`、`iv2`、`oracle`。`mc` 的 Hermite 矩和交叉矩没有改。`ridge` 的 λ 由调用方传入。
3. 实现 `add_white` 和 `sigma_from_rho`。`add_ar1` 的递推没有改。
4. 实现 `expected_monomial`、`expected_monomial_sums`、`ols_limit`、`linear_attenuation_limit`。`mc_limit_ar1` 的已批准公式没有改，函数体也没有改。
5. 按 05 增加了特征、Hermite、估计器、噪声、参考值和 OLS 极限测试。没有放宽 5% 的 σ² 阈值，也没有更换 `test_sigma2_trailing` 的种子或数据。

**σ² 校准：**

同一组测试数据（N=128，K_tot=2000，秩 5，σ=0.3，种子不变）。β=N/K_tot=0.064。远程诊断脚本用修改前的 `pod.py` 和修改后的 `pod.py` 对同一矩阵计算：

| 量 | 数值 |
|---|---|
| 真实 σ² | 0.09 |
| 旧 `trailing_median` | 0.07165561351841215 |
| 旧相对误差 | 0.20382651646208722 |
| μ_{β,0.25} | 0.7894173790126653 |
| μ_{β,0.5} | 0.9786254990139102 |
| 新 `trailing_median` | 0.09077025085010514 |
| 新相对误差 | 0.008558342778945991 |
| 旧 Gavish–Donoho | 0.09179297619324095 |
| 新 Gavish–Donoho | 0.09179297619324095 |
| 两者之差 | 0.0 |
| `mp_median` 与旧实现之差 | 0.0 |

新相对误差小于 05 要求的 5%。`test_sigma2_trailing` 已通过。

**测试状态：**

针对性测试：43 通过，1 失败，17.05 秒。完整 pytest：50 通过，1 失败，19.20 秒。失败项只有 `tests/test_estimators.py::test_iv_intercept_correction`。

该测试在 n=49750 的一次 IV 拟合上，未修正截距与 `-O^T μ` 的差的范数是 3.205055642415369e-05，而 `||-O^T μ||` 是 1.1510620950775699e-07，比值约 278，没有达到 10%。修正量精确等于 σ² 乘以估计出的对角二次行之和，所以修正算术与 02 的公式一致。一次样本的截距误差比这个 O(σ²) 偏差大约两个数量级。在保持轨迹有界的前提下加大 H 或改变 σ，这个比值仍然远大于 0.1。因此这是 02 的渐近偏差与 05 的 10% 有限样本断言对不上，不是实现把公式写错。没有放宽 10%，也没有改 IV 的定义。按 07 第 3 节停止。

其余已通过的必要测试包括：特征指数与二次项顺序、Hermite 多项式及其无偏性、白噪声与 AR(1) 的方差和自相关、OLS、MC、MC-PSD、IV/IV2 的无噪声恢复与大样本误差下降、oracle、奇异设计返回 NaN、`ols_limit`、`linear_attenuation_limit`、原有的 `mc_limit_ar1`，以及 02 第 3.6 节参考值。

**实验运行：** 未运行。没有启动 E0–E8，没有重跑 Stage 1 的大规模系统生成。

**与规格的偏离：**

- 配对测试除了检查长度为 4 和 5 的两条轨迹给出 7 个配对、而不是跨轨迹的 8 个之外，另用两条长度为 12 的轨迹核对系数。宽度为 10 的设计在 7 行时秩亏，`ols` 按规格返回 NaN，无法再比较系数。阈值没有放宽。
- Hermite 的 0 次多项式恒为 1，样本标准差为 0。差为 0 时判定为通过。非退化样本仍要求差严格小于 5 倍标准误。

**文件保护：**

`features.py` 与 `seeds.py` 的 SHA-256 未变。`mc_limit_ar1` 只增加了文件后部和 import，函数体未改。

| 文件 | 原 SHA-256 | 现 SHA-256 |
|---|---|---|
| `features.py` | `24dd2453330d880f416f7783de2b6b4b02cdc5b7c51f37b116c5388fd7da7153` | 相同 |
| `seeds.py` | `ac3a246b35fc718f5eaaa01afcecdf0434f01f3b0637175ddf99a19e4d9b8473` | 相同 |
| `theory.py` | `e762cd190ddd9c92cdf01262a032037189783323f2b199ee39783d3a429cb499` | `264c777fb42388a298ceda6a2fd600fc55ac43076beb4f1d46af0ede16975a13` |
| `estimators.py` | `ba2e5f473275c70037f2344cca7fbe95317da972074962159650b48896f57704` | `26c5458fb6f054e3abec754f39e469853e6c755a289d1e7bf1b8a5b70fd3c71c` |
| `noise.py` | `ea39b097255081046cb0a953578f780f8a5d06be70f78bda919eefe39671c6de` | `5ff9778c7bca3da1e51c85bbde53093ac13b887f87195fe53eb96dc245e0deac` |
| `tests/test_theory.py` | `189b6ff5ad7531a6a35e86fde5567ba22da90385979efad8cc4c6f883c654379` | `5e7e5ceae8b3b0c2a8094890d7417dd55d29df7dc40648f60a9f57fe384cc3b8` |
| `pod.py` | Stage 1 版本 | `83c8ea68ce98ce17caac7eac9d55516f92b3914b0ac0bcb1b5fe3466bd791a6d` |

**修改文件：**

- 校准提交：`src/noisyopinf/pod.py`，`tests/test_data_pod.py`，`p1-noisy-opinf-docs/02_数学与估计器规格.md`
- Stage 2 提交：`src/noisyopinf/estimators.py`，`src/noisyopinf/noise.py`，`src/noisyopinf/theory.py`，`tests/test_theory.py`，`tests/__init__.py`，`tests/quadratic_system.py`，`tests/test_features.py`，`tests/test_hermite.py`，`tests/test_estimators.py`，`tests/test_noise_pod.py`，`tests/test_reference_values.py`

**未完成：**

- `test_iv_intercept_correction` 未通过，见上。
- rollout、metrics、io_utils 和实验报告流水线没有实现。
- E0–E8 没有启动。
- E8 在 φ=0 时 `theory_gap_rel` 的填报口径没有改。
- S1/S2/S3 参数没有改。AR(1) 的 Δ 没有改。

**问题与建议：**

`test_iv_intercept_correction` 要求 n=50000 时未修正的 ĉ 与 `-O^T μ` 的相对差小于 10%。在 05 规定的有界随机二次系统上，这个偏差是 O(σ²)，一次拟合的截距标准误比它大两个数量级。需要负责人决定：是保持 10% 并改用可分辨该偏差的检验方式，还是确认当前 IV 修正实现后把该断言移出本阶段的阻塞项。在此之前不继续实验。

**下一步：** 等待负责人决定。不启动 E0–E8。

**最终状态：** `STAGE2_CORE_ESTIMATORS_AND_THEORY_INCOMPLETE`

## Stage 2 文件清单 — 2026-10-30 16:56

**远程 git commit：** 清单和本段汇报在本提交中。此前的实现提交没有改动。

**做了什么：**

1. 任务 1 到 3 已在 `140c5ce20233a5d726b942e3698973d4f3721344`、`089a91e579107b0d8456369032bdaf8af39ed92c` 和 `a45e005a83c0dd2d2f1fdd121685567c6141d8c3` 中完成。这次没有再改 σ² 公式、估计器、`mc_limit_ar1`、AR(1) 递推或种子。
2. 在该 HEAD 上重跑完整 pytest：50 通过，1 失败，19.11 秒。失败项仍是 `tests/test_estimators.py::test_iv_intercept_correction`，数值与 16:31 的记录相同。
3. 生成 `manifests/pre_implementation_baseline.json`。它记录的是实现已经提交之后的 `a45e005a83c0dd2d2f1fdd121685567c6141d8c3`，不是写代码之前的空树。范围是当时 `git ls-files` 的 49 个文件，不含 `.venv`、`data`、`logs`、`results`。Python 为 `.venv` 中的 Python 3.12.10，并附有 `pip freeze` 的 23 行。

**测试状态：** 50 通过 / 51 项，1 失败。

**实验运行：** 未运行。没有启动 E0–E8。

**阻塞项：** `test_iv_intercept_correction`。一次拟合的截距误差范数是 3.205055642415369e-05，渐近偏差范数是 1.1510620950775699e-07，相对差约 278，没有达到 10%。修正算术与 02 一致。没有放宽阈值，也没有改公式。

**最终状态：** `STAGE2_CORE_ESTIMATORS_AND_THEORY_INCOMPLETE`

## Stage 2 IV 截距验收修订 — 2026-10-30 17:19

**远程 git commit：** 没有提交。改动留在工作区，等待审查。

**做了什么：**

1. 没有改 IV 截距公式，也没有改 `estimators.py`。检查没有发现坐标转换错误：原坐标修正量仍等于 `σ² * sum(O[diag_quad_rows])`。
2. `test_iv_intercept_correction` 分成两部分。总体 IV 矩用干净配对，对角二次特征加上 `μ`，解 `O_pop = (W^T Z)^{-1} W^T Y`，不跨轨迹。截距相对 `-O^T μ` 的差仍要求小于 10%，修正后的截距误差仍要求小于 `0.1 * ||O^T μ||`。实际 `fit` 只检查 `iv` 和 `iv2` 的修正量恒等式，不再把单次拟合误差拿去和理论偏移比。
3. 只改了 `05_测试与验证.md` 第 3 节这一行的文字。02 的公式没有改。

**远程数值：**

同一组轨迹，`||O^T μ|| = 1.15106209507757e-07`。总体矩截距的相对差是 `8.710287162992236e-11`。修正后的截距误差是 `9.974659986866641e-18`，与 `||O^T μ||` 的比是 `8.665614157153224e-11`。10% 标准仍然保留，并且通过。`iv` 与 `iv2` 的代数修正断言通过。

**测试状态：**

`test_iv_intercept_correction`：1 通过，0.84 秒。完整 pytest：51 通过，0 失败，19.03 秒。

**实验运行：** 未运行。没有启动 E0–E8。

**修改文件：**

- `tests/test_estimators.py`
- `p1-noisy-opinf-docs/05_测试与验证.md`
- `reports/PROGRESS.md`

**最终状态：** `STAGE2_CORE_ESTIMATORS_AND_THEORY_COMPLETE`

## Stage 3 实验基础设施 — 2026-11-02 16:16

**远程 git commit：** 本段与基础设施一起提交。起点是 `59f69b737e9f0a0f4e24f04ddbd3beab51f14544`。

**做了什么：**

1. 增加 `rollout.py`。原坐标迭代 `q_{k+1} = features(q_k) @ O`。轨迹长度是 `n_steps + 1`。非有限的新状态以及其后填 NaN。超过 `10^3 * max_train ||q||_∞` 的有限状态保留原值，其后填 NaN。不裁剪，不重开，不改输入。
2. 增加 `metrics.py`。实现 02 第 7 节的算子、A、H、常数项、谱半径、能量违反、预测误差、中位数、非发散均值、发散比例、bias、sd、IQR、投影下限，以及 04 的 `rollout_med_h10`、`rollout_med_h50`。bootstrap 比较单独放在这里，不放进估计器。
3. 增加 `io_utils.py`。`runs.csv` 使用固定列，空值留空，`inf` 保留为发散，NaN 不写成 0。JSON 和 npz 先写临时文件再替换。
4. 增加 `protocol.py`。同一次重复里训练集和验证集独立加噪，测试轨迹保持干净，ridge 的 λ 只在带噪验证轨迹上选择，并列时取较大的 λ。没有运行 E0–E8。

**测试状态：**

HANQING-WIN，项目 `.venv`，Python 3.12.10。

- `tests/test_rollout.py`：6 通过，0 失败，0.12 秒
- `tests/test_metrics.py`：6 通过，0 失败，0.06 秒
- `tests/test_io_utils.py`：4 通过，0 失败，0.09 秒
- `tests/test_protocol.py`：2 通过，0 失败，0.08 秒
- 完整 pytest：71 通过，0 失败，19.26 秒

`git diff --check` 退出码 0。

**实验运行：** 未运行。没有启动 E0–E8。不要把 E0 记成已通过。

**已接受的 Stage 2 文件：** 没有改。`theory.py` 的 SHA-256 仍是 `264c777fb42388a298ceda6a2fd600fc55ac43076beb4f1d46af0ede16975a13`。`tests/test_theory.py` 里的 `test_one_dimensional_delta_blocks` 和 `test_quadratic_cross_term_uses_both_sums` 仍在。`estimators.py` 没有改。

**与规格的偏离：**

- `runs.csv` 在 03 第 13 节的列上追加了 04 第 1.1 节点名的 `rollout_med_h10` 和 `rollout_med_h50`，放在 `rollout_med` 后面。
- 相对误差的分母为 0 时返回 NaN。02 没有规定这个分母。
- 指标函数的秩取自算子列数。03 第 12 节写了单独的 `r` 参数；算子形状已经是 `(p, r)`。
- 结果文件默认不覆盖已有文件。调用方传入 `overwrite=True` 才替换。03 没有写覆盖规则。
- `protocol.py` 只放重复划分和 λ 选择，避免把这些规则写进估计器。

**问题与建议：**

E8 在 φ=0 时 `theory_gap_rel` 的填报口径仍未决定。这次没有改 04 或 05。

**下一步：** 停在这里，不自动开始 E0。

**最终状态：** `STAGE3_EXPERIMENT_INFRASTRUCTURE_COMPLETE`

## Stage 3 End-to-End Integration Acceptance — 2026-11-02 16:53

**远程 git commit：** 本段与集成测试一起提交。起点是 `1cb6b0ad096eda49f54aca96b233efbf02385190`。

**做了什么：**

1. 新增 `pipeline.py` 和 `tests/test_smoke.py`。用 S2、r=4、n_train=3、K=40、rho=0.05、R=1，验证集 20 条、测试集 50 条，跑 oracle、ols、ridge、mc、mc_psd、iv、iv2。噪声加在全阶状态上再投影。输出写在 pytest 临时目录。
2. 视界短于 h+1 时，`rollout_err_h` 抛出 ValueError。K=40 的轨迹只有 41 个状态，`rollout_med_h50` 留空，不用 h=40 代替 h=50。
3. `runs.csv` 遇到未知字段时抛出 ValueError，不再静默丢掉。数学上的 NaN 写成 `nan`，不适用的字段保持空白。
4. 全部测试轨迹发散时，`rollout_mean_finite` 按定义是 NaN，该行 `status` 仍为 `ok`。这不是拟合失败。

**测试状态：**

HANQING-WIN，项目 `.venv`，Python 3.12.10。

- `test_smoke.py`、`test_metrics.py`、`test_io_utils.py`、`test_protocol.py`：17 通过，0 失败，3.70 秒
- 完整 pytest：76 通过，0 失败，22.56 秒

`git diff --check` 退出码 0。

**实验运行：** 未运行正式 E0–E8。没有写入 `results/e3`。

**smoke 配置与结果：**

- 系统 S2，物理参数仍来自 `configs/systems.yaml`。观测步数只用这次测试的 K=40。
- 行数 7，等于 1 个配置 × 7 个估计器 × 1 次重复。
- `sigma2_true = 7.782203674037263e-05`。`sigma2_source` 对 mc、mc_psd、iv、iv2 是 `true`，对其余估计器是 `na`，对应的 `sigma2_used` 为空白。
- 配对数：oracle、ols、ridge、mc、mc_psd 为 120；iv 为 117；iv2 为 114。
- Ridge 的 λ 是 `10.0`。其他行的 `lam` 为空白。
- 七行 `status` 都是 `ok`，`error` 为空白。
- `rollout_med_h50` 全部为空白。
- oracle 与 ols：`diverged_frac=0.66`，`rollout_med=inf`，`rollout_med_h10=inf`，`rollout_mean_finite=1.0235660004976945`。
- ridge：`diverged_frac=0.12`，`rollout_med=1.0415609357980042`，`rollout_med_h10=0.9923008743688002`，`rollout_mean_finite=1.0649725426658625`。
- mc、mc_psd、iv：`diverged_frac=1`，`rollout_med=inf`，`rollout_med_h10=inf`，`rollout_mean_finite=nan`。
- iv2：`diverged_frac=0.74`，`rollout_med=inf`，`rollout_med_h10=inf`，`rollout_mean_finite=1.2495128347380786`。
- `inf` 是发散轨迹的误差，参加了中位数。没有使用惩罚分数。

**五项已接受的偏离：**

- `rollout_med_h10` 和 `rollout_med_h50` 仍保留。
- 相对误差分母为 0 时仍返回 NaN。这次 smoke 的算子误差没有碰到这个分母。
- 秩仍取自算子列数。
- 结果文件默认仍不覆盖。
- 重复划分和 λ 选择仍在 `protocol.py`，没有写进估计器。

**已接受的 Stage 2 文件：** 没有改。`theory.py` 的 SHA-256 仍是 `264c777fb42388a298ceda6a2fd600fc55ac43076beb4f1d46af0ede16975a13`。`estimators.py` 没有改。

**与规格的偏离：**

- 这一行的 `exp` 写成 `smoke`，避免把它记成正式 E3。basis 只用 `clean`，σ² 只用真实值，没有跑 E3 的 basis × σ² 网格。
- 验证集 20 条、测试集 50 条沿用 04 第 1.1 节。05 第 7 节没有另行规定这两个数量。

**问题与建议：**

E8 在 φ=0 时 `theory_gap_rel` 的填报口径仍未决定。

**下一步：** 可以开始准备 E0。这次没有进入 E0。

**最终状态：** `STAGE3_END_TO_END_INTEGRATION_COMPLETE`

## E0 前置检查 — 2026-11-02 17:03

**远程 git commit：** 没有提交。起点仍是 `62fe1b901cfd602f8896123854a6d92b606e0694`。修复留在工作区。

**Gate A：** `BLOCKED`

smoke 里 Oracle 和 OLS 的预测均值相同，是因为 `pipeline.py` 把 `oracle` 也拟合在带噪训练轨迹上。`estimators.fit` 对这两个名字使用同一个最小二乘求解器，差别只应来自训练数据。修复前七个估计器拿到的是同一个 `noisy_train_q` 对象。

修复前，HANQING-WIN 上同一次 smoke 的只读诊断：

- Oracle 与 OLS 的算子都有限，内存不共享，但 \(\|O_{\mathrm{ols}}-O_{\mathrm{oracle}}\|_F=0\)。
- 相对干净训练上的 Oracle，\(\|O_{\mathrm{ols}}-O_{\mathrm{clean}}\|_F/\|O_{\mathrm{clean}}\|_F=1.1708528631199147\)。
- 全阶训练噪声均方是 `7.732426088440512e-05`，目标 `sigma2_true` 是 `7.782203674037263e-05`。降阶投影后的均方是 `8.428184687289594e-05`。
- 测试轨迹与干净测试轨迹一致。发散阈值 `2882.9612563075525` 来自带噪训练状态。
- 配对数都是 120，等于 3 条轨迹各自的 40 个相邻配对，没有跨轨迹。
- 50 条测试轨迹的误差和发散标记完全相同。发散下标：0–4、6–8、11、12、15、17–19、23–25、27、28、31、33–35、37、39、40、42、44–49。未发散下标：5、9、10、13、14、16、20–22、26、29、30、32、36、38、41、43。
- 两者的 `diverged_frac` 都是 0.66，`rollout_mean_finite` 都是 `1.0235660004976945`。

修复只改了 `pipeline.py` 和 `tests/test_smoke.py`。`oracle` 使用 `clean_train_q`，其余估计器仍共用 `noisy_train_q`。每个估计器的发散阈值来自它自己的训练轨迹。没有改估计器、理论或阈值。

修复后同一配置：

- Oracle 在干净数据上，OLS 在带噪数据上，两个算子不共享内存。
- \(\|O_{\mathrm{ols}}-O_{\mathrm{oracle}}\|_F/\|O_{\mathrm{oracle}}\|_F=1.1708528631199147\)。
- Oracle：`err_O_rel=0`，`diverged_frac=0.28`，`rollout_med=1.0110377125145014`，`rollout_mean_finite=2.4780765411692833`。
- OLS：`err_O_rel=1.1708528631199147`，`diverged_frac=0.66`，`rollout_med=inf`，`rollout_mean_finite=1.0235660004976945`。

**测试状态：** 完整 pytest 76 通过，0 失败，22.58 秒。`git diff --check` 通过。

**Gate B：** 没有执行。没有写 `results/e0`。没有运行 E1–E8。

**问题与建议：**

E8 在 φ=0 时 `theory_gap_rel` 的填报口径仍未决定。本次没有涉及该指标。

**下一步：** 等待负责人复核这条集成修复。通过后再运行正式 E0。

**最终状态：** `E0_PREFLIGHT_BLOCKED`

## Gate A 验收 — 2026-11-02

负责人原则上接受上述集成修复。本节只记录代码复核和回归测试，正式 E0 在本节提交之后另行执行。

**数据流：** `training_trajectories` 把 `oracle` 指到 `clean_train_q`，把 `ols`、`ridge`、`mc`、`mc_psd`、`iv`、`iv2` 指到同一次重复的 `noisy_train_q`。每个估计器各自调用 `fit`，返回新的算子数组。参考算子是另一次干净数据上的 `fit`。测试轨迹保持干净，预测初值取自干净测试轨迹。POD 投影与全阶重构使用同一组基。

**回归测试：** `tests/test_smoke.py` 核对六件事：训练列表就是对应的那一份数据；逐条轨迹的数值与该来源一致，并且和另一来源不同；Oracle 与独立的干净数据拟合一致到 `1e-8`；OLS 与 Oracle 的算子差非零；七个拟合结果在同时存活时不是同一个数组，也不能通过改一份影响另一份；其余六个估计器的训练数值是带噪轨迹。

**发散阈值口径：** 02 第 7 节规定 \(\|\hat q_k\|_\infty > 10^3 \max_{\mathrm{train}}\|q\|_\infty\)，没有说明训练数据来源不同的估计器是否共用一个阈值。当前实现不改。每个估计器用自己的训练轨迹计算这个阈值。`inf` 和 `nan` 的处理也不改。E0 不依赖 rollout 发散比例，所以这个口径不阻塞 E0。进入 E1–E8 之前，需要单独明确跨估计器阈值是否可比。

**测试：** 完整 pytest 76 通过，0 失败，0 跳过，22.82 秒。`git diff --check` 通过。工作区只有 `pipeline.py`、`tests/test_smoke.py`、`reports/PROGRESS.md`。起点 HEAD 是 `62fe1b901cfd602f8896123854a6d92b606e0694`。

**本段提交说明：** `fix: fit oracle on clean training trajectories`

**仍未决定：** E8 在 φ=0 时 `theory_gap_rel` 的填报口径。本次没有改这个指标。

## 正式 E0 — 2026-11-02

**最终状态：** `E0_SANITY_CHECK_FAILED`

Gate A 已单独提交：`ae168aaef3a4e8f6aba13daa57a9809177673d0f`，说明是 `fix: fit oracle on clean training trajectories`。E0 在这个提交之后运行。没有通过，所以没有做 E0 提交，也没有 push，没有启动 E1。

**运行：** HANQING-WIN，`scripts/e0_sanity.py`，项目 `.venv` 的 Python 3.12.10。HEAD `ae168aaef3a4e8f6aba13daa57a9809177673d0f`。工作区当时只有未跟踪的 `scripts/e0_sanity.py`。墙钟时间 12.638672099914402 秒。数据由本次进程的 `simulate` / `rollout` 生成，没有读 `data/` 缓存。配置哈希：S1 `096ef3739956fafc85587838b262b774c8f7eb033ddd76c1671918998f373986`，S2 `219c52f59203ad7fb468cec9e59f4e47d58f632105a0056e54cef5d252ba4dfa`，S3 `b3f4a91b39704fe5f1b0c39d2bfc124abbfcb59d34a87e3caed9ac0d164cd23e`。

04 没有规定零噪声一致性和 σ² 估计的训练条数与 POD 秩。本次使用 `n_train=10`。S2/S3 的秩是 8。S2 快照数 `K_tot=2010`，`N=256`，满足 `K_tot >= 2N`。

**分辨率：** S2 最大相对 L2 `9.756859013484725e-09`，阈值 `1e-4`，通过。S3 `0.0004904044747538911`，阈值 `1e-3`，通过。S3 在 `t=1` 的范数比中位数 `0.46985669661102464`，阈值 `0.2`，通过。S2 的 99% / 99.9% / 99.99% POD 秩是 6 / 10 / 14，S3 是 11 / 18 / 27。99% 秩不同。没有改物理参数。

**S1 真算子：** 残差最大值 `1.582067810090848e-14`（阈值 `1e-11`），`||c_true||` `9.07314923274536e-16`（阈值 `1e-12`），`ep_violation` `7.064961743537406e-15`（阈值 `1e-12`）。三项都通过。

**零噪声：** S1 上 ols、mc、mc_psd、iv、iv2 相对 oracle 的误差分别是 `0`、`6.017318142555264e-12`、`3.5815203955181153e-11`、`9.565224260750429e-12`、`4.638941946116838e-13`，都小于 `1e-8`。S3 上 ols、mc、mc_psd 是 `0`、`4.584926680894034e-13`、`5.389390294099423e-12`，通过。S2 上 ols 是 `0`，mc 是 `6.375601524926835e-12`，通过；`mc_psd` 是 `0.0030837080887182624`，没有通过。S2/S3 没有把 IV 列入这项验收。

失败原因留在诊断里，没有改代码。同一配置重算：OLS 与 oracle 的 Gram 条件数是 `165946741.94092986`。`mc_psd` 的 `gram_min_eig` 是 `1.2136297572286345e-05`，`n_clipped=1`。现有 `1e-8 λ_max` 截断改动了一个特征值，相对误差落到 `0.003083708129612571`。`mc` 在 σ²=0 时没有截断，误差是 `6.490971852191922e-12`。没有放宽 `1e-8`，也没有改截断。

**S2 σ²：** 只报告，没有另设通过线。`beta=0.12736318407960198`。

| rho | sigma2_true | sigma2_estimated | relative_error | 干净谱第 N/2 项 / σ² |
|---|---|---|---|---|
| 0.01 | 5.7372844511005595e-06 | 6.050575699663931e-06 | 0.05460619065231706 | 2.722369953455067e-22 |
| 0.05 | 0.00014343211127751398 | 0.00014925730055461968 | 0.040612867127327305 | 1.1035912765850053e-23 |
| 0.2 | 0.0022949137804402237 | 0.0023548902203767556 | 0.02613450685935 | 6.994765761622271e-25 |

**pytest：** 76 通过，0 失败，23.07 秒。

**结果文件：** `results/e0/meta.json`、`solver_acceptance.json`、`s1_true_operator.json`、`zero_noise.json`、`s2_sigma2.json`、`summary.json`。`summary.json` 的 `pass` 是 false。这些文件保留。没有覆盖旧的 E0 输出，因为运行前目录不存在。

**未决：** 跨估计器发散阈值是否可比，仍留到 E1–E8 之前单独决定。E8 在 φ=0 时 `theory_gap_rel` 的口径仍未决定。S2 上 `mc_psd` 在 σ²=0 时被既有特征值截断推离 oracle，等待负责人决定。

## E0 MC-PSD Zero-Noise Amendment and Retry

**最终状态：** `E0_SANITY_CHECK_COMPLETE_AFTER_ZERO_NOISE_AMENDMENT`

上一节的失败记录保持原样。原始 `results/e0/` 没有被覆盖。重跑写入 `results/e0_retry1/`。两组 JSON 都被 `.gitignore` 的 `results/` 忽略，没有进入 Git。

**原始失败原因：** 02 要求 σ²=0 时 `mc_psd` 与 oracle 的相对误差小于 `1e-8`，同时又规定 `mc_psd` 始终把小于 `1e-8 λ_max` 的特征值截断。S2 干净 Gram 的 `λ_max=2013.9790491107137`，`λ_min=1.2136297572304265e-05`，下限 `2.0139790491107137e-05`。45 个特征值里只有最小的那一个低于下限。修改前普通 MC 的误差是 `6.490971852191922e-12`，`n_clipped=0`；MC-PSD 的误差是 `0.003083708129612571`，`n_clipped=1`。失败来自这次截断，不是 Hermite 或交叉矩。

**原始结果 SHA-256：**

| 文件 | 字节 | SHA-256 |
|---|---|---|
| `results/e0/meta.json` | 800 | `b596af93b6bfefc3a7a3ed1e5ebe4fa2bc4df4071939f272a020a2cc7aca0dcc` |
| `results/e0/solver_acceptance.json` | 2105 | `cb3d6601a4f0ba6a4b36b901d23e363e472fca135845d204fdbed3a1a84c82ac` |
| `results/e0/s1_true_operator.json` | 1173 | `2e3bba6324fbed6be58cdc04fe0162d2b02f9921d059069808ae8aa14c105daf` |
| `results/e0/zero_noise.json` | 3769 | `f267f5a08549cb90ec201c8f38cab291860e3c670e4ac73b2b9e068fcf64c41f` |
| `results/e0/s2_sigma2.json` | 2080 | `0a4d580e58a07175c5f3c2ab59e7de0929c7e3738e422af924d390f4dd4f0cd7` |
| `results/e0/summary.json` | 1010 | `3251504f1bc645ed64d4beb959ba17ea0f033a7d735a11d9b2525a665c8a99df` |

重跑之后这些哈希没有变化。`summary.json` 的 `pass` 仍是 false。S2 `mc_psd` 的原始误差仍是 `0.0030837080887182624`。

**负责人约定：** σ² 严格等于 0 时，`mc_psd` 使用与 `mc` 相同的未截断矩求解，`n_clipped=0`。σ² 严格为正时，`δ=1e-8` 的谱截断不变。很小但非零的 σ² 不当作 0。不用条件数决定是否截断。Gram 奇异时仍返回 NaN。

**修订提交：** `3b8f7d5db2ab927b6e2da6e4c4b1f095cb50f827`，`fix: define MC-PSD zero-noise identity`。只包含 `estimators.py`、02 第 3.2 节新增的一段、`tests/test_estimators.py`。

修订前 SHA-256：`estimators.py` `26c5458fb6f054e3abec754f39e469853e6c755a289d1e7bf1b8a5b70fd3c71c`，测试 `79e031d782022fb80df7b2c7fc64efa583a66a8d5f7d89da7190b6440ea0538a`，02 `6b3edefca1ef1c782f75eca8960fc6feb3bc04a7fb34e81f75d51f3bced64eb1`。

修订后 SHA-256：`estimators.py` `a4716e630850e93bafc0acb55197ad4e7dfef0758c8ba4c06a32ccccc0e381c4`，测试 `3d70521680a4b83bae19af78670cfc4858136ef55d9729af4e8b78bb90d07fb3`，02 `bd1ff6f50ba9add5ae4ad10149a552b2d117d28f38ad461d157c6f5839a17e8f`。

**测试：** 新增零噪声等价、S2 病态 Gram、正噪声截断与不截断、很小正 σ² 仍截断、奇异矩返回 NaN。完整 pytest 82 通过，0 失败，25.20 秒。旧的 76 项仍在。

**重跑：** 同一 `scripts/e0_sanity.py`，输出目录改为 `results/e0_retry1/`。配置仍是 `n_train=10`、`r=8`、原阈值、原种子。HEAD `3b8f7d5db2ab927b6e2da6e4c4b1f095cb50f827`。墙钟 12.598161600064486 秒。`summary.json` 的 `pass` 是 true。

S2 `mc_psd` 的误差从 `0.0030837080887182624` 变为 `6.375601524926835e-12`，与 MC 相同，`n_clipped=0`。S1 与 S3 的 `mc_psd` 也与对应的 MC 相同，分别是 `6.017318142555264e-12` 和 `4.584926680894034e-13`。分辨率、S1 真算子和三组 σ² 估计与原始失败运行的数值一致。正噪声截断规则没有改。

**重跑 SHA-256：**

| 文件 | 字节 | SHA-256 |
|---|---|---|
| `results/e0_retry1/meta.json` | 824 | `4cfc8860855fa0da84f32fe345c6c95f69548d08a6c9a8ea6572891627efc9f3` |
| `results/e0_retry1/solver_acceptance.json` | 2129 | `0467dec85498a6ad0f42c2289a6889c57354c224911adef2071555be4bff65c6` |
| `results/e0_retry1/s1_true_operator.json` | 1197 | `371121a2683a12591d01be9e52843331aa2ffd306e96b8f169697781ada9f20f` |
| `results/e0_retry1/zero_noise.json` | 4098 | `43e43b7aef7322abeb0636c0c21186bca942e7c072b21b0e175af0d15d278c8e` |
| `results/e0_retry1/s2_sigma2.json` | 2104 | `4ce552de5b86134bf62ee2e57135b8bdf39db535253e2f7d4660ef2659300be6` |
| `results/e0_retry1/summary.json` | 1033 | `a0b96c7f1d486c0c9e0b55ced1b57b1b5d50be2e94c83ad23f6c7852f0e4969a` |

没有 push。没有启动 E1–E8。跨估计器发散阈值，以及 E8 在 φ=0 时的 `theory_gap_rel`，仍然未决。

## E1：OLS 偏差的理论与模拟对照

状态：`E1_EXPERIMENT_COMPLETE_WITH_SCIENTIFIC_DISCREPANCY`。

核对时 HEAD 为 `d6834cc92cf5f7e7220d472941688a4f4ba30c12`，工作区干净。`results/e0/` 六份 JSON 的 SHA-256 与上一节失败记录一致，`pass=false`。`results/e0_retry1/` 六份与上表一致，`pass=true`。清单见 `reports/E0_CLOSEOUT_MANIFEST.md`。

网格是 S1、r=8、rho ∈ {0.02, 0.05, 0.1, 0.2}、n_train ∈ {10, 30, 100}、只做 OLS、R=200，共 2400 次拟合。训练轨迹是同一条 index 0–99 序列的嵌套前缀。噪声键是 `1,4,1,rho_index,replicate,traj`，rho_index 用主网格下标 2、3、4、5。`experiment_id=1` 是本次对 03 未赋值槽位的记录。σ² 由该前缀的 rho/RMS 规则计算，同一配置的 200 次重复共用。

先导是 rho=0.05、n_train=30、replicate 0–9，目录 `results/e1_pilot/`。检查通过。10 次拟合 0.5411713998764753 秒，外推 202.0373226205508 秒，低于 4 小时，网格未缩减。正式运行 16.38159259990789 秒，运行时 HEAD 仍是 `d6834cc92cf5f7e7220d472941688a4f4ba30c12`。2400 行全部 `ols`、`status=ok`，失败数 0。理论矩条件数在 3.32e5 到 9.86e6 之间，12 个极限都有限，诊断求解与 `ols_limit` 的相对差是 0。

`O` 的 `theory_gap_rel` 从 0.007733273489908429（rho=0.1，n_train=100）到 0.04700872169860776（rho=0.05，n_train=10）。n_train=100 时，均值与极限之差的元素 z 分数平方和接近 360 这个矩阵元素数；n_train=10 时该平方和是 1082 到 1825。后者大于 R=200 的重复噪声尺度，对应有限样本下“比值的均值”和“期望矩之比”的差别。A 的偏差范数随 rho 增大；H 在 n_train=10、rho 从 0.1 到 0.2 时略降。`||c_true||=9.07314923274536e-16`，所以 c 的 `theory_gap_abs` 数值很大，分子本身在 1e-6 到 1e-4。没有把 H1 写成已证实或已否定。

误差棒是偏差范数在对角协方差下的 delta-method 标准误的 ±2 倍，不是 `||SE||_F`。第一次写出的 `summary.csv` 没有给含逗号的 `seed_pattern` 加引号，列因此错位。随后只用已归档算子重写了汇总表，没有重拟合；重算的 `O_theory_gap_rel` 与表内值之差的最大值是 0。`runs.csv`、`operators.npz`、`meta.json` 保持原文件。

完整 pytest 在这版代码上是 94 passed，0 failed，24.92 秒。结果目录被忽略。输出哈希见 `reports/E1_OUTPUT_MANIFEST.md`，分析见 `reports/E1_OLS_BIAS.md`。没有启动 E2–E8。没有修改已验收的理论和估计器。

## E2 估计器比较

状态：`E2_EXPERIMENT_COMPLETE_WITH_SCIENTIFIC_DISCREPANCY`。E1 的数值、图和清单没有重算。进入 E2 前，E0、E0_retry1、E1、E1_pilot 的 SHA-256 与既有清单一致，HEAD 是 `9bc73f190ba92c11d9b969348fd02cc15e6e9639`，工作区干净。`results/e2/` 原先不存在。

**发散阈值。** 旧口径是每个估计器用自己的训练轨迹计算 `1e3 * max ||q||_∞`：oracle 用干净轨迹，其余用带噪轨迹，因此比较时的发散线可以不同。新口径从 E2 起生效：每个 `n_train` 只用该样本量的干净训练前缀，`B=1e3 * max ||q_k||_∞`。同一 `n_train` 下，七个估计器、全部 rho 和全部 replicate 共用这个数。验证集、测试集和预测结果不参与。训练数据不变：oracle 仍拟合干净轨迹，其余六个仍拟合同一份带噪轨迹。历史结果不按新口径重写。三个阈值是 n=10 的 `963.9619697494801`、n=30 的 `1189.4190516808758`、n=100 的 `1570.4767311563792`。`02` 第 7 节只加了这一段说明。`rollout.py` 的预测方程和 smoke 的按估计器阈值都没有改。

正式网格是 S1、r=8、rho 为主网格六档、n_train 为 10/30/100、七个估计器、R=100、`experiment_id=2`，共 12600 行。验证噪声把 `traj_index` 加 100。先导是 rho=0.05、n_train=30、replicate 0 和 1，写入 `results/e2_pilot/`。先导通过。两次重复 6.8555055998731405 秒，按重复外推 6169.955039885826 秒，按配对数加权 9597.707839822397 秒，低于 4 小时，R 未改。正式运行 1400.5735625999514 秒，12600 行，失败拟合 0。运行时 HEAD 仍是 `9bc73f190ba92c11d9b969348fd02cc15e6e9639`，因为代码提交发生在运行之后。pytest 在运行前是 106 passed，0 failed，25.10 秒。

算子误差相对 `O_true`。rho=0 时 oracle、OLS、MC、MC-PSD、IV、IV2 的 `err_O_rel` 中位数都小于 1e-8；Ridge 不要求等于 oracle，n=10 时是 `1.5360367460162803e-08`。n_train=30 和 100 上，MC、IV、IV2 的偏差低于 OLS，并随样本量下降。n_train=10 且 rho≥0.05 时，MC、MC-PSD 和 IV 的误差、方差和发散比例同时变大，MC-PSD 在 rho=0.2 的 `err_O_rel` 中位数是 408。这是样本不足下的不稳定，不是把 E1 的有限样本差当成公式错误后去改估计器。详细数字见 `reports/E2_ESTIMATOR_COMPARISON.md`。E2 随后按 `E2_AUDIT_PASS_WITH_LIMITATIONS` 冻结，没有重跑。`theory.py` 与 `estimators.py` 的 SHA-256 未变。

## E3 Burgers POD 估计器比较

状态：`E3_EXPERIMENT_COMPLETE_WITH_SCIENTIFIC_DISCREPANCY`。E0–E2 的数据和已验收数学实现没有改。

正式运行前的提交是 `ed0006a4d3d37e81091c86d443423cc2be266903`。先导 `results/e3_pilot/` 为 60 行，外推 135.7364175999537 秒，低于 4 小时，网格未缩减。正式目录 `results/e3/` 有 3600 行，失败 0，墙钟 302.1770353000611 秒。运行时 HEAD 就是上述提交，开始和结束的源码 SHA-256 一致。pytest 为 115 passed、0 failed。独立复算 `AUDIT_PASS`。

参考是各 POD 基上的 `O_clean`。noisy basis 的参考随重复改变，所以该基不报告跨重复偏差。rho≥0.02 时 MC、IV、IV2 相对 OLS 的算子误差配对区间整段为负；rho=0.01 时只有 MC 如此。clean basis、rho=0.2 的 `sd_O_rel` 最大约 0.023，没有复现 E2 在 n_train=10 上的方差爆炸。MC-PSD 的 `n_clipped` 全是 0。trailing median 的 σ² 相对误差小于 Gavish–Donoho，但这个差别几乎不改变算子误差。Oracle 的全视界中位数约为 0.134，与池化投影误差同一量级。H3 的谱半径纠正只表现为约 0.0017 的偏移，MC/IV 还落到参考的另一侧。H4 在高噪声下 MC/IV 的能量违反小于 OLS，但 `O_clean` 本身的违反已经约为 0.044。

图是 `figures/fig_e3_operator_err.png`、`fig_e3_rollout.png`、`fig_e3_basis_effect.png`、`fig_e3_sigma_source.png`。分析见 `reports/E3_BURGERS_POD_COMPARISON.md`，哈希见 `reports/E3_OUTPUT_MANIFEST.md`。没有启动 E4–E8。
