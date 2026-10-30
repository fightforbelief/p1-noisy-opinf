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
