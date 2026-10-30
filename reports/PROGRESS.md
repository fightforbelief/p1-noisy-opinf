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
