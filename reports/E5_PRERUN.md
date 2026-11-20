# E5 第 1 部分运行前协议

本文件写在无噪声拟合之前。它只覆盖 E5 第 1 部分。第 2 部分、E6–E8 不在本轮执行。

E4 保持冻结：

- 状态：`E4_AUDIT_PASS_WITH_SCIENTIFIC_DISCREPANCY`
- H6：`H6_PARTIALLY_SUPPORTED`。不把 H6 改成完全支持，不把 κ=64 写成通用常数，不改 H6 原文。
- 冻结提交：`4ab0dc996976a782a3b65519f4782e7835a23605`，说明 `docs: finalize E4 scientific audit`
- 报告：`reports/E4_POSTRUN_AUDIT_AND_SCIENTIFIC_INTERPRETATION.md`
- 报告 SHA-256：`ec65ea47902a8c5dd65b08d346c718ac4929e88e9139a5152f4d1705ef2361fc`

当前分支是 `master`。本轮开始时 HEAD 就是上述提交。工作区里原有的 `manifests/pre_implementation_baseline.json` 和未跟踪的 `uploads/` 不进入 E5 变更。本轮不自动提交。

## 1. 假设、问题与公式来源

`01_研究问题与假设.md` 的 H5 原文是：

> 存在截断误差时（Burgers POD），IV 的目标与无噪声 OLS 不同，偏离量随秩 \(r\) 增大而减小。

RQ4 问的是：PDE 的 POD 降阶动力学不是精确二次时，MC 和 IV 分别收敛到什么，哪一个更接近无噪声数据上的 OLS。

三个估计目标按 `01` 第 4 节区分。S2 没有可得的 \(O_{\text{true}}\)。本部分只比较干净数据上的有限样本目标，不把 \(n_{\text{train}}=100\) 的差异写成 H5 的渐近结论。

公式来自 `02` 第 3.3 节，实验网格来自 `04` 的 E5 第 1 部分：

- \(O_{\text{clean}}\)：干净训练坐标上的 OLS，也就是 `oracle`。
- \(O_{\text{IVc}}\)：同一个 `iv` 估计器用于干净数据，不做常数修正。
- \(O_{\text{IV2c}}\)：同一个 `iv2` 估计器用于干净数据，不做常数修正。

实现上三者都调用现有 `fit`，不改求解公式。`iv` / `iv2` 传入 `sigma2=0` 和 `correct_intercept=False`。σ² 为 0 时常数修正本身是零，显式关闭是为了和“不做常数修正”的原文一致。

## 2. 系统、数据、种子与时间窗

系统是 S2。参数哈希 `219c52f59203ad7fb468cec9e59f4e47d58f632105a0056e54cef5d252ba4dfa`，与已经验收的 S2 缓存相同。每条轨迹形状 `(201, 256)`，观测步数 \(K=200\)，时间下标 `0..200`。

划分：

- 训练：index `0..99`，共 100 条。`0..29` 是 E0–E4 已经使用的缓存，文件时间未改。`30..99` 由现有 `data.get_splits` 按同一生成器补齐，没有跳号。
- 测试：index `0..49`，共 50 条，沿用已有缓存。
- 验证：本部分不用，`n_val=0`，不读、不写验证缓存。
- 训练和测试的 index 都从 0 起编，编号可以相同。种子含 `split_id`，所以 train 0 与 test 0 不是同一条轨迹。运行时若这两条轨迹完全相同则停止。POD 和拟合只接收训练轨迹。

本部分 `rho=0`，不调用噪声生成器，也没有 replicate。初值种子仍是生成器里的 `(system_id=2, split_id, index)`，train 的 split_id 是 1，test 是 3。

训练快照串联后的 SHA-256：`093f84934b9c306b32b6960aba4d78016687b26fa0f6492ae3b5762a70b55f72`。

测试快照串联后的 SHA-256：`a00b692d2ff5e1f2a94630c1dc0176463d46c6839fa3dca2ba7dbcfe0851eacc`。

## 3. Clean POD

基只来自 100 条干净训练轨迹的全部快照。不中心化，一次 `np.linalg.svd`，`r_max=20`。候选秩 `4, 6, 8, 10, 12, 16, 20` 取这组基的前 r 列。不把不同秩的算子相减。

每个秩另做一次独立的 r 维 SVD，只记录与前缀的最大绝对差，以及按列对齐符号后的差。这是诊断。正式基始终是 20 维 SVD 的前缀。

## 4. 样本窗口

一条轨迹有 201 个状态时，三个正式拟合使用的行数不同。滞后不跨轨迹。

| 目标 | 每条轨迹行数 | 100 条合计 | 回归量下标 | 输出下标 | 工具变量下标 |
|---|---:|---:|---|---|---|
| \(O_{\text{clean}}\) | 200 | 20000 | `0..199` | `1..200` | 无，设计矩阵就是回归量 |
| \(O_{\text{IVc}}\) | 199 | 19900 | `1..199` | `2..200` | `0..198` 的完整二次特征 |
| \(O_{\text{IV2c}}\) | 198 | 19800 | `2..199` | `3..200` | `1..198` 与 `0..197` 的特征去掉常数项，再加一列 1 |

这与 `estimators._one_instrument` 的切片一致：`iv` 使用 `traj[1:-1]`、`traj[:-2]`、`traj[2:]`；`iv2` 使用 `traj[2:-1]`、`traj[1:-2]`、`traj[:-3]`、`traj[3:]`。

正式偏离的分母和被减数都是全窗口 \(O_{\text{clean}}\)。样本窗口不同，所以 \(O_{\text{IVc}}-O_{\text{clean}}\) 同时包含估计目标差和可用行差。下面的对照只标成诊断，不替换正式列：

- 把每条轨迹切成 `traj[1:]` 后做 OLS，行数与 IV 相同。记录 \(O_{\text{IVc}}\) 相对这个窗口 OLS 的偏离。
- 把每条轨迹切成 `traj[2:]` 后做 OLS，行数与 IV2 相同。记录 \(O_{\text{IV2c}}\) 相对这个窗口 OLS 的偏离。
- 把正式 \(O_{\text{clean}}\) 代到这两个较短窗口上，另记一步残差。这不是新的 `closure_rel`。

## 5. 指标

正式 `closure_rel` 是 \(\|Y-ZO_{\text{clean}}\|_F/\|Y\|_F\)。\(Z,Y\) 来自上面的 OLS 全窗口。它是既定二次模型对干净训练数据的一步拟合残差。它不是 POD 投影误差，也不能在没有进一步分解时排除时间离散等其他模型形式误差。

测试集上的池化投影残差另存为 `pooled_proj_reference`，不叫 closure。

偏离：

- 绝对偏离 \(\|O_{\text{IVc}}-O_{\text{clean}}\|_F\)，A、H 块同样，始终保留。
- 相对偏离是绝对偏离除以对应参考块的 Frobenius 范数。
- 参考块范数 `< 1e-12` 时，相对偏离写 NaN，状态写 `SMALL_REFERENCE`。`1e-12` 是本轮报告规则，不是估计器公式的一部分。

预测：三个算子都从干净测试初值出发，迭代 200 步，与干净全阶测试轨迹比较。提升使用该秩的 POD 前缀。发散阈值是该秩干净投影训练状态最大绝对值的 \(10^3\) 倍，三个算子共用。发散轨迹的全视界误差保持 inf。`fit_status=ok` 只表示算子有限；`rollout_stable` 才表示发散比例为 0 且全视界中位数有限。

## 6. 诊断，不改求解

正式求解保持现有代码：

- OLS：`np.linalg.lstsq(Z, Y, rcond=None)`，不用伪逆包装。
- IV：`np.linalg.solve(W.T @ Z, W.T @ Y)`。失败时返回 NaN。
- IV2：第一阶段 `lstsq(W, Z, rcond=None)`，第二阶段 `lstsq(Zhat, Y, rcond=None)`。数值秩不足时返回 NaN。

`fit` 先用状态的最大绝对值做坐标缩放，再把算子换回原坐标。条件数诊断按同一次缩放复现矩阵，避免和求解器的矩阵不是同一个。保存的是维数、奇异值、数值秩和条件数，不保存 2 万行的设计矩阵。数值秩的定义是：奇异值大于 `eps * max(m, n) * smax` 的个数，与 NumPy `lstsq` 在 `rcond=None` 时的截止相同。这个截止只用于诊断。IV 的 `solve` 本身没有这个截止。

已有诊断照原样保存：`instrument_cond = cond(W.T @ Z)`，`first_stage_r2_median` 是设计矩阵各列被工具变量线性投影后的决定系数的中位数。不另造 first-stage F。

残差交叉矩在原坐标计算。正式 \(O_{\text{clean}}\) 在各方法自己的样本行上形成残差 \(e\)，再保存 \(W^\top e\)。OLS 行上 \(W=Z\)，所以这一项检查正规方程残差。IV 行上它是干净 OLS 残差与滞后工具变量的交叉矩。

`used_pinv` 对三个正式求解都记 0。

## 7. 将要执行的源码

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `configs/e5_part1.yaml` | 438 | `211a4fe54ae866591ff6d5830792321ee77385c1fe44b9fa75466976d9e81da9` |
| `src/noisyopinf/e5.py` | 14130 | `047775dbeb97a78d9898df9ec0e623699305ece7e27f7103325180713c370bb0` |
| `scripts/e5_part1.py` | 10498 | `1d2778559dfefa7b309df98902fbb42ff27dd2952ae323b823b2d9373db8ce68` |
| `tests/test_e5.py` | 4377 | `7d133d10b3c8c2ffb8113545a095bcceb598cbe75076b4f8ec7f6e2c1cd573e2` |
| `src/noisyopinf/estimators.py` | 11187 | `a4716e630850e93bafc0acb55197ad4e7dfef0758c8ba4c06a32ccccc0e381c4` |

估计器文件与 E4 冻结时的字节一致，本轮没有改公式。

输出目录：

- 端到端检查：`results/e5_part1_smoke`，只跑 r=6。
- 正式结果：`results/e5_part1`，七个秩。
- 目录若已有文件，脚本拒绝覆盖。

运行命令：

```text
.\.venv\Scripts\python.exe scripts\e5_part1.py smoke
.\.venv\Scripts\python.exe scripts\e5_part1.py run
```

单元测试在拟合之前已经通过：`tests/test_e5.py` 当时为 8 passed。

## 8. 运行后的诊断修正

正式算子和 `metrics.csv` 由当时的 `src/noisyopinf/e5.py`（`047775dbeb97a78d9898df9ec0e623699305ece7e27f7103325180713c370bb0`）生成。第一次写出的 OLS 谱误用了缩放后的状态矩阵，而不是二次特征矩阵。求解公式没有改。之后只重写了 `spectra.npz`、`residual_cross.npz`、`diagnostics.csv` 和 manifest。修正后的源码哈希是：

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `src/noisyopinf/e5.py` | 14165 | `e06bceeed252cfcfc5e0e48426fe5aba0e351c283a45a2b309240b5abaddbdff` |
| `scripts/e5_part1.py` | 12143 | `1a0ffa44af986c7afe51323beab71b0eb64c0f05f2d6a2aa442b6280dfb17517` |
| `tests/test_e5.py` | 4732 | `265cb19d449c188c1c816068833e5f50bad66e7c75842137232df02a1a4d815a` |
