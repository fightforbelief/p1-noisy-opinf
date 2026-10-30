# AR(1) MC 极限公式规格修订记录

修订日期：2026-10-29。

本次只改 `02_数学与估计器规格.md` 第 6 节。依据是同目录的 `AR1_MC极限公式冲突审阅.md`（结论 CONFIRMED）以及负责人对其中第 10 节最小修改的批准。原审阅报告保留，本记录不覆盖其中的数值证据，也没有重跑 Monte Carlo。

## 修订原因

原第 6 节写「修正后的二次特征与下一时刻噪声的相关为零」，并把

\[
\Delta=n\phi\sigma^2\begin{bmatrix}I_r\\0\\0\end{bmatrix}
\]

的二次块写成 0。三阶矩只消去纯噪声项。信号与噪声的二阶交叉 \(q_i\varepsilon_j\)、\(q_j\varepsilon_i\)、\(2q_i\varepsilon_i\) 仍在，二次块因此不为零。线性块、常数块，以及 \(E[\hat G]=\bar G_0\)，原来就是对的。

## 修改前

\[
\hat O_{\mathrm{MC}}\to O_{\mathrm{clean}}+\bar G_0^{-1}\Delta,
\qquad
\Delta=n\phi\sigma^2\begin{bmatrix}I_r\\0\\0\end{bmatrix}.
\]

\(\bar G_0=\sum_k z(q_k)z(q_k)^\top\)。求和范围没有写成「只含配对输入」。二次相关被写成 0。箭头没有区分期望之比、概率极限和 \(E[\hat G^{-1}\hat C]\)。

## 修改后

\(P\) 是进入 \(\hat C\) 与 \(\bar G_0\) 的配对输入：每条轨迹 \(k=0,\ldots,K-1\)，不含末端快照，不跨轨迹。\(n=|P|\)，\(S_i=\sum_{k\in P}q_{k,i}\)。

\[
\Delta=
\begin{bmatrix}
n\phi\sigma^2 I_r\\
D_2(S)\\
0_{1\times r}
\end{bmatrix}
\in\mathbb R^{p\times r}.
\]

- 线性行：\(\Delta_{i,b}=n\phi\sigma^2\delta_{ib}\)。
- 二次行 \((i,j)\)，\(0\le i\le j\le r-1\)，字典序，行号 \(r+\mathrm{pos}(i,j)\)：\(\Delta_{(i,j),b}=\phi\sigma^2(\delta_{jb}S_i+\delta_{ib}S_j)\)。
- 对角：\(\Delta_{(i,i),b}=2\phi\sigma^2 S_i\delta_{ib}\)。
- 常数行：0。
- \(\phi=0\) 时 \(\Delta=0\)。

\[
\bar G_0=\sum_{k\in P}z(q_k)z(q_k)^\top,
\qquad
C_0=\sum_{k\in P}z(q_k)q_{k+1}^\top,
\]

\[
O_{\mathrm{lim}}=\bar G_0^{-1}(C_0+\Delta)=O_{\mathrm{clean}}+\bar G_0^{-1}\Delta,
\qquad
O_{\mathrm{clean}}=\bar G_0^{-1}C_0.
\]

\(O_{\mathrm{lim}}=E[\hat G]^{-1}E[\hat C]\)。\(n\to\infty\) 且 \(\bar G_0/n\) 收敛到可逆极限时，它也是 \(\hat O_{\mathrm{MC}}\) 的概率极限。它不是有限样本的 \(E[\hat G^{-1}\hat C]\)。

## 逐块推导摘要

干净轨迹固定，只对测量噪声取期望。\(\varepsilon_+=\phi\varepsilon+\sqrt{1-\phi^2}\,\xi\)，\(\xi\) 与 \(\varepsilon\) 独立、零均值、协方差 \(\sigma^2 I\)。\(q_{k+1}\) 不是随机的。

**Gram。** \(\hat G\) 只用同一时刻。第 2.3 节每个时刻的边际都是 \(\mathcal N(0,\sigma^2 I)\)。Hermite 无偏性给出 \(E[\mathcal H_\alpha(\tilde q_k)]=q_k^\alpha\)，对次数不超过 4 的 \(\alpha\) 也成立。对 \(k\in P\) 求和得到有限样本等式 \(E[\hat G]=\bar G_0\)。

**线性。** \(\hat z_i=q_i+\varepsilon_i\)，\(E[\hat z_i\varepsilon_{+,b}]=\phi\sigma^2\delta_{ib}\)。对 \(n\) 个配对求和得到线性块。

**常数。** 特征是 1，\(E[\varepsilon_+]=0\)。

**二次，\(i\neq j\)。** \(\hat z=(q_i+\varepsilon_i)(q_j+\varepsilon_j)\)。独立零均值高斯的三阶矩 \(E[\varepsilon_i\varepsilon_j\varepsilon_b]=0\)，剩下

\[
E[\hat z\,\varepsilon_{+,b}]=\phi\sigma^2(q_i\delta_{jb}+q_j\delta_{ib}).
\]

**二次，\(i=j\)。** \(\hat z=(q_i+\varepsilon_i)^2-\sigma^2\)。\(E[(\varepsilon_i^2-\sigma^2)\varepsilon_+]=0\)，剩下 \(E[\hat z\,\varepsilon_{+,b}]=2q_i\phi\sigma^2\delta_{ib}\)。减去 \(\sigma^2\) 不改变与零均值 \(\varepsilon_+\) 的协方差。对 \(k\in P\) 求和得到 \(D_2(S)\)。末端快照只作为 \(q_{k+1}\)，不进入 \(S_i\)。

数值核对仍以审阅报告第 8 节为准，本次没有新的计算。

## 影响范围

- 改动的权威公式只有 `02` 第 6 节的 \(\Delta\) 及其极限含义。
- `mc`、`mc_psd`、`ols_limit`、`iv`、`iv2` 的定义未改。第 3.2 节的 \(\hat z\)、\(\hat G\)、\(\hat C\) 未改。AR(1) 不另加估计器修正。
- 白噪声下 \(\phi=0\)，\(\Delta=0\)，第 3.2 节的交叉矩公式保持原样。
- E0–E7 未改。`00`、`03`、`04`、`05`、`07` 未改。
- `03` 仍把 `mc_limit_ar1` 指向第 6 节，并允许理论函数在内部做 \(\kappa\) 缩放再换回。权威表达式写在原坐标。以后实现时，若内部使用 \(y=q/\kappa\)，须用 \(\sigma^2/\kappa^2\) 重算 \(S_i\) 与 \(\Delta\)，再把算子换回。本次不实现该函数，也不改 `03`。
- 没有 Python 代码、测试、实验或 Git 提交。

## 尚未解决：E8 在 \(\phi=0\) 的 `theory_gap_rel`

`04` 的 E8 网格含 \(\phi=0\)，`theory_gap_rel` 与 E1 相同，分母是 \(\|O_{\mathrm{lim}}-O_{\mathrm{true}}\|_F\)。\(\phi=0\) 时新旧公式都给出 \(\Delta=0\)。S1 上 \(O_{\mathrm{lim}}\) 与 \(O_{\mathrm{true}}\) 只差干净 OLS 的数值误差，分母接近 0，该比值没有稳定含义。

本次不决定这个网格点的填报口径。`05` 的 `test_mc_limit_ar1` 使用 \(\phi=0.5\)，不由这一点决定，本次也没有改 `05`。

审阅报告另记了一件不在本次最小修订内的事：算子范数上，二次块误差随 \(S_i/n\) 变化；坐标均值接近 0 时，原公式可能落在 10% 算子宽容度内。是否另加非零均值对照，仍未决定，也不属于本修订。

## 复核澄清（2026-10-29）

负责人复核决定：`APPROVE_WITH_ONE_CLARIFICATION`。\(\Delta\) 的分块、求和范围、\(\phi=0\) 以及有限样本期望之比与 \(E[\hat G^{-1}\hat C]\) 的区分均保持不变。

补全前，第 6 节只写「\(n\to\infty\) 且 \(\bar G_0/n\) 收敛到可逆极限时，该矩阵也是 \(\hat O_{\mathrm{MC}}\) 的概率极限」。这不够：有限样本恒等式 \(E[\hat G]^{-1}E[\hat C]=\bar G_0^{-1}(C_0+\Delta)\) 对每个 \(n\) 都成立，但右边随 \(n\) 变化。概率极限是这些矩的极限，因此还要 \(C_0/n\to C_\infty\)、\(\Delta/n\to\Delta_\infty\)，并且噪声下的 \(\hat G/n\)、\(\hat C/n\) 满足相应大数定律，才能得到

\[
\hat O_{\mathrm{MC}}\xrightarrow{p} G_\infty^{-1}(C_\infty+\Delta_\infty).
\]

`theory.mc_limit_ar1` 仍返回有限样本的 \(O_{\mathrm{lim}}=\bar G_0^{-1}(C_0+\Delta)\)，在上述收敛下它趋向同一个极限。本次只改了这一处条件，没有改 \(\Delta\)，没有改其他规格，没有实现该函数。E8 在 \(\phi=0\) 的 `theory_gap_rel` 口径仍未决定。
