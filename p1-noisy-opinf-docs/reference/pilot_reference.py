# ============================================================================
# 仅供参考的前期 pilot 代码 —— 不要直接复用
#
# 已知缺陷：
#   1. "corrected" 方法只修正了 Gram 矩阵，没有修正交叉矩（应使用 02 第 3.2 节的 \hat z_k）。
#      这会留下明显的偏差（见 02 第 3.6 节：0.0277 vs 0.0006）。
#   2. IV 没有做常数项修正。
#   3. 没有坐标缩放（02 第 1.4 节），r 较大时 Gram 矩阵可能病态。
#   4. 初值、种子、噪声定义、指标都与正式规格不同。
#   5. Hermite Gram 的计算方式是 O(p^2) 次逐元素循环，速度很慢；正式实现要按 alpha 去重。
#   6. 用的是 RK4 + 小步长，正式实现用 ETDRK4（03 第 4.2 节）。
#
# 它的用途只是说明问题的形态。原始 pilot 输出（r=8，30 条训练轨迹，5 次重复）：
#   clean OLS 预测误差中位数 0.145（POD 投影下限 0.125）
#   10% 噪声：ols 0.216 | 只修 Gram 的 mc 0.217 | iv 0.171
# ============================================================================
"""Pilot: does measurement noise bias quadratic discrete OpInf, and can a moment
(Hermite/Wick) correction or a lagged instrumental variable fix it? 1D viscous Burgers."""
import numpy as np, itertools
rng = np.random.default_rng(0)
N, nu, L = 256, 0.01, 1.0
x = np.linspace(0, L, N, endpoint=False); k = 2*np.pi*np.fft.rfftfreq(N, d=L/N)
def rhs(u):
    uh = np.fft.rfft(u); ux = np.fft.irfft(1j*k*uh, N); uxx = np.fft.irfft(-k**2*uh, N)
    return -u*ux + nu*uxx
def simulate(u0, nsteps=200, dt_obs=0.005, sub=50):
    h = dt_obs/sub; U = [u0]; u = u0.copy()
    for _ in range(nsteps):
        for _ in range(sub):
            a=rhs(u); b=rhs(u+h/2*a); c=rhs(u+h/2*b); d=rhs(u+h*c); u = u + h/6*(a+2*b+2*c+d)
        U.append(u.copy())
    return np.array(U)
def ic():
    u = np.zeros(N)
    for m in range(1, 4): u += rng.normal()/m*np.sin(2*np.pi*m*x + rng.uniform(0, 2*np.pi))
    return 0.5*u
train = [simulate(ic()) for _ in range(30)]; test = [simulate(ic()) for _ in range(10)]
S = np.concatenate(train).T
r = 8
V = np.linalg.svd(S, full_matrices=False)[0][:, :r]
idx = [(i, j) for i in range(r) for j in range(i, r)]
def feats(Q):   # rows: [q, q_i q_j (i<=j), 1]
    return np.hstack([Q, np.stack([Q[:, i]*Q[:, j] for i, j in idx], 1), np.ones((len(Q), 1))])
nf = r + len(idx) + 1
# exponents of each feature as a multi-index
E = np.zeros((nf, r), int)
for a in range(r): E[a, a] = 1
for m, (i, j) in enumerate(idx): E[r+m, i] += 1; E[r+m, j] += 1
def He(n, z, s2):
    return [np.ones_like(z), z, z**2-s2, z**3-3*s2*z, z**4-6*s2*z**2+3*s2**2][n]
def corrected_gram(Qt, s2):
    G = np.zeros((nf, nf))
    for a in range(nf):
        for b in range(a, nf):
            al = E[a]+E[b]; v = np.ones(len(Qt))
            for c in np.flatnonzero(al): v = v*He(al[c], Qt[:, c], s2)
            G[a, b] = G[b, a] = v.sum()
    return G
def fit(trajs, method, s2=0.0, lam=1e-8):
    Zs, Ys, Ws, Qs = [], [], [], []
    for Q in trajs:
        if method == "iv":
            Zs.append(feats(Q[1:-1])); Ws.append(feats(Q[:-2])); Ys.append(Q[2:])
        else:
            Zs.append(feats(Q[:-1])); Ys.append(Q[1:]); Qs.append(Q[:-1])
    Z, Y = np.vstack(Zs), np.vstack(Ys)
    if method == "ols":
        G = Z.T@Z
    elif method == "corrected":
        G = corrected_gram(np.vstack(Qs), s2)
    if method in ("ols", "corrected"):
        return np.linalg.solve(G + lam*np.eye(nf), Z.T@Y)
    W = np.vstack(Ws)
    return np.linalg.lstsq(W.T@Z + lam*np.eye(nf), W.T@Y, rcond=None)[0]
def rollout(O, q0, n):
    Q = [q0]
    for _ in range(n):
        q = feats(Q[-1][None])[0]@O
        if not np.all(np.isfinite(q)) or np.abs(q).max() > 1e6: return None
        Q.append(q)
    return np.array(Q)
def score(O):
    errs = []
    for U in test:
        Qc = U@V; P = rollout(O, Qc[0], len(U)-1)
        errs.append(np.inf if P is None else np.linalg.norm(P@V.T-U)/np.linalg.norm(U))
    errs = np.array(errs); return np.median(errs), int(np.isinf(errs).sum())
clean = [U@V for U in train]
print("projection error floor:", np.median([np.linalg.norm(U@V@V.T-U)/np.linalg.norm(U) for U in test]))
print("clean OLS:", score(fit(clean, "ols")))
rms = np.sqrt(np.mean(S**2))
for lvl in [0.02, 0.05, 0.1, 0.2]:
    sig = lvl*rms
    res = {m: [] for m in ["ols", "corrected", "iv"]}
    for rep in range(5):
        noisy = [(U + sig*rng.normal(size=U.shape))@V for U in train]
        for m in res: res[m].append(score(fit(noisy, m, s2=sig**2)))
    print(f"noise {lvl:.0%}: " + " | ".join(f"{m}: med err {np.median([a for a,_ in v]):.3f}, blowups {sum(b for _,b in v)}/50" for m, v in res.items()))
