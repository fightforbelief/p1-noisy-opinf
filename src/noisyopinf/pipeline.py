"""一次 S2 配置的估计流程。不是 E0–E8 网格。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from noisyopinf.data import SPLIT_ID
from noisyopinf.estimators import fit
from noisyopinf.io_utils import RUN_COLUMNS
from noisyopinf.metrics import (
    ep_violation,
    err_A_rel,
    err_H_rel,
    err_O_rel,
    err_c_abs,
    err_c_scaled,
    proj_floor,
    rollout_aggregate,
    rollout_err_h,
    specrad_trio,
    state_rms,
)
from noisyopinf.noise import sigma_from_rho
from noisyopinf.pod import pod_basis, project
from noisyopinf.protocol import noise_replicate, select_ridge_lambda
from noisyopinf.rollout import rollout, threshold_from_train
from noisyopinf.systems.burgers import linear_symbol
from noisyopinf.systems.config import load_system
from noisyopinf.systems.spectral import sine_initial, solve

ESTIMATORS = ("oracle", "ols", "ridge", "mc", "mc_psd", "iv", "iv2")
SIGMA_METHODS = {"mc", "mc_psd", "iv", "iv2"}
N_VAL = 20
N_TEST = 50
ALWAYS_METRICS = (
    "err_O_rel", "err_A_rel", "err_H_rel", "err_c_abs", "err_c_scaled",
    "specrad_hat", "specrad_ref", "specrad_err", "ep_violation", "ep_violation_ref",
    "rollout_med", "rollout_med_h10", "rollout_mean_finite", "diverged_frac", "proj_floor",
)


@dataclass
class SmokeCase:
    """一次小规模 S2 重复所需的干净数据、带噪数据和参考算子。"""

    rho: float
    rank: int
    replicate: int
    sigma2: float
    sigma: float
    basis: np.ndarray
    threshold: float
    reference: np.ndarray
    train_u: list
    test_u: list
    noisy_train_u: list
    noisy_val_u: list
    clean_train_q: list
    noisy_train_q: list
    noisy_val_q: list
    clean_test_q: list
    clean_test_u: list


def simulate_s2(split: str, index: int, n_obs: int) -> np.ndarray:
    """用 yaml 里的 S2 物理参数积分 n_obs 个观测步。返回 (n_obs+1, N)。"""
    cfg = load_system("s2")
    width = int(cfg["N"])
    step = float(cfg["dt"])
    substeps = int(round(float(cfg["dt_obs"]) / step))
    initial = sine_initial(width, 2, SPLIT_ID[split], int(index), float(cfg["amplitude"]))
    field = solve(initial, linear_symbol(width, float(cfg["nu"])), step, substeps, int(n_obs))
    if not np.isfinite(field).all():
        raise RuntimeError(f"non-finite S2 trajectory {split}/{index}")
    return field


def prepare_case(rank: int, n_train: int, n_obs: int, rho: float, replicate: int) -> SmokeCase:
    """生成干净轨迹，在全阶上加噪，再投影到干净训练 POD 基。"""
    train_u = [simulate_s2("train", index, n_obs) for index in range(n_train)]
    val_u = [simulate_s2("val", index, n_obs) for index in range(N_VAL)]
    test_u = [simulate_s2("test", index, n_obs) for index in range(N_TEST)]
    sigma = sigma_from_rho("s2", rho, train_u)
    noisy_train_u, noisy_val_u, clean_test_u = noise_replicate(train_u, val_u, test_u, sigma, replicate)
    basis, _ = pod_basis(train_u, rank)
    clean_train_q = _project_all(train_u, basis)
    reference = fit(clean_train_q, "oracle").O
    if not np.isfinite(reference).all():
        raise RuntimeError("clean reference operator is non-finite")
    return SmokeCase(
        rho=float(rho), rank=int(rank), replicate=int(replicate), sigma2=float(sigma) ** 2,
        sigma=float(sigma), basis=basis, threshold=threshold_from_train(_project_all(noisy_train_u, basis)),
        reference=reference, train_u=train_u, test_u=test_u, noisy_train_u=noisy_train_u,
        noisy_val_u=noisy_val_u, clean_train_q=clean_train_q,
        noisy_train_q=_project_all(noisy_train_u, basis), noisy_val_q=_project_all(noisy_val_u, basis),
        clean_test_q=_project_all(clean_test_u, basis), clean_test_u=clean_test_u,
    )


def training_trajectories(case: SmokeCase, method: str) -> list:
    """oracle 用干净训练轨迹。其余估计器共用该重复的带噪训练轨迹。"""
    if method == "oracle":
        return case.clean_train_q
    return case.noisy_train_q


def evaluate_case(case: SmokeCase) -> list[dict]:
    """带噪估计器共用同一份训练轨迹。只有 ridge 读取验证轨迹。"""
    chosen = select_ridge_lambda(case.noisy_train_q, case.noisy_val_q, case.threshold)
    return [_one_row(case, method, chosen) for method in ESTIMATORS]


def judgement(row: dict, applicable: tuple[str, ...]) -> tuple[str, str | None]:
    """区分未定义、不适用、失败和可以计算的指标。"""
    missing = [name for name in applicable if _kind(row.get(name)) == "missing"]
    undefined = [name for name in applicable if _is_undefined(row, name)]
    if missing:
        return "failed", "missing " + ",".join(missing)
    if undefined:
        return "undefined", "undefined " + ",".join(undefined)
    return "ok", None


def _project_all(fields, basis: np.ndarray) -> list[np.ndarray]:
    return [project(field, basis) for field in fields]


def _one_row(case: SmokeCase, method: str, chosen: float) -> dict:
    lam = chosen if method == "ridge" else None
    variance = case.sigma2 if method in SIGMA_METHODS else None
    train_q = training_trajectories(case, method)
    result = fit(train_q, method, sigma2=variance, lam=lam)
    row = _identity(case, method, result.n_pairs, lam, result.fit_seconds, result.diagnostics)
    if not np.isfinite(result.O).all():
        row["status"] = "failed"
        row["error"] = str(result.diagnostics.get("error", "non-finite operator"))
        return row
    _fill_metrics(row, case, result.O, threshold_from_train(train_q))
    status, reason = judgement(row, _applicable(case, method))
    row["status"] = status
    row["error"] = reason
    return row


def _identity(case: SmokeCase, method: str, n_pairs: int, lam, seconds: float, diagnostics: dict) -> dict:
    row = {column: None for column in RUN_COLUMNS}
    row.update({
        "exp": "smoke", "system": "s2", "rho": case.rho, "sigma2_true": case.sigma2,
        "sigma2_used": case.sigma2 if method in SIGMA_METHODS else None,
        "sigma2_source": "true" if method in SIGMA_METHODS else "na",
        "r": case.rank, "n_train": len(case.clean_train_q), "n_pairs": int(n_pairs),
        "basis": "clean", "estimator": method, "lam": lam, "replicate": case.replicate,
        "seed_keys": f"{case.replicate},1|{case.replicate},2", "fit_seconds": float(seconds),
        "n_clipped": diagnostics.get("n_clipped"), "gram_min_eig": diagnostics.get("gram_min_eig"),
        "gram_cond": diagnostics.get("gram_cond"),
    })
    return row


def _fill_metrics(row: dict, case: SmokeCase, operator: np.ndarray, limit: float) -> None:
    reference = case.reference
    train_q = np.vstack(case.clean_train_q)
    test_q = np.vstack(case.clean_test_q)
    test_u = np.vstack(case.clean_test_u)
    row["err_O_rel"] = err_O_rel(operator, reference)
    row["err_A_rel"] = err_A_rel(operator, reference)
    row["err_H_rel"] = err_H_rel(operator, reference)
    row["err_c_abs"] = err_c_abs(operator, reference)
    row["err_c_scaled"] = err_c_scaled(operator, reference, state_rms(train_q))
    row["specrad_hat"], row["specrad_ref"], row["specrad_err"] = specrad_trio(operator, reference)
    row["ep_violation"] = ep_violation(operator, test_q)
    row["ep_violation_ref"] = ep_violation(reference, test_q)
    row["proj_floor"] = proj_floor(case.basis, test_u)
    _fill_rollout(row, case, operator, limit)


def _fill_rollout(row: dict, case: SmokeCase, operator: np.ndarray, limit: float) -> None:
    errors, flags, early = [], [], []
    for reduced, field in zip(case.clean_test_q, case.clean_test_u, strict=True):
        steps = int(reduced.shape[0] - 1)
        predicted, diverged = rollout(operator, reduced[0], steps, limit)
        errors.append(float("inf") if diverged else rollout_err_h(predicted, field, steps, case.basis))
        flags.append(diverged)
        early.append(rollout_err_h(predicted, field, 10, case.basis))
    summary = rollout_aggregate(errors, flags)
    row.update(summary)
    row["rollout_med_h10"] = float(np.median(np.asarray(early, dtype=np.float64)))
    row["rollout_med_h50"] = None


def _applicable(case: SmokeCase, method: str) -> tuple[str, ...]:
    names = list(ALWAYS_METRICS)
    if case.clean_test_q[0].shape[0] >= 51:
        names.append("rollout_med_h50")
    if method == "ridge":
        names.append("lam")
    return tuple(names)


def _is_undefined(row: dict, name: str) -> bool:
    if _kind(row.get(name)) != "undefined":
        return False
    if name == "rollout_mean_finite" and float(row.get("diverged_frac")) == 1.0:
        return False
    return True


def _kind(value) -> str:
    if value is None:
        return "missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "present"
    if np.isnan(number):
        return "undefined"
    return "present"
