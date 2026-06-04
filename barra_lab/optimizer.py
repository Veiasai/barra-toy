from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize

from barra_lab.data import STYLE_FACTORS


@dataclass(frozen=True)
class OptimizerResult:
    weights: pd.Series
    success: bool
    message: str
    metrics: dict[str, float]
    exposures: pd.Series
    constraint_report: pd.DataFrame
    objective_terms: pd.DataFrame


OBJECTIVE_COEFFICIENTS = {
    "expected_return": 1.0,
    "total_risk": 20.0,
    "active_risk": 0.0,
    "benchmark_deviation": 0.0,
    "concentration": 0.0,
    "factor_exposure": 0.0,
}


def default_objective_coefficients(risk_aversion: float = 20.0) -> dict[str, float]:
    coefficients = OBJECTIVE_COEFFICIENTS.copy()
    coefficients["total_risk"] = float(risk_aversion)
    return coefficients


def _clean_objective_coefficients(
    coefficients: dict[str, float] | None,
    risk_aversion: float,
) -> dict[str, float]:
    clean = default_objective_coefficients(risk_aversion)
    if coefficients:
        for key, value in coefficients.items():
            if key not in clean:
                continue
            clean[key] = float(value)

    for key, value in clean.items():
        if key == "expected_return":
            continue
        if value < 0:
            raise ValueError(f"{key} coefficient must be non-negative.")
    return clean


def _normal_start(n: int, upper_bound: float) -> np.ndarray:
    x0 = np.full(n, 1.0 / n)
    if upper_bound * n < 1.0:
        return x0
    return np.minimum(x0, upper_bound) / np.minimum(x0, upper_bound).sum()


def _cap_and_redistribute(weights: np.ndarray, upper_bound: float) -> np.ndarray:
    """Project a long-only weight vector onto sum=1 and max-weight bounds."""
    w = np.maximum(weights.astype(float), 0.0)
    if w.sum() <= 0:
        return _normal_start(len(w), upper_bound)
    w = w / w.sum()

    capped = np.minimum(w, upper_bound)
    remaining = 1.0 - capped.sum()
    while remaining > 1e-12:
        room = upper_bound - capped
        eligible = room > 1e-12
        if not np.any(eligible):
            break
        add = remaining * room / room[eligible].sum()
        add[~eligible] = 0.0
        step = np.minimum(add, room)
        capped += step
        new_remaining = 1.0 - capped.sum()
        if abs(new_remaining - remaining) < 1e-14:
            break
        remaining = new_remaining
    return capped / capped.sum()


def optimize_portfolio(
    alpha: pd.Series,
    sigma: pd.DataFrame,
    exposures: pd.DataFrame,
    benchmark: pd.Series | None = None,
    current: pd.Series | None = None,
    risk_aversion: float = 20.0,
    upper_bound: float = 0.08,
    turnover_limit: float = 0.50,
    style_active_limit: float | None = 0.25,
    industry_active_limit: float | None = 0.15,
    linear_cost: float = 0.0005,
    quadratic_cost: float = 0.0,
    objective_coefficients: dict[str, float] | None = None,
) -> OptimizerResult:
    tickers = list(sigma.index)
    n = len(tickers)
    alpha_v = alpha.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    sigma_m = sigma.loc[tickers, tickers].to_numpy(dtype=float)
    x_m = exposures.loc[tickers].to_numpy(dtype=float)
    factor_names = list(exposures.columns)

    if benchmark is None:
        benchmark_v = np.full(n, 1.0 / n)
    else:
        benchmark_v = benchmark.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
        benchmark_v = benchmark_v / benchmark_v.sum()

    if current is None:
        current_v = benchmark_v.copy()
    else:
        current_v = current.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
        current_v = current_v / current_v.sum()

    coefficients = _clean_objective_coefficients(objective_coefficients, risk_aversion)

    def unpack(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return z[:n], z[n:]

    def objective(z: np.ndarray) -> float:
        w, turnover_aux = unpack(z)
        trade = w - current_v
        active = w - benchmark_v
        active_exposure = x_m.T @ active
        return float(
            -coefficients["expected_return"] * float(alpha_v @ w)
            + 0.5 * coefficients["total_risk"] * float(w @ sigma_m @ w)
            + 0.5 * coefficients["active_risk"] * float(active @ sigma_m @ active)
            + coefficients["benchmark_deviation"] * float(active @ active)
            + coefficients["concentration"] * float(w @ w)
            + coefficients["factor_exposure"] * float(active_exposure @ active_exposure)
            + linear_cost * np.sum(turnover_aux)
            + quadratic_cost * float(trade @ trade)
        )

    def objective_grad(z: np.ndarray) -> np.ndarray:
        w, _ = unpack(z)
        trade = w - current_v
        active = w - benchmark_v
        active_exposure = x_m.T @ active
        grad_w = (
            -coefficients["expected_return"] * alpha_v
            + coefficients["total_risk"] * (sigma_m @ w)
            + coefficients["active_risk"] * (sigma_m @ active)
            + 2.0 * coefficients["benchmark_deviation"] * active
            + 2.0 * coefficients["concentration"] * w
            + 2.0 * coefficients["factor_exposure"] * (x_m @ active_exposure)
            + 2.0 * quadratic_cost * trade
        )
        grad_u = np.full(n, linear_cost)
        return np.concatenate([grad_w, grad_u])

    constraints = []
    sum_row = np.zeros((1, 2 * n))
    sum_row[0, :n] = 1.0
    constraints.append(LinearConstraint(sum_row, 1.0, 1.0))

    turnover_row = np.zeros((1, 2 * n))
    turnover_row[0, n:] = 1.0
    constraints.append(LinearConstraint(turnover_row, 0.0, turnover_limit))

    abs_rows = np.zeros((2 * n, 2 * n))
    abs_lower = np.empty(2 * n)
    abs_upper = np.full(2 * n, np.inf)
    for i in range(n):
        abs_rows[2 * i, i] = -1.0
        abs_rows[2 * i, n + i] = 1.0
        abs_lower[2 * i] = -current_v[i]

        abs_rows[2 * i + 1, i] = 1.0
        abs_rows[2 * i + 1, n + i] = 1.0
        abs_lower[2 * i + 1] = current_v[i]
    constraints.append(LinearConstraint(abs_rows, abs_lower, abs_upper))

    style_idx = [i for i, f in enumerate(factor_names) if f in STYLE_FACTORS]
    industry_idx = [i for i, f in enumerate(factor_names) if f not in STYLE_FACTORS]

    def add_active_exposure_constraints(indices: list[int], limit: float | None) -> None:
        if limit is None:
            return
        rows = np.zeros((len(indices), 2 * n))
        lower = np.empty(len(indices))
        upper = np.empty(len(indices))
        for idx in indices:
            row_idx = indices.index(idx)
            rows[row_idx, :n] = x_m[:, idx]
            benchmark_exposure = float(x_m[:, idx] @ benchmark_v)
            lower[row_idx] = benchmark_exposure - limit
            upper[row_idx] = benchmark_exposure + limit
        constraints.append(LinearConstraint(rows, lower, upper))

    add_active_exposure_constraints(style_idx, style_active_limit)
    add_active_exposure_constraints(industry_idx, industry_active_limit)

    w0 = _cap_and_redistribute(current_v, upper_bound)
    u0 = np.abs(w0 - current_v)
    if turnover_limit is not None and u0.sum() > turnover_limit:
        u0 = u0 * (turnover_limit / u0.sum())
    z0 = np.concatenate([w0, u0])

    result = minimize(
        objective,
        z0,
        method="SLSQP",
        jac=objective_grad,
        bounds=Bounds(np.zeros(2 * n), np.concatenate([np.full(n, upper_bound), np.full(n, 2.0)])),
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    w = result.x[:n]
    weights = pd.Series(w, index=tickers, name="optimized_weight")
    active = w - benchmark_v
    trade = w - current_v
    risk = float(w @ sigma_m @ w)
    active_risk = float(active @ sigma_m @ active)
    benchmark_deviation = float(active @ active)
    concentration = float(w @ w)
    expected_return = float(alpha_v @ w)
    factor_exposure_penalty = float((x_m.T @ active) @ (x_m.T @ active))
    linear_cost_value = float(linear_cost * np.sum(np.abs(trade)))
    quadratic_cost_value = float(quadratic_cost * trade @ trade)
    objective_rows = [
        {
            "term": "expected_return",
            "description": "alpha 带来的组合预期收益，贡献为 coefficient * alpha'w。",
            "coefficient": coefficients["expected_return"],
            "raw_value": expected_return,
            "contribution": coefficients["expected_return"] * expected_return,
        },
        {
            "term": "total_risk_penalty",
            "description": "组合自身的总风险惩罚，raw value 是 w'Sigma w。",
            "coefficient": coefficients["total_risk"],
            "raw_value": risk,
            "contribution": -0.5 * coefficients["total_risk"] * risk,
        },
        {
            "term": "active_risk_penalty",
            "description": "相对 benchmark 的主动风险惩罚，raw value 是 (w-b)'Sigma(w-b)。",
            "coefficient": coefficients["active_risk"],
            "raw_value": active_risk,
            "contribution": -0.5 * coefficients["active_risk"] * active_risk,
        },
        {
            "term": "benchmark_deviation_penalty",
            "description": "权重偏离 benchmark 的惩罚，raw value 是 sum((w-b)^2)。",
            "coefficient": coefficients["benchmark_deviation"],
            "raw_value": benchmark_deviation,
            "contribution": -coefficients["benchmark_deviation"] * benchmark_deviation,
        },
        {
            "term": "concentration_penalty",
            "description": "持仓集中度惩罚，raw value 是 sum(w^2)。",
            "coefficient": coefficients["concentration"],
            "raw_value": concentration,
            "contribution": -coefficients["concentration"] * concentration,
        },
        {
            "term": "factor_exposure_penalty",
            "description": "因子主动暴露惩罚，raw value 是 sum(active factor exposure^2)。",
            "coefficient": coefficients["factor_exposure"],
            "raw_value": factor_exposure_penalty,
            "contribution": -coefficients["factor_exposure"] * factor_exposure_penalty,
        },
        {
            "term": "linear_trade_cost",
            "description": "线性交易成本，raw value 是 sum(abs(w-current))。",
            "coefficient": linear_cost,
            "raw_value": float(np.sum(np.abs(trade))),
            "contribution": -linear_cost_value,
        },
        {
            "term": "quadratic_trade_cost",
            "description": "二次交易成本，raw value 是 sum((w-current)^2)。",
            "coefficient": quadratic_cost,
            "raw_value": float(trade @ trade),
            "contribution": -quadratic_cost_value,
        },
    ]
    objective_terms = pd.DataFrame(objective_rows)
    objective_value = float(objective_terms["contribution"].sum())

    active_exposure = pd.Series(x_m.T @ active, index=factor_names, name="active_exposure")
    rows = [
        {
            "constraint": "sum(weights) = 1",
            "value": weights.sum(),
            "lower": 1.0,
            "upper": 1.0,
            "slack": abs(weights.sum() - 1.0),
        },
        {
            "constraint": "turnover",
            "value": float(np.sum(np.abs(trade))),
            "lower": 0.0,
            "upper": turnover_limit,
            "slack": turnover_limit - float(np.sum(np.abs(trade))),
        },
        {
            "constraint": "max single-name weight",
            "value": float(weights.max()),
            "lower": 0.0,
            "upper": upper_bound,
            "slack": upper_bound - float(weights.max()),
        },
    ]
    for factor, value in active_exposure.items():
        limit = style_active_limit if factor in STYLE_FACTORS else industry_active_limit
        if limit is None:
            continue
        rows.append(
            {
                "constraint": f"active {factor}",
                "value": float(value),
                "lower": -limit,
                "upper": limit,
                "slack": limit - abs(float(value)),
            }
        )

    metrics = {
        "expected_return": expected_return,
        "variance": risk,
        "volatility": float(np.sqrt(max(risk, 0.0))),
        "active_variance": active_risk,
        "active_volatility": float(np.sqrt(max(active_risk, 0.0))),
        "benchmark_deviation": benchmark_deviation,
        "concentration": concentration,
        "factor_exposure_penalty": factor_exposure_penalty,
        "linear_cost": linear_cost_value,
        "quadratic_cost": quadratic_cost_value,
        "objective": objective_value,
        "turnover": float(np.sum(np.abs(trade))),
        "active_share_like": float(0.5 * np.sum(np.abs(active))),
    }

    return OptimizerResult(
        weights=weights,
        success=bool(result.success),
        message=str(result.message),
        metrics=metrics,
        exposures=active_exposure,
        constraint_report=pd.DataFrame(rows),
        objective_terms=objective_terms,
    )
