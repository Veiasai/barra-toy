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

    def unpack(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return z[:n], z[n:]

    def objective(z: np.ndarray) -> float:
        w, turnover_aux = unpack(z)
        trade = w - current_v
        return float(
            -alpha_v @ w
            + 0.5 * risk_aversion * float(w @ sigma_m @ w)
            + linear_cost * np.sum(turnover_aux)
            + quadratic_cost * float(trade @ trade)
        )

    def objective_grad(z: np.ndarray) -> np.ndarray:
        w, _ = unpack(z)
        trade = w - current_v
        grad_w = -alpha_v + risk_aversion * (sigma_m @ w) + 2.0 * quadratic_cost * trade
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
    expected_return = float(alpha_v @ w)
    linear_cost_value = float(linear_cost * np.sum(np.abs(trade)))
    quadratic_cost_value = float(quadratic_cost * trade @ trade)
    objective_value = expected_return - 0.5 * risk_aversion * risk - linear_cost_value - quadratic_cost_value

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
    )
