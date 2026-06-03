from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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

    def objective(w: np.ndarray) -> float:
        trade = w - current_v
        utility = (
            alpha_v @ w
            - 0.5 * risk_aversion * float(w @ sigma_m @ w)
            - linear_cost * np.sum(np.abs(trade))
            - quadratic_cost * float(trade @ trade)
        )
        return -utility

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if turnover_limit is not None:
        constraints.append({"type": "ineq", "fun": lambda w: turnover_limit - np.sum(np.abs(w - current_v))})

    style_idx = [i for i, f in enumerate(factor_names) if f in STYLE_FACTORS]
    industry_idx = [i for i, f in enumerate(factor_names) if f not in STYLE_FACTORS]

    def add_active_exposure_constraints(indices: list[int], limit: float | None) -> None:
        if limit is None:
            return
        for idx in indices:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, j=idx: limit - float(x_m[:, j] @ (w - benchmark_v)),
                }
            )
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, j=idx: limit + float(x_m[:, j] @ (w - benchmark_v)),
                }
            )

    add_active_exposure_constraints(style_idx, style_active_limit)
    add_active_exposure_constraints(industry_idx, industry_active_limit)

    result = minimize(
        objective,
        _normal_start(n, upper_bound),
        method="SLSQP",
        bounds=[(0.0, upper_bound)] * n,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    w = result.x
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

