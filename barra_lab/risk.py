from __future__ import annotations

import numpy as np
import pandas as pd


def factor_covariance(factor_returns: pd.DataFrame, annualize: bool = False) -> pd.DataFrame:
    cov = factor_returns.cov()
    return cov * 252.0 if annualize else cov


def specific_variance(
    residuals: pd.DataFrame | None,
    specific_vol: pd.Series,
    annualize: bool = False,
) -> pd.Series:
    if residuals is None or residuals.empty:
        var = specific_vol.pow(2)
    else:
        var = residuals.var(axis=0, ddof=1).reindex(specific_vol.index).fillna(specific_vol.pow(2))
    return var * 252.0 if annualize else var


def stock_covariance(
    exposures: pd.DataFrame,
    factor_cov: pd.DataFrame,
    specific_var: pd.Series,
) -> pd.DataFrame:
    factors = list(exposures.columns)
    x = exposures[factors].to_numpy(dtype=float)
    f = factor_cov.loc[factors, factors].to_numpy(dtype=float)
    d = np.diag(specific_var.loc[exposures.index].to_numpy(dtype=float))
    sigma = x @ f @ x.T + d
    return pd.DataFrame(sigma, index=exposures.index, columns=exposures.index)


def covariance_diagnostics(sigma: pd.DataFrame) -> dict[str, object]:
    eigvals = np.linalg.eigvalsh(sigma.to_numpy(dtype=float))
    return {
        "rank": int(np.linalg.matrix_rank(sigma.to_numpy(dtype=float))),
        "min_eigenvalue": float(eigvals.min()),
        "max_eigenvalue": float(eigvals.max()),
        "condition_number": float(np.linalg.cond(sigma.to_numpy(dtype=float))),
        "is_positive_semidefinite": bool(eigvals.min() >= -1e-10),
        "eigenvalues": pd.Series(eigvals, name="eigenvalue"),
    }


def portfolio_risk_breakdown(
    weights: pd.Series,
    exposures: pd.DataFrame,
    factor_cov: pd.DataFrame,
    specific_var: pd.Series,
) -> dict[str, object]:
    tickers = exposures.index
    factors = exposures.columns
    w = weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    x = exposures.to_numpy(dtype=float)
    f = factor_cov.loc[factors, factors].to_numpy(dtype=float)
    d = specific_var.loc[tickers].to_numpy(dtype=float)

    factor_exposure = x.T @ w
    factor_var = float(factor_exposure.T @ f @ factor_exposure)
    specific_var_total = float(np.sum(w * w * d))
    total_var = factor_var + specific_var_total

    factor_component = factor_exposure * (f @ factor_exposure)
    factor_contrib = pd.Series(factor_component, index=factors, name="variance_contribution")
    stock_specific_contrib = pd.Series(w * w * d, index=tickers, name="specific_variance_contribution")

    return {
        "total_variance": total_var,
        "factor_variance": factor_var,
        "specific_variance": specific_var_total,
        "volatility": float(np.sqrt(max(total_var, 0.0))),
        "factor_exposure": pd.Series(factor_exposure, index=factors, name="portfolio_factor_exposure"),
        "factor_contribution": factor_contrib,
        "specific_contribution": stock_specific_contrib,
    }

