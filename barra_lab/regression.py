from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_factor_returns(
    stock_returns: pd.DataFrame,
    exposures: pd.DataFrame,
    weights: pd.Series,
) -> pd.DataFrame:
    x = exposures.to_numpy(dtype=float)
    w = weights.loc[exposures.index].to_numpy(dtype=float)
    xtw = x.T * w
    xtwx_inv_xtw = np.linalg.pinv(xtw @ x) @ xtw

    estimated = {}
    for dt, row in stock_returns.iterrows():
        estimated[dt] = xtwx_inv_xtw @ row.loc[exposures.index].to_numpy(dtype=float)

    return pd.DataFrame.from_dict(estimated, orient="index", columns=exposures.columns)


def regression_steps(
    stock_return: pd.Series,
    exposures: pd.DataFrame,
    weights: pd.Series,
) -> dict[str, object]:
    x = exposures.to_numpy(dtype=float)
    r = stock_return.loc[exposures.index].to_numpy(dtype=float)
    w = weights.loc[exposures.index].to_numpy(dtype=float)

    xtw = x.T * w
    xtwx = xtw @ x
    xtwr = xtw @ r
    xtwx_pinv = np.linalg.pinv(xtwx)
    f_hat = xtwx_pinv @ xtwr
    fitted = x @ f_hat
    residual = r - fitted
    eigvals = np.linalg.eigvalsh(xtwx)
    rank = int(np.linalg.matrix_rank(xtwx))
    cond = float(np.linalg.cond(xtwx))

    factor_returns = pd.Series(f_hat, index=exposures.columns, name="factor_return")
    attribution = pd.DataFrame(
        {
            "actual_return": r,
            "factor_return_part": fitted,
            "specific_return_residual": residual,
        },
        index=exposures.index,
    )
    matrices = {
        "X": exposures.copy(),
        "W_diag": pd.Series(w, index=exposures.index, name="W_diag"),
        "X_transpose_W_X": pd.DataFrame(xtwx, index=exposures.columns, columns=exposures.columns),
        "X_transpose_W_r": pd.Series(xtwr, index=exposures.columns, name="X_transpose_W_r"),
        "pseudo_inverse": pd.DataFrame(xtwx_pinv, index=exposures.columns, columns=exposures.columns),
    }
    diagnostics = {
        "rank": rank,
        "n_factors": exposures.shape[1],
        "condition_number": cond,
        "min_eigenvalue": float(eigvals.min()),
        "max_eigenvalue": float(eigvals.max()),
        "weighted_sse": float(np.sum(w * residual * residual)),
        "residual_std": float(np.std(residual, ddof=0)),
    }

    return {
        "factor_returns": factor_returns,
        "attribution": attribution,
        "matrices": matrices,
        "eigenvalues": pd.Series(eigvals, name="eigenvalue"),
        "diagnostics": diagnostics,
    }


def residuals_panel(
    stock_returns: pd.DataFrame,
    exposures: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    fitted = factor_returns @ exposures.T
    fitted = fitted.loc[stock_returns.index, stock_returns.columns]
    return stock_returns - fitted

