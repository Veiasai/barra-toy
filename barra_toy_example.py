"""
Toy Barra-style multi-factor model example.

This script generates a small synthetic equity universe, builds style and
industry factor exposures, then estimates daily factor returns with
cross-sectional weighted least squares.

Run:
    python barra_toy_example.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, z: float = 3.0) -> pd.Series:
    """Clip a cross-section at mean +/- z standard deviations."""
    mu = s.mean()
    sigma = s.std(ddof=0)
    return s.clip(mu - z * sigma, mu + z * sigma)


def standardize(s: pd.Series) -> pd.Series:
    """Convert a cross-section to z-scores."""
    sigma = s.std(ddof=0)
    if sigma == 0:
        return s * 0.0
    return (s - s.mean()) / sigma


def make_universe(n_stocks: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"S{i:03d}" for i in range(n_stocks)]
    industries = np.array(["Bank", "Tech", "Energy", "Consumer"])

    raw = pd.DataFrame(
        {
            "ticker": tickers,
            "industry": rng.choice(industries, n_stocks),
            "market_cap": np.exp(rng.normal(22, 1.0, n_stocks)),
            "book_to_price": rng.lognormal(mean=-1.0, sigma=0.35, size=n_stocks),
            "daily_turnover": rng.lognormal(mean=-3.5, sigma=0.45, size=n_stocks),
            "momentum_12m": rng.normal(0.08, 0.22, n_stocks),
            "volatility_60d": rng.lognormal(mean=-2.5, sigma=0.25, size=n_stocks),
        }
    ).set_index("ticker")

    return raw


def build_exposures(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Build factor exposures X.

    Barra production models use many carefully engineered descriptors. This toy
    version uses five style factors and four industry dummy factors:
      - Size: log market cap
      - Value: book-to-price
      - Momentum: trailing 12 month return
      - Liquidity: daily turnover
      - Volatility: 60 day realized volatility
      - Bank/Tech/Energy/Consumer: one-hot industry exposures
    """
    x = pd.DataFrame(index=universe.index)

    raw_styles = {
        "Size": np.log(universe["market_cap"]),
        "Value": universe["book_to_price"],
        "Momentum": universe["momentum_12m"],
        "Liquidity": universe["daily_turnover"],
        "Volatility": universe["volatility_60d"],
    }

    for name, values in raw_styles.items():
        x[name] = standardize(winsorize(pd.Series(values, index=universe.index)))

    industry_x = pd.get_dummies(universe["industry"], dtype=float)
    x = pd.concat([x, industry_x], axis=1)

    return x


def simulate_daily_returns(
    exposures: pd.DataFrame,
    dates: pd.DatetimeIndex,
    specific_vol: pd.Series,
    seed: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate stock returns from true factor returns plus specific returns.

    r_it = sum_k X_ik * f_kt + epsilon_it
    """
    rng = np.random.default_rng(seed)

    style_factors = ["Size", "Value", "Momentum", "Liquidity", "Volatility"]
    industry_factors = [c for c in exposures.columns if c not in style_factors]
    factors = style_factors + industry_factors

    true_factor_returns = pd.DataFrame(index=dates, columns=factors, dtype=float)
    true_factor_returns[style_factors] = rng.normal(0.0002, 0.006, (len(dates), len(style_factors)))
    true_factor_returns[industry_factors] = rng.normal(0.0001, 0.004, (len(dates), len(industry_factors)))

    stock_returns = pd.DataFrame(index=dates, columns=exposures.index, dtype=float)
    x = exposures[factors].to_numpy()

    for dt in dates:
        factor_part = x @ true_factor_returns.loc[dt, factors].to_numpy()
        specific = rng.normal(0.0, specific_vol.to_numpy())
        stock_returns.loc[dt] = factor_part + specific

    return stock_returns, true_factor_returns


def estimate_factor_returns(
    stock_returns: pd.DataFrame,
    exposures: pd.DataFrame,
    weights: pd.Series,
) -> pd.DataFrame:
    """
    Estimate daily factor returns with weighted least squares:

        f_hat_t = (X' W X)^(-1) X' W r_t

    In real Barra-style risk models, W is often linked to issuer size or
    inverse specific variance. Here we use sqrt market-cap normalized weights.
    """
    x = exposures.to_numpy(dtype=float)
    w = np.diag(weights.loc[exposures.index].to_numpy(dtype=float))
    xtwx_inv_xtw = np.linalg.pinv(x.T @ w @ x) @ x.T @ w

    estimated = {}
    for dt, row in stock_returns.iterrows():
        estimated[dt] = xtwx_inv_xtw @ row.loc[exposures.index].to_numpy(dtype=float)

    return pd.DataFrame.from_dict(estimated, orient="index", columns=exposures.columns)


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    universe = make_universe()
    exposures = build_exposures(universe)

    dates = pd.bdate_range("2026-01-02", periods=30)
    specific_vol = 0.010 + 0.020 / np.sqrt(universe["market_cap"] / universe["market_cap"].median())

    stock_returns, true_factor_returns = simulate_daily_returns(exposures, dates, specific_vol)

    w = np.sqrt(universe["market_cap"])
    w = w / w.sum()
    estimated_factor_returns = estimate_factor_returns(stock_returns, exposures, w)

    print("\n=== 原始股票特征样例 ===")
    print(universe.head(8))

    print("\n=== Barra 风格/行业因子暴露 X 样例 ===")
    print(exposures.head(8).round(3))

    print("\n=== 个股日收益 r_it 样例 ===")
    print(stock_returns.iloc[:5, :8].round(4))

    print("\n=== 估计出来的每日因子收益 f_hat_t 样例 ===")
    print(estimated_factor_returns.head(8).round(5))

    print("\n=== 真实因子收益 vs 估计因子收益：相关系数 ===")
    corr = pd.Series(
        {
            factor: true_factor_returns[factor].corr(estimated_factor_returns[factor])
            for factor in true_factor_returns.columns
        }
    )
    print(corr.round(3))

    sample_day = estimated_factor_returns.index[0]
    reconstructed = exposures @ estimated_factor_returns.loc[sample_day]
    residual = stock_returns.loc[sample_day] - reconstructed

    print(f"\n=== {sample_day.date()} 收益分解样例 ===")
    attribution = pd.DataFrame(
        {
            "actual_return": stock_returns.loc[sample_day],
            "factor_return_part": reconstructed,
            "specific_return_residual": residual,
        }
    )
    print(attribution.head(10).round(5))


if __name__ == "__main__":
    main()
