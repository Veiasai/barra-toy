from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


STYLE_FACTORS = ["Size", "Value", "Momentum", "Liquidity", "Volatility"]
INDUSTRIES = ["Bank", "Tech", "Energy", "Consumer"]


@dataclass(frozen=True)
class Scenario:
    universe: pd.DataFrame
    exposures: pd.DataFrame
    exposure_steps: pd.DataFrame
    specific_vol: pd.Series
    stock_returns: pd.DataFrame
    true_factor_returns: pd.DataFrame


def winsorize(s: pd.Series, z: float = 3.0) -> pd.Series:
    """Clip a cross-section at mean +/- z standard deviations."""
    mu = s.mean()
    sigma = s.std(ddof=0)
    if sigma == 0:
        return s.copy()
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

    raw = pd.DataFrame(
        {
            "ticker": tickers,
            "industry": rng.choice(INDUSTRIES, n_stocks),
            "market_cap": np.exp(rng.normal(22, 1.0, n_stocks)),
            "book_to_price": rng.lognormal(mean=-1.0, sigma=0.35, size=n_stocks),
            "daily_turnover": rng.lognormal(mean=-3.5, sigma=0.45, size=n_stocks),
            "momentum_12m": rng.normal(0.08, 0.22, n_stocks),
            "volatility_60d": rng.lognormal(mean=-2.5, sigma=0.25, size=n_stocks),
        }
    ).set_index("ticker")

    return raw


def build_exposures(
    universe: pd.DataFrame,
    winsor_z: float = 3.0,
    standardize_styles: bool = True,
    include_industries: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build factor exposures and a long-form table of descriptor steps."""
    x = pd.DataFrame(index=universe.index)
    steps = []

    raw_styles = {
        "Size": np.log(universe["market_cap"]),
        "Value": universe["book_to_price"],
        "Momentum": universe["momentum_12m"],
        "Liquidity": universe["daily_turnover"],
        "Volatility": universe["volatility_60d"],
    }

    for factor, values in raw_styles.items():
        raw = pd.Series(values, index=universe.index, name="raw")
        clipped = winsorize(raw, winsor_z)
        exposure = standardize(clipped) if standardize_styles else clipped
        x[factor] = exposure

        factor_steps = pd.DataFrame(
            {
                "ticker": universe.index,
                "factor": factor,
                "raw": raw.to_numpy(dtype=float),
                "winsorized": clipped.to_numpy(dtype=float),
                "exposure": exposure.to_numpy(dtype=float),
            }
        )
        steps.append(factor_steps)

    if include_industries:
        industry_x = pd.get_dummies(universe["industry"], dtype=float)
        x = pd.concat([x, industry_x], axis=1)

    return x, pd.concat(steps, ignore_index=True)


def make_specific_vol(universe: pd.DataFrame, scale: float = 1.0) -> pd.Series:
    base = 0.010 + 0.020 / np.sqrt(universe["market_cap"] / universe["market_cap"].median())
    return (base * scale).rename("specific_vol")


def simulate_daily_returns(
    exposures: pd.DataFrame,
    dates: pd.DatetimeIndex,
    specific_vol: pd.Series,
    seed: int = 11,
    style_mean: float = 0.0002,
    industry_mean: float = 0.0001,
    style_vol: float = 0.006,
    industry_vol: float = 0.004,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate stock returns from true factor returns plus specific returns."""
    rng = np.random.default_rng(seed)

    style_factors = [c for c in STYLE_FACTORS if c in exposures.columns]
    industry_factors = [c for c in exposures.columns if c not in style_factors]
    factors = style_factors + industry_factors

    true_factor_returns = pd.DataFrame(index=dates, columns=factors, dtype=float)
    if style_factors:
        true_factor_returns[style_factors] = rng.normal(
            style_mean,
            style_vol,
            (len(dates), len(style_factors)),
        )
    if industry_factors:
        true_factor_returns[industry_factors] = rng.normal(
            industry_mean,
            industry_vol,
            (len(dates), len(industry_factors)),
        )

    stock_returns = pd.DataFrame(index=dates, columns=exposures.index, dtype=float)
    x = exposures[factors].to_numpy(dtype=float)

    for dt in dates:
        factor_part = x @ true_factor_returns.loc[dt, factors].to_numpy(dtype=float)
        specific = rng.normal(0.0, specific_vol.loc[exposures.index].to_numpy(dtype=float))
        stock_returns.loc[dt] = factor_part + specific

    return stock_returns, true_factor_returns


def make_scenario(
    n_stocks: int = 80,
    n_days: int = 60,
    universe_seed: int = 7,
    return_seed: int = 11,
    winsor_z: float = 3.0,
    standardize_styles: bool = True,
    specific_scale: float = 1.0,
    style_vol: float = 0.006,
    industry_vol: float = 0.004,
) -> Scenario:
    universe = make_universe(n_stocks=n_stocks, seed=universe_seed)
    exposures, exposure_steps = build_exposures(
        universe,
        winsor_z=winsor_z,
        standardize_styles=standardize_styles,
    )
    specific_vol = make_specific_vol(universe, scale=specific_scale)
    dates = pd.bdate_range("2026-01-02", periods=n_days)
    stock_returns, true_factor_returns = simulate_daily_returns(
        exposures,
        dates,
        specific_vol,
        seed=return_seed,
        style_vol=style_vol,
        industry_vol=industry_vol,
    )
    return Scenario(
        universe=universe,
        exposures=exposures,
        exposure_steps=exposure_steps,
        specific_vol=specific_vol,
        stock_returns=stock_returns,
        true_factor_returns=true_factor_returns,
    )


def regression_weights(
    universe: pd.DataFrame,
    specific_vol: pd.Series,
    method: str = "sqrt_market_cap",
) -> pd.Series:
    if method == "equal":
        raw = pd.Series(1.0, index=universe.index)
    elif method == "market_cap":
        raw = universe["market_cap"]
    elif method == "inverse_specific_var":
        raw = 1.0 / np.square(specific_vol)
    else:
        raw = np.sqrt(universe["market_cap"])
    return (raw / raw.sum()).rename("weight")

