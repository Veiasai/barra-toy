from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from barra_lab.data import regression_weights
from barra_lab.optimizer import optimize_portfolio
from barra_lab.plots import bar
from barra_lab.regression import estimate_factor_returns, residuals_panel
from barra_lab.risk import factor_covariance, specific_variance, stock_covariance
from barra_lab.ui import configure_page, load_scenario, scenario_sidebar


configure_page("Optimizer")

st.title("Portfolio Optimizer")
st.caption("这个页面在 alpha、Barra 风险、交易成本和约束之间权衡，求目标组合权重。")

params = scenario_sidebar()
scenario = load_scenario(params)

st.sidebar.header("Optimizer")
risk_aversion = st.sidebar.slider("Risk aversion lambda", 1.0, 100.0, 20.0, step=1.0)
upper_bound = st.sidebar.slider("Single-name upper bound", 0.01, 0.30, 0.08, step=0.01)
turnover_limit = st.sidebar.slider("Turnover limit", 0.05, 2.0, 0.50, step=0.05)
style_limit = st.sidebar.slider("Style active exposure limit", 0.05, 1.50, 0.25, step=0.05)
industry_limit = st.sidebar.slider("Industry active exposure limit", 0.05, 1.00, 0.15, step=0.05)
linear_cost = st.sidebar.slider("Linear cost", 0.0, 0.005, 0.0005, step=0.0001, format="%.4f")
quadratic_cost = st.sidebar.slider("Quadratic cost", 0.0, 0.050, 0.0000, step=0.001, format="%.3f")

reg_w = regression_weights(scenario.universe, scenario.specific_vol, "sqrt_market_cap")
estimated = estimate_factor_returns(scenario.stock_returns, scenario.exposures, reg_w)
residuals = residuals_panel(scenario.stock_returns, scenario.exposures, estimated)
f_cov = factor_covariance(estimated, annualize=True)
spec_var = specific_variance(residuals, scenario.specific_vol, annualize=True)
sigma = stock_covariance(scenario.exposures, f_cov, spec_var)

tickers = scenario.exposures.index
market_cap_weight = scenario.universe.loc[tickers, "market_cap"] / scenario.universe.loc[tickers, "market_cap"].sum()
benchmark = market_cap_weight.rename("benchmark_weight")
current = benchmark.copy()

rng = np.random.default_rng(st.sidebar.number_input("Alpha seed", 0, 9999, 23, step=1))
alpha_scale = st.sidebar.slider("Random alpha scale", 0.000, 0.050, 0.015, step=0.001, format="%.3f")
momentum_tilt = st.sidebar.slider("Momentum alpha tilt", -0.050, 0.050, 0.010, step=0.002, format="%.3f")
value_tilt = st.sidebar.slider("Value alpha tilt", -0.050, 0.050, 0.000, step=0.002, format="%.3f")

alpha = pd.Series(rng.normal(0.0, alpha_scale, len(tickers)), index=tickers, name="alpha")
if "Momentum" in scenario.exposures:
    alpha = alpha + momentum_tilt * scenario.exposures["Momentum"]
if "Value" in scenario.exposures:
    alpha = alpha + value_tilt * scenario.exposures["Value"]
alpha = alpha.rename("alpha")

edit_frame = pd.DataFrame(
    {
        "ticker": tickers,
        "industry": scenario.universe.loc[tickers, "industry"].to_numpy(),
        "alpha": alpha.loc[tickers].to_numpy(dtype=float),
        "benchmark_weight": benchmark.loc[tickers].to_numpy(dtype=float),
        "current_weight": current.loc[tickers].to_numpy(dtype=float),
    }
)

st.caption("可以在求解前直接编辑 alpha、基准权重和当前权重。")
edited = st.data_editor(
    edit_frame,
    hide_index=True,
    width="stretch",
    column_config={
        "alpha": st.column_config.NumberColumn(format="%.5f"),
        "benchmark_weight": st.column_config.NumberColumn(min_value=0.0, format="%.6f"),
        "current_weight": st.column_config.NumberColumn(min_value=0.0, format="%.6f"),
    },
)

alpha = pd.Series(edited["alpha"].to_numpy(dtype=float), index=edited["ticker"], name="alpha")
benchmark = pd.Series(edited["benchmark_weight"].to_numpy(dtype=float), index=edited["ticker"], name="benchmark")
current = pd.Series(edited["current_weight"].to_numpy(dtype=float), index=edited["ticker"], name="current")
if benchmark.sum() <= 0 or current.sum() <= 0:
    st.error("Benchmark and current weights must both have positive sums.")
    st.stop()
benchmark = benchmark / benchmark.sum()
current = current / current.sum()

result = optimize_portfolio(
    alpha=alpha,
    sigma=sigma,
    exposures=scenario.exposures,
    benchmark=benchmark,
    current=current,
    risk_aversion=risk_aversion,
    upper_bound=upper_bound,
    turnover_limit=turnover_limit,
    style_active_limit=style_limit,
    industry_active_limit=industry_limit,
    linear_cost=linear_cost,
    quadratic_cost=quadratic_cost,
)

if not result.success:
    st.warning(f"Solver status: {result.message}")
else:
    st.success(f"Solver status: {result.message}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Objective", f"{result.metrics['objective']:.4%}")
m2.metric("Expected return", f"{result.metrics['expected_return']:.4%}")
m3.metric("Volatility", f"{result.metrics['volatility']:.2%}")
m4.metric("Turnover", f"{result.metrics['turnover']:.1%}")
m5.metric("Active share-like", f"{result.metrics['active_share_like']:.1%}")

weights = result.weights
output = pd.DataFrame(
    {
        "optimized_weight": weights,
        "benchmark_weight": benchmark.reindex(weights.index),
        "active_weight": weights - benchmark.reindex(weights.index),
        "current_weight": current.reindex(weights.index),
        "trade": weights - current.reindex(weights.index),
        "alpha": alpha.reindex(weights.index),
        "industry": scenario.universe.loc[weights.index, "industry"],
    }
).sort_values("optimized_weight", ascending=False)

tab1, tab2, tab3 = st.tabs(["Weights", "Exposures & Constraints", "Objective Parts"])

with tab1:
    st.caption("优化权重是目标持仓；主动权重是目标持仓相对基准组合的偏离。")
    st.dataframe(output.style.format({c: "{:.4%}" for c in output.columns if c != "industry"}), width="stretch")
    st.plotly_chart(bar(output["optimized_weight"].head(40), "Top optimized weights"))

with tab2:
    st.caption("主动暴露展示组合相对基准的风格或行业偏离；约束表展示哪些限制已经接近或达到上限。")
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(bar(result.exposures, "Active factor exposure"))
    with right:
        report = result.constraint_report.copy()
        report["active"] = report["slack"].abs() < 1e-5
        st.dataframe(report.round(6), width="stretch")

with tab3:
    st.caption("目标函数可以拆成预期收益、风险惩罚和交易成本几部分，方便观察优化器为什么这么选权重。")
    parts = pd.Series(
        {
            "expected_return": result.metrics["expected_return"],
            "risk_penalty": -0.5 * risk_aversion * result.metrics["variance"],
            "linear_cost": -result.metrics["linear_cost"],
            "quadratic_cost": -result.metrics["quadratic_cost"],
            "objective": result.metrics["objective"],
        },
        name="value",
    )
    st.plotly_chart(bar(parts, "Objective decomposition"))
    st.dataframe(parts.to_frame().style.format("{:.5%}"), width="stretch")
