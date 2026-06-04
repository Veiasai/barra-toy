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

OBJECTIVE_PRESETS = {
    "Classic mean-risk": {
        "description": "经典均值-方差目标：追求 alpha，同时惩罚组合总风险。",
        "expected_return": 1.0,
        "total_risk": 20.0,
        "active_risk": 0.0,
        "benchmark_deviation": 0.0,
        "concentration": 0.0,
        "factor_exposure": 0.0,
    },
    "Benchmark-aware active": {
        "description": "更关注相对基准的主动风险，适合学习 benchmark 约束下的调仓。",
        "expected_return": 1.0,
        "total_risk": 5.0,
        "active_risk": 25.0,
        "benchmark_deviation": 0.0,
        "concentration": 0.0,
        "factor_exposure": 0.0,
    },
    "Diversified long-only": {
        "description": "额外惩罚集中持仓，让权重更分散。",
        "expected_return": 0.8,
        "total_risk": 15.0,
        "active_risk": 0.0,
        "benchmark_deviation": 0.0,
        "concentration": 1.0,
        "factor_exposure": 0.0,
    },
    "Factor-neutral tilt": {
        "description": "惩罚组合相对基准的因子暴露，减少风格和行业偏离。",
        "expected_return": 1.0,
        "total_risk": 10.0,
        "active_risk": 10.0,
        "benchmark_deviation": 0.0,
        "concentration": 0.0,
        "factor_exposure": 0.5,
    },
    "Stay close to benchmark": {
        "description": "惩罚权重偏离基准，适合观察轻微调仓。",
        "expected_return": 0.8,
        "total_risk": 10.0,
        "active_risk": 10.0,
        "benchmark_deviation": 2.0,
        "concentration": 0.0,
        "factor_exposure": 0.0,
    },
    "Custom coefficients": {
        "description": "从经典目标开始，自行调整每个目标项的系数。",
        "expected_return": 1.0,
        "total_risk": 20.0,
        "active_risk": 0.0,
        "benchmark_deviation": 0.0,
        "concentration": 0.0,
        "factor_exposure": 0.0,
    },
}

st.title("Portfolio Optimizer")
st.caption("这个页面在 alpha、Barra 风险、交易成本和约束之间权衡，求目标组合权重。")

params = scenario_sidebar()
scenario = load_scenario(params)

st.sidebar.header("Objective function")
objective_preset = st.sidebar.selectbox("Objective preset", list(OBJECTIVE_PRESETS))
st.sidebar.caption(OBJECTIVE_PRESETS[objective_preset]["description"])
preset = OBJECTIVE_PRESETS[objective_preset]

expected_return_coef = st.sidebar.slider(
    "Expected return coefficient",
    -2.0,
    5.0,
    preset["expected_return"],
    step=0.1,
)
st.sidebar.caption("alpha 收益项的系数；越大，优化器越愿意买高 alpha 股票。")
total_risk_coef = st.sidebar.slider("Total risk coefficient", 0.0, 100.0, preset["total_risk"], step=1.0)
st.sidebar.caption("总风险惩罚系数；越大，组合整体波动越受压制。")
active_risk_coef = st.sidebar.slider("Active risk coefficient", 0.0, 100.0, preset["active_risk"], step=1.0)
st.sidebar.caption("主动风险惩罚系数；主动风险是组合相对 benchmark 的风险。")
benchmark_deviation_coef = st.sidebar.slider(
    "Benchmark deviation coefficient",
    0.0,
    20.0,
    preset["benchmark_deviation"],
    step=0.5,
)
st.sidebar.caption("权重偏离基准的惩罚系数；越大，optimized weight 越贴近 benchmark weight。")
concentration_coef = st.sidebar.slider("Concentration coefficient", 0.0, 10.0, preset["concentration"], step=0.25)
st.sidebar.caption("集中度惩罚系数；越大，单个股票权重越不容易过大。")
factor_exposure_coef = st.sidebar.slider("Factor exposure coefficient", 0.0, 5.0, preset["factor_exposure"], step=0.1)
st.sidebar.caption("因子主动暴露惩罚系数；越大，风格和行业暴露越贴近基准。")

objective_coefficients = {
    "expected_return": expected_return_coef,
    "total_risk": total_risk_coef,
    "active_risk": active_risk_coef,
    "benchmark_deviation": benchmark_deviation_coef,
    "concentration": concentration_coef,
    "factor_exposure": factor_exposure_coef,
}

st.sidebar.header("Risk model scale")
annualize_risk = st.sidebar.checkbox("Annualize covariance and variance", value=True)
st.sidebar.caption("打开时 F、D、Sigma 是年化口径；关闭时保留日频口径。alpha 应和这里使用同一时间口径。")

st.sidebar.header("Optimizer constraints")
upper_bound = st.sidebar.slider("Single-name upper bound", 0.01, 0.30, 0.08, step=0.01)
turnover_limit = st.sidebar.slider("Turnover limit", 0.05, 2.0, 0.50, step=0.05)
style_limit = st.sidebar.slider("Style active exposure limit", 0.05, 1.50, 0.25, step=0.05)
industry_limit = st.sidebar.slider("Industry active exposure limit", 0.05, 1.00, 0.15, step=0.05)
linear_cost = st.sidebar.slider("Linear cost", 0.0, 0.005, 0.0005, step=0.0001, format="%.4f")
quadratic_cost = st.sidebar.slider("Quadratic cost", 0.0, 0.050, 0.0000, step=0.001, format="%.3f")

reg_w = regression_weights(scenario.universe, scenario.specific_vol, "sqrt_market_cap")
estimated = estimate_factor_returns(scenario.stock_returns, scenario.exposures, reg_w)
residuals = residuals_panel(scenario.stock_returns, scenario.exposures, estimated)
f_cov = factor_covariance(estimated, annualize=annualize_risk)
spec_var = specific_variance(residuals, scenario.specific_vol, annualize=annualize_risk)
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
    risk_aversion=total_risk_coef,
    upper_bound=upper_bound,
    turnover_limit=turnover_limit,
    style_active_limit=style_limit,
    industry_active_limit=industry_limit,
    linear_cost=linear_cost,
    quadratic_cost=quadratic_cost,
    objective_coefficients=objective_coefficients,
)

if not result.success:
    st.warning(f"Solver status: {result.message}")
else:
    st.success(f"Solver status: {result.message}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Objective", f"{result.metrics['objective']:.4%}")
m2.metric("Expected return", f"{result.metrics['expected_return']:.4%}")
risk_scale_label = "annualized" if annualize_risk else "daily"
m3.metric(f"Volatility ({risk_scale_label})", f"{result.metrics['volatility']:.2%}")
m4.metric("Turnover", f"{result.metrics['turnover']:.1%}")
m5.metric("Active share-like", f"{result.metrics['active_share_like']:.1%}")

with st.expander("Objective formula and coefficients", expanded=True):
    st.caption(
        "目标函数是优化器真正最大化的打分公式；系数越大，该目标项对最终权重的影响越大。"
    )
    st.caption(f"当前风险矩阵口径：{risk_scale_label}。alpha 的口径应与风险矩阵保持一致。")
    st.code(
        "maximize c_alpha * alpha'w\n"
        "       - 0.5 * c_total_risk * w'Sigma w\n"
        "       - 0.5 * c_active_risk * (w-b)'Sigma(w-b)\n"
        "       - c_benchmark_deviation * sum((w-b)^2)\n"
        "       - c_concentration * sum(w^2)\n"
        "       - c_factor_exposure * sum(active_factor_exposure^2)\n"
        "       - trading_costs",
        language="text",
    )
    coefficient_frame = pd.DataFrame.from_dict(objective_coefficients, orient="index", columns=["coefficient"])
    st.dataframe(coefficient_frame, width="stretch")

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
    parts = result.objective_terms.set_index("term")["contribution"]
    parts.loc["objective"] = result.metrics["objective"]
    st.plotly_chart(bar(parts, "Objective decomposition"))
    objective_terms = result.objective_terms.copy()
    objective_terms["enabled"] = objective_terms["coefficient"].abs() > 1e-12
    st.dataframe(
        objective_terms.style.format(
            {
                "coefficient": "{:.4f}",
                "raw_value": "{:.5%}",
                "contribution": "{:.5%}",
            }
        ),
        width="stretch",
    )
