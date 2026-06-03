from __future__ import annotations

import pandas as pd
import streamlit as st

from barra_lab.data import regression_weights
from barra_lab.plots import bar, heatmap, line
from barra_lab.regression import estimate_factor_returns, residuals_panel
from barra_lab.risk import covariance_diagnostics, factor_covariance, specific_variance, stock_covariance
from barra_lab.ui import configure_page, load_scenario, scenario_sidebar


configure_page("Barra Learning Lab")

st.title("Barra Learning Lab")

params = scenario_sidebar()
scenario = load_scenario(params)

weights = regression_weights(scenario.universe, scenario.specific_vol, "sqrt_market_cap")
estimated_factor_returns = estimate_factor_returns(
    scenario.stock_returns,
    scenario.exposures,
    weights,
)
residuals = residuals_panel(scenario.stock_returns, scenario.exposures, estimated_factor_returns)
factor_cov = factor_covariance(estimated_factor_returns)
spec_var = specific_variance(residuals, scenario.specific_vol)
sigma = stock_covariance(scenario.exposures, factor_cov, spec_var)
diag = covariance_diagnostics(sigma)

left, mid, right, last = st.columns(4)
left.metric("Stocks", f"{scenario.universe.shape[0]}")
mid.metric("Factors", f"{scenario.exposures.shape[1]}")
right.metric("Trading days", f"{scenario.stock_returns.shape[0]}")
last.metric("Sigma min eigenvalue", f"{diag['min_eigenvalue']:.2e}")

st.caption(
    "这个总览页会跑完整 toy Barra 流水线：原始描述符 -> 因子暴露 -> 截面回归 -> 风险模型 -> 组合优化。"
)

tab1, tab2, tab3, tab4 = st.tabs(["Pipeline Snapshot", "Stock Returns", "Factor Returns", "Risk Matrix"])

with tab1:
    st.caption("这里先展示流水线产出的主要对象，细节可以去左侧各页面逐步查看。")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Raw Universe")
        st.caption("模拟股票池，包含行业、市值、估值、换手率、动量和波动率等原始描述符。")
        st.dataframe(scenario.universe, width="stretch", height=520)
        st.download_button(
            "Download raw universe CSV",
            scenario.universe.to_csv().encode("utf-8"),
            file_name="raw_universe.csv",
            mime="text/csv",
        )
    with c2:
        st.subheader("Exposure Matrix X")
        st.caption("X 记录每只股票对风格和行业风险因子的暴露，也是截面回归里的设计矩阵。")
        st.dataframe(scenario.exposures.round(3), width="stretch", height=520)
        st.download_button(
            "Download exposure matrix CSV",
            scenario.exposures.to_csv().encode("utf-8"),
            file_name="exposure_matrix.csv",
            mime="text/csv",
        )

    st.subheader("Regression Weight Distribution")
    st.caption("这是加权最小二乘回归里的样本权重，用来估计因子收益；它不是组合持仓权重。")
    st.plotly_chart(bar(weights.sort_values(ascending=False).head(30), "Top regression weights"))

with tab2:
    st.caption("这里展示个股历史日收益和累计收益，用来观察输入到截面回归的股票收益序列。")
    default_stocks = scenario.universe["market_cap"].sort_values(ascending=False).head(5).index.tolist()
    selected_stocks = st.multiselect(
        "Stocks to plot",
        scenario.stock_returns.columns.tolist(),
        default=default_stocks,
    )
    if selected_stocks:
        daily = scenario.stock_returns[selected_stocks]
        cumulative = (1.0 + daily).cumprod() - 1.0
        st.plotly_chart(line(daily, "Historical daily stock returns"))
        st.plotly_chart(line(cumulative, "Cumulative stock returns"))
    else:
        st.info("Select at least one stock to show the historical return chart.")

with tab3:
    st.caption("因子收益表示截面回归后，被归因到公共风险因子上的每日收益。")
    st.plotly_chart(line(estimated_factor_returns, "Estimated daily factor returns"))
    corr = pd.Series(
        {
            factor: scenario.true_factor_returns[factor].corr(estimated_factor_returns[factor])
            for factor in scenario.true_factor_returns.columns
        },
        name="true_vs_estimated_corr",
    )
    st.dataframe(corr.to_frame().round(3), width="stretch")

with tab4:
    st.caption("风险模型把因子协方差 F 和特质风险 D 合成为股票协方差矩阵 Sigma。")
    st.plotly_chart(heatmap(factor_cov.round(8), "Factor covariance F"))
    top = scenario.universe["market_cap"].sort_values(ascending=False).head(40).index
    st.plotly_chart(heatmap(sigma.loc[top, top], "Stock covariance Sigma, top 40 by market cap"))
