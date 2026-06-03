from __future__ import annotations

import pandas as pd
import streamlit as st

from barra_lab.data import regression_weights
from barra_lab.plots import bar, heatmap
from barra_lab.regression import estimate_factor_returns, residuals_panel
from barra_lab.risk import (
    covariance_diagnostics,
    factor_covariance,
    portfolio_risk_breakdown,
    specific_variance,
    stock_covariance,
)
from barra_lab.ui import configure_page, load_scenario, scenario_sidebar


configure_page("Risk Model")

st.title("Risk Model")
st.caption("这个页面构造 Sigma = X F X' + D，并展示它如何决定个股风险和组合风险。")

params = scenario_sidebar()
scenario = load_scenario(params)

weight_method = st.selectbox("Regression weight", ["sqrt_market_cap", "market_cap", "inverse_specific_var", "equal"])
annualize = st.checkbox("Annualize covariance and variance", value=True)

reg_w = regression_weights(scenario.universe, scenario.specific_vol, weight_method)
estimated = estimate_factor_returns(scenario.stock_returns, scenario.exposures, reg_w)
residuals = residuals_panel(scenario.stock_returns, scenario.exposures, estimated)
f_cov = factor_covariance(estimated, annualize=annualize)
spec_var = specific_variance(residuals, scenario.specific_vol, annualize=annualize)
sigma = stock_covariance(scenario.exposures, f_cov, spec_var)
diag = covariance_diagnostics(sigma)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rank of Sigma", f"{diag['rank']} / {sigma.shape[0]}")
c2.metric("Min eigenvalue", f"{diag['min_eigenvalue']:.2e}")
c3.metric("Condition number", f"{diag['condition_number']:.2e}")
c4.metric("PSD", "yes" if diag["is_positive_semidefinite"] else "no")

tab1, tab2, tab3, tab4 = st.tabs(["Sigma = X F X' + D", "Portfolio Risk", "Eigenvalues", "Raw Tables"])

with tab1:
    st.caption("F 衡量因子收益之间的协方差；Sigma 是由因子结构推导出的股票收益协方差。")
    st.plotly_chart(heatmap(f_cov, "Factor covariance F"))
    top = scenario.universe["market_cap"].sort_values(ascending=False).head(50).index
    st.plotly_chart(heatmap(sigma.loc[top, top], "Stock covariance Sigma, top 50 by market cap"))

with tab2:
    st.caption("组合风险是 w' Sigma w，可以拆成公共因子风险和股票特质风险。")
    base = pd.DataFrame(
        {
            "ticker": scenario.exposures.index,
            "weight": 1.0 / len(scenario.exposures),
            "market_cap_weight": (
                scenario.universe.loc[scenario.exposures.index, "market_cap"]
                / scenario.universe.loc[scenario.exposures.index, "market_cap"].sum()
            ).to_numpy(dtype=float),
        }
    )
    preset = st.radio("Portfolio preset", ["equal", "market cap"], horizontal=True)
    base["weight"] = base["weight"] if preset == "equal" else base["market_cap_weight"]
    edited = st.data_editor(
        base[["ticker", "weight"]],
        hide_index=True,
        width="stretch",
        column_config={"weight": st.column_config.NumberColumn(min_value=0.0, format="%.6f")},
    )
    portfolio_w = pd.Series(edited["weight"].to_numpy(dtype=float), index=edited["ticker"])
    if portfolio_w.sum() <= 0:
        st.error("Portfolio weights must have a positive sum.")
        st.stop()
    portfolio_w = portfolio_w / portfolio_w.sum()
    breakdown = portfolio_risk_breakdown(portfolio_w, scenario.exposures, f_cov, spec_var)

    r1, r2, r3 = st.columns(3)
    r1.metric("Total variance", f"{breakdown['total_variance']:.4e}")
    r2.metric("Volatility", f"{breakdown['volatility']:.2%}")
    r3.metric("Factor risk share", f"{breakdown['factor_variance'] / breakdown['total_variance']:.1%}")

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(bar(breakdown["factor_exposure"], "Portfolio factor exposure"))
    with right:
        st.plotly_chart(bar(breakdown["factor_contribution"], "Factor variance contribution"))

    st.dataframe(
        pd.DataFrame(
            {
                "value": {
                    "factor_variance": breakdown["factor_variance"],
                    "specific_variance": breakdown["specific_variance"],
                    "total_variance": breakdown["total_variance"],
                    "volatility": breakdown["volatility"],
                }
            }
        ).style.format("{:.6f}"),
        width="stretch",
    )

with tab3:
    st.caption("特征值用于检查协方差矩阵是否半正定，以及数值上是否稳定。")
    st.dataframe(diag["eigenvalues"].to_frame().round(10), width="stretch")
    st.plotly_chart(bar(diag["eigenvalues"], "Eigenvalues of Sigma"))

with tab4:
    st.subheader("F")
    st.caption("F 是根据回归估计出的因子收益序列计算出来的因子协方差矩阵。")
    st.dataframe(f_cov.round(8), width="stretch")
    st.subheader("D diagonal")
    st.caption("D 只在对角线上保留每只股票的特质方差，近似认为特质收益彼此不相关。")
    st.dataframe(spec_var.to_frame("specific_variance").round(8), width="stretch")
    st.subheader("Sigma sample")
    st.caption("Sigma 是完整的股票协方差矩阵，会被组合风险计算和优化器直接使用。")
    st.dataframe(sigma.iloc[:60, :60].round(8), width="stretch")
