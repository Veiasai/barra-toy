from __future__ import annotations

import pandas as pd
import streamlit as st

from barra_lab.data import regression_weights
from barra_lab.plots import bar, heatmap, line
from barra_lab.regression import regression_steps
from barra_lab.ui import configure_page, load_scenario, scenario_sidebar


configure_page("Regression Steps")

st.title("Cross-Sectional Regression Steps")
st.caption("这个页面估计某一天的因子收益，并展示加权最小二乘公式里的主要中间矩阵。")

params = scenario_sidebar()
scenario = load_scenario(params)

date = st.selectbox("Regression date", scenario.stock_returns.index, format_func=lambda x: x.strftime("%Y-%m-%d"))
weight_method = st.selectbox(
    "Initial regression weight",
    ["sqrt_market_cap", "market_cap", "inverse_specific_var", "equal"],
    format_func=lambda x: {
        "sqrt_market_cap": "sqrt market cap",
        "market_cap": "market cap",
        "inverse_specific_var": "inverse specific variance",
        "equal": "equal",
    }[x],
)

st.subheader("Historical Stock Returns")
st.caption("这张图展示进入截面回归的个股收益历史；选择某一天回归时，本质上就是取这条历史序列中的一个截面。")
default_stocks = scenario.universe["market_cap"].sort_values(ascending=False).head(5).index.tolist()
selected_stocks = st.multiselect(
    "Stocks to plot",
    scenario.stock_returns.columns.tolist(),
    default=default_stocks,
)
if selected_stocks:
    daily_returns = scenario.stock_returns[selected_stocks]
    cumulative_returns = (1.0 + daily_returns).cumprod() - 1.0
    chart_left, chart_right = st.columns([1, 1])
    with chart_left:
        st.plotly_chart(line(daily_returns, "Historical daily stock returns"))
    with chart_right:
        st.plotly_chart(line(cumulative_returns, "Cumulative stock returns"))
else:
    st.info("Select at least one stock to show historical returns.")

base_weights = regression_weights(scenario.universe, scenario.specific_vol, weight_method)
edit_frame = pd.DataFrame(
    {
        "ticker": scenario.exposures.index,
        "industry": scenario.universe.loc[scenario.exposures.index, "industry"].to_numpy(),
        "actual_return": scenario.stock_returns.loc[date, scenario.exposures.index].to_numpy(dtype=float),
        "regression_weight": base_weights.loc[scenario.exposures.index].to_numpy(dtype=float),
    }
)

st.caption("可以直接编辑个股收益或回归权重，下方的 WLS 矩阵会随之重新计算。")
edited = st.data_editor(
    edit_frame,
    hide_index=True,
    width="stretch",
    column_config={
        "actual_return": st.column_config.NumberColumn(format="%.5f"),
        "regression_weight": st.column_config.NumberColumn(min_value=0.0, format="%.6f"),
    },
)

returns = pd.Series(edited["actual_return"].to_numpy(dtype=float), index=edited["ticker"], name=date)
weights = pd.Series(edited["regression_weight"].to_numpy(dtype=float), index=edited["ticker"], name="weight")
if weights.sum() <= 0:
    st.error("Regression weights must have a positive sum.")
    st.stop()
weights = weights / weights.sum()

steps = regression_steps(returns, scenario.exposures.loc[weights.index], weights)
diag = steps["diagnostics"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rank of X'WX", f"{diag['rank']} / {diag['n_factors']}")
c2.metric("Condition number", f"{diag['condition_number']:.2e}")
c3.metric("Min eigenvalue", f"{diag['min_eigenvalue']:.2e}")
c4.metric("Weighted SSE", f"{diag['weighted_sse']:.2e}")

tab1, tab2, tab3, tab4 = st.tabs(["Formula Output", "Matrices", "Attribution", "Eigenvalues"])

with tab1:
    st.subheader("f_hat = pinv(X' W X) X' W r")
    st.caption("f_hat 是所选日期估计出的因子收益向量；这里用 pinv 是为了让玩具样例在矩阵接近奇异时也能稳定运行。")
    st.dataframe(steps["factor_returns"].to_frame().style.format("{:.5%}"), width="stretch")
    st.plotly_chart(bar(steps["factor_returns"], "Estimated factor returns", "return"))

with tab2:
    st.caption("这些矩阵是加权最小二乘计算中的中间对象，可以对应到公式里的每一项。")
    matrices = steps["matrices"]
    selected_matrix = st.selectbox("Matrix", list(matrices.keys()))
    matrix = matrices[selected_matrix]
    if isinstance(matrix, pd.Series):
        st.dataframe(matrix.to_frame().round(8), width="stretch")
    else:
        st.dataframe(matrix.round(8), width="stretch")
        if matrix.shape[0] <= 120 and matrix.shape[1] <= 120:
            st.plotly_chart(heatmap(matrix, selected_matrix))

with tab3:
    st.caption("收益归因把每只股票的实际收益拆成因子解释部分和剩下的特质残差。")
    attribution = steps["attribution"]
    st.dataframe(attribution.style.format("{:.4%}"), width="stretch")
    st.plotly_chart(
        bar(attribution["specific_return_residual"].sort_values(key=abs, ascending=False).head(30), "Largest residuals")
    )

with tab4:
    st.caption("特征值用于观察数值稳定性；如果有很小的特征值，说明 X'WX 可能接近奇异。")
    st.dataframe(steps["eigenvalues"].to_frame().round(10), width="stretch")
    st.plotly_chart(bar(steps["eigenvalues"], "Eigenvalues of X'WX"))
