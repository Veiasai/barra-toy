from __future__ import annotations

import plotly.express as px
import streamlit as st

from barra_lab.data import STYLE_FACTORS
from barra_lab.plots import heatmap
from barra_lab.ui import configure_page, load_scenario, scenario_sidebar


configure_page("Factor Exposures")

st.title("Factor Exposures")

params = scenario_sidebar()
scenario = load_scenario(params)

st.caption("观察原始股票描述符如何变成 Barra 风格的因子暴露。")

factor = st.selectbox("Style factor", STYLE_FACTORS)
steps = scenario.exposure_steps[scenario.exposure_steps["factor"] == factor].set_index("ticker")

summary = steps[["raw", "winsorized", "exposure"]].describe().T
st.subheader("Descriptor Transformation Summary")
st.caption("这张表展示原始描述符如何经过去极值处理，再转换成最终因子暴露。")
st.dataframe(summary.round(4), width="stretch")

left, right = st.columns([1, 1])
with left:
    fig = px.histogram(
        steps.reset_index(),
        x=["raw", "winsorized", "exposure"],
        barmode="overlay",
        title=f"{factor}: raw -> winsorized -> exposure",
        opacity=0.65,
    )
    fig.update_layout(height=440, margin=dict(l=20, r=20, t=50, b=30))
    st.plotly_chart(fig)
with right:
    st.dataframe(steps.sort_values("exposure", ascending=False).head(30).round(4), width="stretch")

st.subheader("Exposure Matrix X")
st.caption("行是股票，列是风险因子；这个矩阵把个股收益和因子收益连接起来。")
row_limit = st.slider("Rows to display", 10, len(scenario.exposures), min(80, len(scenario.exposures)), step=10)
st.dataframe(scenario.exposures.head(row_limit).round(3), width="stretch")

st.subheader("Exposure Heatmap")
st.caption("热力图用来快速观察哪些股票在某些风格或行业因子上暴露较高。")
st.plotly_chart(heatmap(scenario.exposures.head(row_limit).T, "X: factors x stocks"))

st.subheader("Factor Correlation")
st.caption("这里检查不同因子暴露之间是否高度相关，相关过高会影响回归稳定性和解释性。")
st.plotly_chart(heatmap(scenario.exposures.corr().round(3), "Cross-sectional factor exposure correlation"))
