from __future__ import annotations

import pandas as pd
import streamlit as st

from barra_lab.data import make_scenario


def configure_page(title: str) -> None:
    st.set_page_config(page_title=title, layout="wide")


def scenario_sidebar() -> dict[str, object]:
    st.sidebar.header("Scenario")
    n_stocks = st.sidebar.slider("Stocks", 20, 200, 80, step=10, key="scenario_n_stocks")
    n_days = st.sidebar.slider("Trading days", 20, 252, 80, step=10, key="scenario_n_days")
    universe_seed = st.sidebar.number_input("Universe seed", 0, 9999, 7, step=1, key="scenario_universe_seed")
    return_seed = st.sidebar.number_input("Return seed", 0, 9999, 11, step=1, key="scenario_return_seed")
    winsor_z = st.sidebar.slider("Winsorize z", 1.0, 5.0, 3.0, step=0.25, key="scenario_winsor_z")
    standardize_styles = st.sidebar.checkbox(
        "Standardize style factors",
        value=True,
        key="scenario_standardize_styles",
    )
    specific_scale = st.sidebar.slider(
        "Specific risk scale",
        0.25,
        3.0,
        1.0,
        step=0.25,
        key="scenario_specific_scale",
    )
    style_vol = st.sidebar.slider(
        "Style factor daily vol",
        0.001,
        0.020,
        0.006,
        step=0.001,
        format="%.3f",
        key="scenario_style_vol",
    )
    industry_vol = st.sidebar.slider(
        "Industry factor daily vol",
        0.001,
        0.020,
        0.004,
        step=0.001,
        format="%.3f",
        key="scenario_industry_vol",
    )
    return {
        "n_stocks": n_stocks,
        "n_days": n_days,
        "universe_seed": int(universe_seed),
        "return_seed": int(return_seed),
        "winsor_z": winsor_z,
        "standardize_styles": standardize_styles,
        "specific_scale": specific_scale,
        "style_vol": style_vol,
        "industry_vol": industry_vol,
    }


@st.cache_data(show_spinner=False)
def load_scenario(params: dict[str, object]):
    return make_scenario(**params)


def format_pct_table(df: pd.DataFrame, columns: list[str]) -> pd.io.formats.style.Styler:
    return df.style.format({col: "{:.3%}" for col in columns if col in df.columns})
