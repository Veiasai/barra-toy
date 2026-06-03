from __future__ import annotations

import pandas as pd
import streamlit as st

from barra_lab.data import make_scenario


SIMULATION_PRESETS = {
    "Clean teaching": {
        "specific_scale": 0.35,
        "style_vol": 0.010,
        "industry_vol": 0.008,
        "description": "低特质噪声、高因子信号，适合先看清回归链路。",
    },
    "Realistic noisy": {
        "specific_scale": 1.00,
        "style_vol": 0.006,
        "industry_vol": 0.004,
        "description": "特质噪声较高，更接近真实股票收益的嘈杂状态。",
    },
    "Stress test": {
        "specific_scale": 2.00,
        "style_vol": 0.003,
        "industry_vol": 0.002,
        "description": "高噪声、弱因子信号，用来观察模型解释力下降。",
    },
}


def configure_page(title: str) -> None:
    st.set_page_config(page_title=title, layout="wide")


def _scenario_widget(field: str, default: object) -> tuple[str, str]:
    """
    Keep scenario values alive across Streamlit pages.

    Streamlit can clean up widget-owned keys when changing pages. The visible
    widget therefore uses a temporary key, while the durable value is stored in
    a non-widget session_state key.
    """
    store_key = f"scenario_value_{field}"
    old_widget_key = f"scenario_{field}"
    widget_key = f"_scenario_{field}"

    if store_key not in st.session_state:
        st.session_state[store_key] = st.session_state.get(old_widget_key, default)
    st.session_state[widget_key] = st.session_state[store_key]
    return widget_key, store_key


def _save_scenario_widget(widget_key: str, store_key: str) -> None:
    st.session_state[store_key] = st.session_state[widget_key]


def scenario_sidebar() -> dict[str, object]:
    st.sidebar.header("Scenario")

    n_stocks_key, n_stocks_store = _scenario_widget("n_stocks", 80)
    n_days_key, n_days_store = _scenario_widget("n_days", 80)
    universe_seed_key, universe_seed_store = _scenario_widget("universe_seed", 7)
    return_seed_key, return_seed_store = _scenario_widget("return_seed", 11)
    winsor_z_key, winsor_z_store = _scenario_widget("winsor_z", 3.0)
    standardize_key, standardize_store = _scenario_widget("standardize_styles", True)
    simulation_mode_key, simulation_mode_store = _scenario_widget("simulation_mode", "Clean teaching")
    specific_scale_key, specific_scale_store = _scenario_widget("specific_scale", 1.0)
    style_vol_key, style_vol_store = _scenario_widget("style_vol", 0.006)
    industry_vol_key, industry_vol_store = _scenario_widget("industry_vol", 0.004)

    n_stocks = st.sidebar.slider(
        "Stocks",
        20,
        200,
        step=10,
        key=n_stocks_key,
        on_change=_save_scenario_widget,
        args=(n_stocks_key, n_stocks_store),
    )
    n_days = st.sidebar.slider(
        "Trading days",
        20,
        252,
        step=10,
        key=n_days_key,
        on_change=_save_scenario_widget,
        args=(n_days_key, n_days_store),
    )
    universe_seed = st.sidebar.number_input(
        "Universe seed",
        0,
        9999,
        step=1,
        key=universe_seed_key,
        on_change=_save_scenario_widget,
        args=(universe_seed_key, universe_seed_store),
    )
    return_seed = st.sidebar.number_input(
        "Return seed",
        0,
        9999,
        step=1,
        key=return_seed_key,
        on_change=_save_scenario_widget,
        args=(return_seed_key, return_seed_store),
    )
    winsor_z = st.sidebar.slider(
        "Winsorize z",
        1.0,
        5.0,
        step=0.25,
        key=winsor_z_key,
        on_change=_save_scenario_widget,
        args=(winsor_z_key, winsor_z_store),
    )
    standardize_styles = st.sidebar.checkbox(
        "Standardize style factors",
        key=standardize_key,
        on_change=_save_scenario_widget,
        args=(standardize_key, standardize_store),
    )
    simulation_mode_options = ["Clean teaching", "Realistic noisy", "Stress test", "Custom"]
    if st.session_state[simulation_mode_key] not in simulation_mode_options:
        st.session_state[simulation_mode_key] = "Clean teaching"
        st.session_state[simulation_mode_store] = "Clean teaching"
    simulation_mode = st.sidebar.selectbox(
        "Simulation mode",
        simulation_mode_options,
        key=simulation_mode_key,
        on_change=_save_scenario_widget,
        args=(simulation_mode_key, simulation_mode_store),
    )
    if simulation_mode == "Custom":
        specific_scale = st.sidebar.slider(
            "Specific risk scale",
            0.25,
            3.0,
            step=0.25,
            key=specific_scale_key,
            on_change=_save_scenario_widget,
            args=(specific_scale_key, specific_scale_store),
        )
        style_vol = st.sidebar.slider(
            "Style factor daily vol",
            0.001,
            0.020,
            step=0.001,
            format="%.3f",
            key=style_vol_key,
            on_change=_save_scenario_widget,
            args=(style_vol_key, style_vol_store),
        )
        industry_vol = st.sidebar.slider(
            "Industry factor daily vol",
            0.001,
            0.020,
            step=0.001,
            format="%.3f",
            key=industry_vol_key,
            on_change=_save_scenario_widget,
            args=(industry_vol_key, industry_vol_store),
        )
    else:
        preset = SIMULATION_PRESETS[simulation_mode]
        specific_scale = preset["specific_scale"]
        style_vol = preset["style_vol"]
        industry_vol = preset["industry_vol"]
        st.sidebar.caption(preset["description"])
        st.sidebar.caption(
            f"effective: specific scale={specific_scale:.2f}, style vol={style_vol:.3f}, "
            f"industry vol={industry_vol:.3f}"
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
