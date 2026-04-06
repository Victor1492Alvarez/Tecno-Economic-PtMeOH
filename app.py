from __future__ import annotations
from pathlib import Path
import streamlit as st
from application.case_runner import CaseRunner
from infrastructure.model_registry import ModelRegistry
from presentation.plotting import heatmap, line_profile, tornado

PROJECT_ROOT = Path(__file__).resolve().parent
runner = CaseRunner(PROJECT_ROOT)
registry = ModelRegistry(PROJECT_ROOT)

st.set_page_config(page_title="PtMeOH Sizing Tool V1", layout="wide")
st.title("PtMeOH Plant Sizing Tool — Version 1")
st.caption("Annual deterministic simulator, multi-surrogate PtMeOH response, techno-economics, and grid-search design exploration")

catalog_df = registry.discover_packages()

with st.sidebar:
    st.header("Case inputs")
    scenario_name = st.selectbox("Techno-economic scenario", ["optimistic", "moderate", "pessimistic"], index=1)
    renewable_peak_power_mw = st.slider("Renewable peak power [MW]", 50.0, 250.0, 145.0, 5.0)
    electrolyzer_power_mw = st.slider("Electrolyzer nominal power [MW]", 10.0, 180.0, 82.0, 2.0)
    module_count = st.slider("Electrolyzer module count [-]", 1, 12, 4, 1)
    storage_enabled = st.toggle("Enable H2 storage", value=True)
    storage_kg_h2 = st.slider("Usable H2 storage capacity [kg H2]", 0.0, 120000.0, 26000.0, 1000.0)
    operating_mode = st.selectbox("PtMeOH operating mode", ["quasi_base_load", "flexible"])
    surrogate_library = st.selectbox("Surrogate model library", ["variable_h2_constant_co2", "variable_h2_variable_co2"])
    target_h2_kg_per_h = st.slider("Target H2 feed to PtMeOH [kg/h]", 100.0, 4000.0, 1850.0, 50.0)
    max_h2_feed_kg_per_h = st.slider("Maximum PtMeOH H2 intake [kg/h]", 200.0, 5000.0, 2200.0, 50.0)

    filtered_catalog = catalog_df[catalog_df["library"] == surrogate_library].copy() if not catalog_df.empty else catalog_df
    st.subheader("Detected model bundles")
    if filtered_catalog.empty:
        st.warning("No package folders were detected yet for the selected library.")
    else:
        st.dataframe(filtered_catalog[["model_name", "ready_for_runtime", "ready_for_qa", "missing_files"]], use_container_width=True, hide_index=True)
    confirm_bundle = st.checkbox("I confirm that the detected model folders and file sets correspond to the intended surrogate library for this run.", value=not filtered_catalog.empty)

if not confirm_bundle:
    st.error("Confirm the detected surrogate library bundle in the sidebar to continue.")
    st.stop()

case = runner.build_case(
    scenario_name=scenario_name,
    electrolyzer_power_mw=electrolyzer_power_mw,
    module_count=module_count,
    storage_enabled=storage_enabled,
    storage_kg_h2=storage_kg_h2,
    operating_mode=operating_mode,
    surrogate_library=surrogate_library,
    target_h2_kg_per_h=target_h2_kg_per_h,
    max_h2_feed_kg_per_h=max_h2_feed_kg_per_h,
    renewable_peak_power_mw=renewable_peak_power_mw,
)

results = runner.run_all(case)
simulation = results["simulation"]
optimization = results["optimization"]
sensitivities = results["sensitivities"]

tab1, tab2, tab3, tab4 = st.tabs(["Inputs", "Annual Simulation", "Techno-Economic Optimum", "Sensitivities"])

with tab1:
    c1, c2, c3 = st.columns([1.1, 1.1, 1.0])
    with c1:
        st.subheader("Case definition")
        st.json({
            "scenario": scenario_name,
            "renewable_peak_power_mw": renewable_peak_power_mw,
            "electrolyzer_power_mw": electrolyzer_power_mw,
            "module_count": module_count,
            "storage_enabled": storage_enabled,
            "storage_kg_h2": storage_kg_h2,
            "operating_mode": operating_mode,
            "surrogate_library": surrogate_library,
        })
    with c2:
        st.subheader("Model package summary")
        st.write(simulation.surrogate_info)
        st.dataframe(simulation.model_summary, use_container_width=True, hide_index=True)
    with c3:
        st.subheader("Plant schematic")
        st.markdown("""
        ```
        Renewable Power -> Electrolyzer Battery -> H2 Buffer Tank -> PtMeOH Train
                               |                      |
                            Curtailment           Dispatch smoothing
        ```
        """)
        st.info("The engine preserves stepwise traceability: renewable input -> electrolysis -> H2 balance -> surrogate response -> economics -> ranking.")

with tab2:
    for warning in simulation.warnings:
        st.warning(warning)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Annual MeOH [t/y]", f"{simulation.kpis['annual_methanol_t']:,.0f}")
    m2.metric("Electrolyzer FLH [h/y]", f"{simulation.kpis['electrolyzer_full_load_hours_h']:,.0f}")
    m3.metric("PtMeOH utilization [-]", f"{simulation.kpis['ptmeoh_utilization_factor']:.2f}")
    m4.metric("Renewable utilization [-]", f"{simulation.kpis['renewable_utilization_fraction']:.2f}")
    m5.metric("LCOMeOH [USD/t]", f"{simulation.kpis['lcomeoh_usd_per_t_meoh']:,.1f}")
    m6.metric("NPV [USD]", f"{simulation.kpis['npv_usd']:,.0f}")

    ts = simulation.time_series
    st.plotly_chart(line_profile(ts.iloc[:336], ["renewable_power_mw", "power_to_electrolyzer_mw"], "Renewable and electrolyzer power — first two weeks", "Power [MW]"), use_container_width=True)
    st.plotly_chart(line_profile(ts.iloc[:336], ["h2_produced_kg_per_h", "h2_to_ptmeoh_kg_per_h", "tank_soc_kg_h2"], "Hydrogen production, PtMeOH feed, and tank state of charge — first two weeks", "H2 / SOC [kg or kg/h]"), use_container_width=True)
    st.plotly_chart(line_profile(ts.iloc[:336], ["methanol_t_per_h", "unmet_h2_kg_per_h"], "Methanol production and unmet H2 — first two weeks", "Production / unmet H2"), use_container_width=True)
    st.subheader("Traceable hourly results preview")
    st.dataframe(ts.head(48), use_container_width=True)

with tab3:
    left, right = st.columns([1.0, 1.6])
    with left:
        st.subheader("Recommended configuration")
        st.dataframe(optimization.best_row.to_frame(name="value"), use_container_width=True)
        st.success("Recommended design balances CAPEX, renewable utilization, PtMeOH stability, and storage buffering under the chosen scenario.")
        st.info("CAPEX vs stability: larger electrolyzers and larger tanks reduce unmet H2 risk, but they can increase capital intensity and leave more underutilized capacity if the renewable profile is not strong enough.")
    with right:
        st.plotly_chart(heatmap(optimization.results, z_col="lcomeoh_usd_per_t_meoh"), use_container_width=True)
    st.subheader("Ranked shortlist")
    st.dataframe(optimization.results.sort_values(["lcomeoh_usd_per_t_meoh", "warning_count"]).head(12), use_container_width=True)

with tab4:
    st.plotly_chart(tornado(sensitivities), use_container_width=True)
    st.dataframe(sensitivities, use_container_width=True)
