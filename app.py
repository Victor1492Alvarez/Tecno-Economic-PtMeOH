from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
import pandas as pd
import streamlit as st
from application.case_runner import CaseRunner
from infrastructure.model_registry import ModelRegistry
from presentation.plotting import heatmap, lineprofile, tornado

PROJECT_ROOT = Path(__file__).resolve().parent
runner = CaseRunner(PROJECT_ROOT)
registry = ModelRegistry(PROJECT_ROOT)

st.set_page_config(page_title='PtMeOH Sizing Tool V3', layout='wide')
st.title('PtMeOH Plant Sizing Tool — Version 3')
st.caption('PtMeOH dispatch governed by the surrogate domain, fixed set point at the validated maximum, and controlled extrapolation below the minimum.')

def call_first_available(obj: Any, candidate_names: Iterable[str], *args, default=None, **kwargs):
    for name in candidate_names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    return default

def safe_discover_catalog() -> pd.DataFrame:
    result = call_first_available(registry, ['discover_packages', 'discoverpackages', 'catalog', 'discover'], default=pd.DataFrame())
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame()

def safe_library_names(catalog_df: pd.DataFrame) -> list[str]:
    names = call_first_available(registry, ['get_library_names', 'getlibrarynames', 'list_library_names', 'listlibrarynames'], default=None)
    if names is not None:
        names = [str(x) for x in names]
        if names:
            return names
    if not catalog_df.empty and 'library' in catalog_df.columns:
        libs = sorted(catalog_df['library'].dropna().astype(str).unique().tolist())
        if libs:
            return libs
    return ['variable_h2_constant_co2', 'variable_h2_variable_co2']

catalog_df = safe_discover_catalog()
library_names = safe_library_names(catalog_df)

with st.sidebar:
    st.header('Case inputs')
    scenario_name = st.selectbox('Techno-economic scenario', ['optimistic', 'moderate', 'pessimistic'], index=1)
    electricity_price_usd_per_kwh = st.number_input('Electricity price [USD/kWh]', min_value=0.0, value=0.055, step=0.001, format='%.4f')
    renewable_peak_power_mw = st.slider('Renewable peak power [MW]', 20.0, 300.0, 145.0, 5.0)
    electrolyzer_power_mw = st.slider('Electrolyzer nominal power [MW]', 5.0, 200.0, 82.0, 1.0)
    module_count = st.slider('Electrolyzer module count [-]', 1, 12, 4, 1)
    storage_enabled = st.toggle('Enable H2 storage', value=True)
    storage_kg_h2 = st.slider('Usable H2 storage capacity [kg H2]', 0.0, 120000.0, 26000.0, 1000.0)
    operating_mode = st.selectbox('PtMeOH operating mode', ['quasi_base_load', 'flexible'])
    surrogate_library = st.selectbox('Surrogate model library', library_names, index=0)
    allow_soft_extrapolation_below_min = st.toggle('Allow soft extrapolation below surrogate minimum', value=True)
    soft_extrapolation_margin_fraction = st.slider('Soft extrapolation margin below minimum [%]', 0.0, 10.0, 3.0, 0.5) / 100.0

    filtered_catalog = catalog_df[catalog_df['library'] == surrogate_library].copy() if not catalog_df.empty and 'library' in catalog_df.columns else pd.DataFrame()
    st.subheader('Detected model bundles')
    if filtered_catalog.empty:
        st.warning('No package folders were detected for the selected library.')
    else:
        cols = [c for c in ['model_name', 'ready_for_runtime', 'ready_for_qa', 'missing_files'] if c in filtered_catalog.columns]
        st.dataframe(filtered_catalog[cols], use_container_width=True, hide_index=True)
    confirm_bundle = st.checkbox('I confirm that the detected model folders and file sets correspond to the intended surrogate library for this run.', value=not filtered_catalog.empty)

if not confirm_bundle:
    st.error('Confirm the detected surrogate library bundle in the sidebar to continue.')
    st.stop()

case = runner.build_case(
    scenario_name=scenario_name,
    electrolyzer_power_mw=electrolyzer_power_mw,
    module_count=module_count,
    storage_enabled=storage_enabled,
    storage_kg_h2=storage_kg_h2,
    operating_mode=operating_mode,
    surrogate_library=surrogate_library,
    renewable_peak_power_mw=renewable_peak_power_mw,
    renewable_profile_df=None,
    electricity_price_usd_per_kwh=electricity_price_usd_per_kwh,
    allow_soft_extrapolation_below_min=allow_soft_extrapolation_below_min,
    soft_extrapolation_margin_fraction=soft_extrapolation_margin_fraction,
)

st.info(f"Fixed PtMeOH set point = {case.ptmeoh.fixed_setpoint_kg_per_h:.4f} kg/h | surrogate domain = [{case.ptmeoh.surrogate_domain_min_kg_per_h:.4f}, {case.ptmeoh.surrogate_domain_max_kg_per_h:.4f}] kg/h")

c1, c2, c3, c4 = st.columns(4)
run_sim = c1.button('Run annual simulation', use_container_width=True)
run_opt = c2.button('Run optimization', use_container_width=True)
run_sens = c3.button('Run sensitivities', use_container_width=True)
run_all = c4.button('Run all', use_container_width=True)

if run_all or run_sim:
    st.session_state['simulation'] = runner.run_simulation(case)
if run_all or run_opt:
    st.session_state['optimization'] = runner.run_optimization(case)
if run_all or run_sens:
    st.session_state['sensitivities'] = runner.run_sensitivity(case)

simulation = st.session_state.get('simulation')
optimization = st.session_state.get('optimization')
sensitivities = st.session_state.get('sensitivities')

inputs_tab, sim_tab, opt_tab, sens_tab = st.tabs(['Inputs', 'Annual Simulation', 'Techno-Economic Optimum', 'Sensitivities'])

with inputs_tab:
    left, right = st.columns([1, 1])
    with left:
        st.subheader('Case definition')
        st.json({
            'scenario_name': case.scenario_name,
            'electricity_price_usd_per_mwh': case.economic.electricity_price_usd_per_mwh,
            'electrolyzer_power_mw': case.electrolyzer.nominal_power_mw,
            'module_count': module_count,
            'storage_enabled': case.storage.enabled,
            'storage_kg_h2': case.storage.usable_capacity_kg_h2,
            'surrogate_library': case.ptmeoh.surrogate_library,
            'surrogate_domain_min_kg_per_h': case.ptmeoh.surrogate_domain_min_kg_per_h,
            'surrogate_domain_max_kg_per_h': case.ptmeoh.surrogate_domain_max_kg_per_h,
            'fixed_setpoint_kg_per_h': case.ptmeoh.fixed_setpoint_kg_per_h,
            'allow_soft_extrapolation_below_min': case.ptmeoh.allow_soft_extrapolation_below_min,
            'soft_extrapolation_margin_fraction': case.ptmeoh.soft_extrapolation_margin_fraction,
        })
    with right:
        st.subheader('Dispatch policy')
        st.markdown('- The PtMeOH set point is fixed at the maximum validated surrogate input.')
        st.markdown('- Above the maximum, the feed is saturated and excess H2 is stored or spilled.')
        st.markdown('- Slightly below the minimum, the model can use flagged soft extrapolation.')
        st.markdown('- Far below the minimum, the response is bounded conservatively at the lower edge and reported as hard out-of-range.')

with sim_tab:
    if simulation is None:
        st.info('Press Run annual simulation to generate results.')
    else:
        for warning in simulation.warnings:
            st.warning(warning)
        kpis = simulation.kpis
        ts = simulation.time_series
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric('Annual MeOH [t/y]', f"{kpis['annual_methanol_t']:,.2f}")
        m2.metric('Electrolyzer FLH [h/y]', f"{kpis['electrolyzer_full_load_hours_h']:,.1f}")
        m3.metric('PtMeOH utilization [-]', f"{kpis['ptmeoh_utilization_factor']:.3f}")
        m4.metric('Renewable utilization [-]', f"{kpis['renewable_utilization_fraction']:.3f}")
        m5.metric('LCOMeOH [USD/t]', f"{kpis['lcomeoh_usd_per_t_meoh']:,.2f}")
        m6.metric('NPV [USD]', f"{kpis['npv_usd']:,.0f}")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric('Soft extrapolated [-]', f"{kpis['soft_extrapolated_fraction']:.3f}")
        d2.metric('Hard out-of-range [-]', f"{kpis['hard_out_of_range_fraction']:.3f}")
        d3.metric('Set point gap [t/y]', f"{kpis['h2_not_supplied_t']:,.3f}")
        d4.metric('Curtailment fraction [-]', f"{kpis['curtailment_fraction']:.3f}")
        st.plotly_chart(lineprofile(ts.iloc[:336], ['renewable_power_mw', 'power_to_electrolyzer_mw'], 'Renewable and electrolyzer power — first two weeks', 'Power [MW]'), use_container_width=True)
        st.plotly_chart(lineprofile(ts.iloc[:336], ['h2_produced_kg_per_h', 'h2_to_ptmeoh_kg_per_h', 'tank_soc_kg_h2'], 'Hydrogen production, PtMeOH feed, and tank state of charge — first two weeks', 'H2 / SOC'), use_container_width=True)
        st.plotly_chart(lineprofile(ts.iloc[:336], ['methanol_t_per_h', 'ptmeoh_setpoint_gap_kg_per_h'], 'Methanol production and PtMeOH set point gap — first two weeks', 'Output / gap'), use_container_width=True)
        with st.expander('Surrogate and runtime diagnostics', expanded=False):
            st.write(simulation.surrogate_info)
            if not simulation.model_summary.empty:
                st.dataframe(simulation.model_summary.head(200), use_container_width=True, hide_index=True)
        with st.expander('Potentially problematic timesteps', expanded=False):
            flagged = ts[(ts['surrogate_soft_extrapolated'] > 0) | (ts['surrogate_hard_out_of_range'] > 0) | (ts['ptmeoh_setpoint_gap_kg_per_h'] > 0)]
            st.dataframe(flagged.head(200), use_container_width=True)
        with st.expander('Traceable hourly results preview', expanded=False):
            st.dataframe(ts.head(96), use_container_width=True)

with opt_tab:
    if optimization is None:
        st.info('Press Run optimization to generate the design ranking.')
    else:
        left, right = st.columns([1, 1.4])
        with left:
            st.subheader('Recommended configuration')
            st.dataframe(optimization.best_row.to_frame(name='value'), use_container_width=True)
        with right:
            st.plotly_chart(heatmap(optimization.results, z_col='lcomeoh_usd_per_t_meoh'), use_container_width=True)
        st.subheader('Ranked shortlist')
        cols = [c for c in ['case_name', 'electrolyzer_power_mw', 'module_count', 'storage_kg_h2', 'fixed_setpoint_kg_per_h', 'annual_methanol_t', 'ptmeoh_utilization_factor', 'renewable_utilization_fraction', 'soft_extrapolated_fraction', 'hard_out_of_range_fraction', 'warning_count', 'feasible', 'lcomeoh_usd_per_t_meoh', 'npv_usd'] if c in optimization.results.columns]
        st.dataframe(optimization.results.sort_values(['lcomeoh_usd_per_t_meoh', 'warning_count'])[cols].head(20), use_container_width=True)

with sens_tab:
    if sensitivities is None:
        st.info('Press Run sensitivities to generate the tornado view.')
    else:
        st.plotly_chart(tornado(sensitivities), use_container_width=True)
        st.dataframe(sensitivities, use_container_width=True)
