from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile

import pandas as pd
import streamlit as st

from application.case_runner import CaseRunner
from infrastructure.model_registry import ModelRegistry
from presentation.plotting import heatmap, line_profile, tornado

PROJECT_ROOT = Path(__file__).resolve().parent
PERSIST_ROOT = PROJECT_ROOT / "user_data"
PROFILE_STORE = PERSIST_ROOT / "renewable_profiles"
MODEL_ARCHIVE_STORE = PERSIST_ROOT / "model_archives"

runner = CaseRunner(PROJECT_ROOT)
registry = ModelRegistry(PROJECT_ROOT)

st.set_page_config(page_title="PtMeOH Sizing Tool V1", layout="wide")
st.title("PtMeOH Plant Sizing Tool — Version 1")
st.caption(
    "Annual deterministic simulator, multi-surrogate PtMeOH response, "
    "techno-economics, and grid-search design exploration"
)


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    return cleaned.strip("_") or "asset"


def build_case_signature(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def reset_results_for_new_case(case_signature: str) -> None:
    previous_signature = st.session_state.get("case_signature")
    if previous_signature != case_signature:
        st.session_state["case_signature"] = case_signature
        st.session_state["simulation"] = None
        st.session_state["optimization"] = None
        st.session_state["sensitivities"] = None


def flash_message(kind: str, text: str) -> None:
    st.session_state["flash_kind"] = kind
    st.session_state["flash_text"] = text


def render_flash_message() -> None:
    kind = st.session_state.pop("flash_kind", None)
    text = st.session_state.pop("flash_text", None)
    if kind and text:
        getattr(st, kind)(text)


def read_tabular_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    if suffix == ".tsv":
        return pd.read_csv(uploaded_file, sep="\t")
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    if suffix == ".parquet":
        return pd.read_parquet(uploaded_file)
    raise ValueError(f"Unsupported file type: {suffix}")


def choose_default_column(columns: list[str], keywords: list[str]) -> int:
    lowered = [c.lower() for c in columns]
    for kw in keywords:
        for idx, col in enumerate(lowered):
            if kw in col:
                return idx
    return 0


def normalize_renewable_profile(
    raw_df: pd.DataFrame,
    timestamp_col: str,
    renewable_col: str,
    unit_mode: str,
) -> pd.DataFrame:
    df = raw_df[[timestamp_col, renewable_col]].copy()
    df.columns = ["timestamp", "raw_value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["raw_value"] = pd.to_numeric(df["raw_value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "raw_value"]).sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError("The selected renewable profile columns produced an empty dataset.")

    diffs_h = (
        df["timestamp"].diff().dropna().dt.total_seconds().div(3600.0)
        if len(df) > 1
        else pd.Series(dtype=float)
    )
    median_step_h = float(diffs_h.median()) if not diffs_h.empty else 24.0

    looks_daily = (
        len(df) <= 370
        and df["timestamp"].dt.hour.eq(0).all()
        and df["timestamp"].dt.minute.eq(0).all()
        and df["timestamp"].dt.second.eq(0).all()
        and df["timestamp"].dt.normalize().nunique() == len(df)
    ) or median_step_h >= 23.0

    if unit_mode == "MW":
        df["renewable_power_mw"] = df["raw_value"]
    elif unit_mode == "kW":
        df["renewable_power_mw"] = df["raw_value"] / 1000.0
    elif unit_mode == "MWh/day":
        df["renewable_power_mw"] = df["raw_value"] / 24.0
    elif unit_mode == "kWh/day":
        df["renewable_power_mw"] = df["raw_value"] / 24000.0
    elif unit_mode == "MWh/interval":
        interval_h = max(median_step_h, 1.0)
        df["renewable_power_mw"] = df["raw_value"] / interval_h
    elif unit_mode == "kWh/interval":
        interval_h = max(median_step_h, 1.0)
        df["renewable_power_mw"] = df["raw_value"] / 1000.0 / interval_h
    else:
        raise ValueError(f"Unsupported unit mode: {unit_mode}")

    df["renewable_power_mw"] = df["renewable_power_mw"].clip(lower=0.0)

    if looks_daily:
        expanded_rows: list[dict] = []
        for _, row in df.iterrows():
            day_start = row["timestamp"].normalize()
            for hour in range(24):
                expanded_rows.append(
                    {
                        "timestamp": day_start + pd.Timedelta(hours=hour),
                        "renewable_power_mw": float(row["renewable_power_mw"]),
                    }
                )
        out = pd.DataFrame(expanded_rows)
    else:
        out = df[["timestamp", "renewable_power_mw"]].copy()

    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return out


def save_profile_assets(
    source_file,
    normalized_df: pd.DataFrame,
    profile_label: str,
    persist_enabled: bool,
) -> None:
    if not persist_enabled:
        return

    PROFILE_STORE.mkdir(parents=True, exist_ok=True)
    slug = slugify(profile_label)

    if source_file is not None:
        raw_suffix = Path(source_file.name).suffix.lower() or ".bin"
        raw_path = PROFILE_STORE / f"{slug}__source{raw_suffix}"
        raw_path.write_bytes(source_file.getvalue())

    normalized_path = PROFILE_STORE / f"{slug}__normalized.csv"
    normalized_df.to_csv(normalized_path, index=False)


def list_saved_profiles() -> list[Path]:
    if not PROFILE_STORE.exists():
        return []
    return sorted(PROFILE_STORE.glob("*__normalized.csv"))


def load_saved_profile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["renewable_power_mw"] = pd.to_numeric(df["renewable_power_mw"], errors="coerce")
    df = df.dropna(subset=["timestamp", "renewable_power_mw"]).sort_values("timestamp").reset_index(drop=True)
    return df


def expected_bundle_status(project_root: Path, library_name: str, model_names: list[str]) -> pd.DataFrame:
    rows = []
    for model_name in model_names:
        model_dir = project_root / "models" / "packages" / library_name / model_name
        expected = {
            "joblib": model_dir / f"{model_name}.joblib",
            "py": model_dir / f"{model_name}.py",
            "txt": model_dir / f"{model_name}.txt",
            "metadata": model_dir / "metadata.json",
            "parameters": model_dir / "model_parameters.xlsx",
            "consolidated_report": model_dir / "consolidated_model_report.pdf",
            "training_report": model_dir / "training_validation_report.pdf",
        }
        status = {key: path.exists() for key, path in expected.items()}
        rows.append(
            {
                "library": library_name,
                "model_name": model_name,
                "folder_exists": model_dir.exists(),
                "joblib_found": status["joblib"],
                "py_found": status["py"],
                "txt_found": status["txt"],
                "metadata_found": status["metadata"],
                "parameters_found": status["parameters"],
                "consolidated_report_found": status["consolidated_report"],
                "training_report_found": status["training_report"],
                "ready_for_runtime": status["joblib"] and status["py"],
                "ready_for_qa": (
                    status["metadata"]
                    and status["parameters"]
                    and status["consolidated_report"]
                    and status["training_report"]
                ),
                "missing_files": ", ".join([name for name, ok in status.items() if not ok]),
            }
        )
    return pd.DataFrame(rows)


def install_model_zip(
    project_root: Path,
    library_name: str,
    model_name: str,
    uploaded_zip,
    persist_enabled: bool,
) -> dict:
    target_dir = project_root / "models" / "packages" / library_name / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    if persist_enabled:
        archive_dir = MODEL_ARCHIVE_STORE / library_name
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{model_name}.zip"
        archive_path.write_bytes(uploaded_zip.getvalue())

    written_files: list[str] = []
    zip_bytes = BytesIO(uploaded_zip.getvalue())

    with ZipFile(zip_bytes) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if not members:
            raise ValueError("The ZIP file is empty or does not contain files.")

        for member in members:
            filename = Path(member.filename).name
            if not filename:
                continue

            destination = target_dir / filename
            with zf.open(member) as src, open(destination, "wb") as dst:
                dst.write(src.read())
            written_files.append(filename)

    if not written_files:
        raise ValueError("No valid files were extracted from the ZIP archive.")

    return {
        "library": library_name,
        "model_name": model_name,
        "target_dir": str(target_dir.relative_to(project_root)),
        "written_files": sorted(written_files),
        "written_count": len(written_files),
    }


def run_simulation(case):
    return runner.engine.run(case)


def run_optimization(case):
    return runner.optimizer.run(case)


def run_sensitivities(case):
    return runner.sensitivity.run(case)


render_flash_message()

active_profile_df = st.session_state.get("renewable_profile_df")
active_profile_name = st.session_state.get("renewable_profile_name", "default_synthetic_profile")

with st.sidebar:
    st.header("Case inputs")

    persist_assets = st.toggle(
        "Persist uploaded model ZIPs and renewable profile for future iterations",
        value=True,
        help="When enabled, uploaded model archives and normalized renewable profiles are written to disk under user_data/ for reuse in later sessions.",
    )

    scenario_name = st.selectbox(
        "Techno-economic scenario",
        ["optimistic", "moderate", "pessimistic"],
        index=1,
    )

    default_electricity_price_usd_per_kwh = (
        float(runner.scenario_config[scenario_name]["electricity_price_usd_per_mwh"]) / 1000.0
    )

    electricity_price_usd_per_kwh = st.number_input(
        "Electricity price [USD/kWh]",
        min_value=0.0,
        value=default_electricity_price_usd_per_kwh,
        step=0.001,
        format="%.4f",
        help="This value overrides the electricity-price assumption of the selected techno-economic scenario for the current case.",
    )

    electrolyzer_power_mw = st.number_input(
        "Electrolyzer nominal power [MW]",
        min_value=0.1,
        value=82.0,
        step=1.0,
        format="%.3f",
    )

    module_count = st.number_input(
        "Electrolyzer module count [-]",
        min_value=1,
        value=4,
        step=1,
    )

    storage_enabled = st.toggle("Enable H2 storage", value=True)

    storage_kg_h2 = st.number_input(
        "Usable H2 storage capacity [kg H2]",
        min_value=0.0,
        value=26000.0,
        step=100.0,
        format="%.2f",
    )

    operating_mode = st.selectbox(
        "PtMeOH operating mode",
        ["quasi_base_load", "flexible"],
    )

    surrogate_library = st.selectbox(
        "Surrogate model library",
        ["variable_h2_constant_co2", "variable_h2_variable_co2"],
    )

    target_h2_kg_per_h = st.number_input(
        "Target H2 feed to PtMeOH [kg/h]",
        min_value=0.0,
        value=1850.0,
        step=10.0,
        format="%.3f",
        help=(
            "Dispatch target to the downstream PtMeOH section. "
            "The simulator tries to deliver this hourly H2 flow to the methanol train; "
            "if renewable production is insufficient, the H2 buffer discharges first and any remaining deficit becomes unmet H2."
        ),
    )

    max_h2_feed_kg_per_h = st.number_input(
        "Maximum PtMeOH H2 intake [kg/h]",
        min_value=0.0,
        value=2200.0,
        step=10.0,
        format="%.3f",
        help=(
            "Hard upper bound on how much H2 the downstream PtMeOH train can physically absorb. "
            "This caps reactor/compression throughput, methanol production and the electricity consumption predicted by downstream compressor models."
        ),
    )

    with st.expander("Explain downstream H2 variables", expanded=False):
        st.markdown(
            """
- **Target H2 feed to PtMeOH [kg/h]**: hourly H2 delivery target for the methanol train.  
  It affects storage dispatch, unmet-H2 events, PtMeOH utilization and all surrogate outputs.

- **Maximum PtMeOH H2 intake [kg/h]**: downstream processing ceiling.  
  It limits the H2 that can actually enter the PtMeOH block, so it caps methanol output and the auxiliary electric demand estimated by C1, C2 and C3.
            """
        )

    st.subheader("Renewable profile database")

    profile_source = st.radio(
        "Renewable profile source",
        ["Synthetic default profile", "Upload renewable profile file", "Use saved renewable profile"],
        index=0,
    )

    renewable_peak_power_mw = 145.0

    if profile_source == "Synthetic default profile":
        renewable_peak_power_mw = st.number_input(
            "Renewable peak power [MW]",
            min_value=0.1,
            value=145.0,
            step=1.0,
            format="%.3f",
        )
        active_profile_df = None
        active_profile_name = "default_synthetic_profile"

    elif profile_source == "Upload renewable profile file":
        uploaded_profile_file = st.file_uploader(
            "Upload renewable profile database",
            type=["csv", "txt", "tsv", "xlsx", "xls", "parquet"],
            accept_multiple_files=False,
            help="Supported types: CSV, TXT, TSV, XLSX, XLS and Parquet.",
        )

        if uploaded_profile_file is not None:
            try:
                raw_profile_df = read_tabular_file(uploaded_profile_file)
                st.caption(f"Detected columns: {', '.join(raw_profile_df.columns.astype(str).tolist())}")

                timestamp_idx = choose_default_column(
                    raw_profile_df.columns.astype(str).tolist(),
                    ["timestamp", "datetime", "date", "time", "fecha"],
                )
                renewable_idx = choose_default_column(
                    raw_profile_df.columns.astype(str).tolist(),
                    ["renewable", "energy", "power", "mw", "mwh", "generation", "available", "disponible"],
                )

                timestamp_col = st.selectbox(
                    "Date / timestamp column",
                    raw_profile_df.columns,
                    index=timestamp_idx,
                )
                renewable_col = st.selectbox(
                    "Renewable availability column",
                    raw_profile_df.columns,
                    index=renewable_idx,
                )
                unit_mode = st.selectbox(
                    "Selected renewable column units",
                    ["MW", "kW", "MWh/day", "kWh/day", "MWh/interval", "kWh/interval"],
                    index=0,
                )

                st.dataframe(raw_profile_df.head(12), use_container_width=True)

                if st.button("Load this renewable profile", use_container_width=True):
                    normalized_profile = normalize_renewable_profile(
                        raw_profile_df,
                        str(timestamp_col),
                        str(renewable_col),
                        str(unit_mode),
                    )
                    st.session_state["renewable_profile_df"] = normalized_profile
                    st.session_state["renewable_profile_name"] = uploaded_profile_file.name
                    save_profile_assets(
                        uploaded_profile_file,
                        normalized_profile,
                        uploaded_profile_file.name,
                        persist_assets,
                    )
                    flash_message(
                        "success",
                        f"Renewable profile loaded with {len(normalized_profile)} rows and saved name '{uploaded_profile_file.name}'.",
                    )
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not read the renewable profile file: {exc}")

    else:
        saved_profiles = list_saved_profiles()
        saved_profile_options = [p.name for p in saved_profiles]

        if not saved_profile_options:
            st.warning("No saved renewable profiles were found under user_data/renewable_profiles/.")
        else:
            saved_profile_name = st.selectbox("Saved renewable profile", saved_profile_options)
            if st.button("Load saved renewable profile", use_container_width=True):
                selected_path = next(p for p in saved_profiles if p.name == saved_profile_name)
                normalized_profile = load_saved_profile(selected_path)
                st.session_state["renewable_profile_df"] = normalized_profile
                st.session_state["renewable_profile_name"] = saved_profile_name
                flash_message(
                    "success",
                    f"Saved renewable profile '{saved_profile_name}' loaded with {len(normalized_profile)} rows.",
                )
                st.rerun()

    active_profile_df = st.session_state.get("renewable_profile_df")
    active_profile_name = st.session_state.get("renewable_profile_name", active_profile_name)

    if profile_source != "Synthetic default profile":
        if active_profile_df is None:
            st.info("Load a renewable profile above before running the simulation.")
        else:
            st.caption(f"Active profile: {active_profile_name}")
            st.write(
                {
                    "rows": int(len(active_profile_df)),
                    "start": str(active_profile_df["timestamp"].min()),
                    "end": str(active_profile_df["timestamp"].max()),
                    "peak_mw": float(active_profile_df["renewable_power_mw"].max()),
                    "mean_mw": float(active_profile_df["renewable_power_mw"].mean()),
                }
            )
            renewable_peak_power_mw = float(active_profile_df["renewable_power_mw"].max())

    st.subheader("Detected model bundles")

    model_names = registry.get_models_by_library(surrogate_library)
    bundle_df = expected_bundle_status(PROJECT_ROOT, surrogate_library, model_names)

    if bundle_df.empty:
        st.warning("No configured model names were found for the selected surrogate library.")
    else:
        st.dataframe(
            bundle_df[
                [
                    "model_name",
                    "joblib_found",
                    "py_found",
                    "txt_found",
                    "ready_for_runtime",
                    "ready_for_qa",
                    "missing_files",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Upload model ZIP", expanded=False):
        st.caption(
            "Upload one ZIP per model. The archive is flattened into "
            "models/packages/<library>/<model_name>/ so the runtime can find .joblib, .py and .txt directly."
        )

        if model_names:
            target_model_name = st.selectbox(
                "Target model folder",
                model_names,
                key=f"target_model_{surrogate_library}",
            )
            uploaded_model_zip = st.file_uploader(
                "Upload model ZIP",
                type=["zip"],
                accept_multiple_files=False,
                key=f"zip_uploader_{surrogate_library}_{target_model_name}",
            )

            if st.button("Upload selected model ZIP", use_container_width=True):
                if uploaded_model_zip is None:
                    flash_message("error", "Select a ZIP file before pressing upload.")
                    st.rerun()

                try:
                    summary = install_model_zip(
                        PROJECT_ROOT,
                        surrogate_library,
                        target_model_name,
                        uploaded_model_zip,
                        persist_assets,
                    )
                    flash_message(
                        "success",
                        (
                            f"ZIP extracted into {summary['target_dir']} with "
                            f"{summary['written_count']} file(s): {', '.join(summary['written_files'])}"
                        ),
                    )
                    st.rerun()
                except BadZipFile:
                    flash_message("error", "The uploaded file is not a valid ZIP archive.")
                    st.rerun()
                except Exception as exc:
                    flash_message("error", f"Upload failed: {exc}")
                    st.rerun()

    confirm_bundle = st.checkbox(
        "I confirm that the detected model folders and file sets correspond to the intended surrogate library for this run.",
        value=not bundle_df.empty,
    )

if not confirm_bundle:
    st.error("Confirm the detected surrogate library bundle in the sidebar to continue.")
    st.stop()

if profile_source != "Synthetic default profile" and active_profile_df is None:
    st.warning("A renewable profile source was selected, but no active profile is loaded yet.")
    st.stop()

case_payload = {
    "scenario_name": scenario_name,
    "electricity_price_usd_per_kwh": electricity_price_usd_per_kwh,
    "renewable_profile_source": profile_source,
    "renewable_profile_name": active_profile_name,
    "renewable_peak_power_mw": renewable_peak_power_mw,
    "electrolyzer_power_mw": electrolyzer_power_mw,
    "module_count": module_count,
    "storage_enabled": storage_enabled,
    "storage_kg_h2": storage_kg_h2,
    "operating_mode": operating_mode,
    "surrogate_library": surrogate_library,
    "target_h2_kg_per_h": target_h2_kg_per_h,
    "max_h2_feed_kg_per_h": max_h2_feed_kg_per_h,
}
case_signature = build_case_signature(case_payload)
reset_results_for_new_case(case_signature)

case = runner.build_case(
    scenario_name=scenario_name,
    electrolyzer_power_mw=float(electrolyzer_power_mw),
    module_count=int(module_count),
    storage_enabled=bool(storage_enabled),
    storage_kg_h2=float(storage_kg_h2),
    operating_mode=str(operating_mode),
    surrogate_library=str(surrogate_library),
    target_h2_kg_per_h=float(target_h2_kg_per_h),
    max_h2_feed_kg_per_h=float(max_h2_feed_kg_per_h),
    renewable_peak_power_mw=float(renewable_peak_power_mw),
    renewable_profile_df=active_profile_df,
    electricity_price_usd_per_kwh=float(electricity_price_usd_per_kwh),
)

action_col1, action_col2, action_col3, action_col4 = st.columns(4)

with action_col1:
    if st.button("Run annual simulation", use_container_width=True):
        with st.spinner("Running annual simulation..."):
            st.session_state["simulation"] = run_simulation(case)

with action_col2:
    if st.button("Run optimization", use_container_width=True):
        with st.spinner("Running grid optimization..."):
            st.session_state["optimization"] = run_optimization(case)

with action_col3:
    if st.button("Run sensitivities", use_container_width=True):
        with st.spinner("Running sensitivity analysis..."):
            st.session_state["sensitivities"] = run_sensitivities(case)

with action_col4:
    if st.button("Run all", use_container_width=True):
        with st.spinner("Running full workflow..."):
            st.session_state["simulation"] = run_simulation(case)
            st.session_state["optimization"] = run_optimization(case)
            st.session_state["sensitivities"] = run_sensitivities(case)

simulation = st.session_state.get("simulation")
optimization = st.session_state.get("optimization")
sensitivities = st.session_state.get("sensitivities")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Inputs", "Annual Simulation", "Techno-Economic Optimum", "Sensitivities"]
)

with tab1:
    c1, c2, c3 = st.columns([1.1, 1.2, 1.1])

    with c1:
        st.subheader("Case definition")
        st.json(
            {
                "scenario": scenario_name,
                "electricity_price_usd_per_kwh": electricity_price_usd_per_kwh,
                "renewable_profile_source": profile_source,
                "renewable_profile_name": active_profile_name,
                "renewable_peak_power_mw": renewable_peak_power_mw,
                "electrolyzer_power_mw": electrolyzer_power_mw,
                "module_count": int(module_count),
                "storage_enabled": storage_enabled,
                "storage_kg_h2": storage_kg_h2,
                "operating_mode": operating_mode,
                "surrogate_library": surrogate_library,
                "target_h2_kg_per_h": target_h2_kg_per_h,
                "max_h2_feed_kg_per_h": max_h2_feed_kg_per_h,
            }
        )

    with c2:
        st.subheader("Model bundle checklist")
        st.dataframe(bundle_df, use_container_width=True, hide_index=True)

    with c3:
        st.subheader("Renewable profile summary")
        if active_profile_df is None:
            st.info("Using synthetic renewable profile generated from the selected renewable peak power.")
        else:
            st.write(
                {
                    "rows": int(len(active_profile_df)),
                    "start": str(active_profile_df["timestamp"].min()),
                    "end": str(active_profile_df["timestamp"].max()),
                    "peak_mw": float(active_profile_df["renewable_power_mw"].max()),
                    "mean_mw": float(active_profile_df["renewable_power_mw"].mean()),
                    "annual_energy_gwh_equiv": float(active_profile_df["renewable_power_mw"].sum() / 1000.0),
                }
            )
            st.dataframe(active_profile_df.head(24), use_container_width=True)

with tab2:
    if simulation is None:
        st.info("Press 'Run annual simulation' to generate results.")
    else:
        for warning in simulation.warnings:
            st.warning(warning)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Annual MeOH [t/y]", f"{simulation.kpis['annual_methanol_t']:,.0f}")
        m2.metric(
            "Electrolyzer FLH [h/y]",
            f"{simulation.kpis['electrolyzer_full_load_hours_h']:,.0f}",
        )
        m3.metric(
            "PtMeOH utilization [-]",
            f"{simulation.kpis['ptmeoh_utilization_factor']:.2f}",
        )
        m4.metric(
            "Renewable utilization [-]",
            f"{simulation.kpis['renewable_utilization_fraction']:.2f}",
        )
        m5.metric(
            "LCOMeOH [USD/t]",
            f"{simulation.kpis['lcomeoh_usd_per_t_meoh']:,.1f}",
        )
        m6.metric("NPV [USD]", f"{simulation.kpis['npv_usd']:,.0f}")

        ts = simulation.time_series
        st.plotly_chart(
            line_profile(
                ts.iloc[:336],
                ["renewable_power_mw", "power_to_electrolyzer_mw"],
                "Renewable and electrolyzer power — first two weeks",
                "Power [MW]",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            line_profile(
                ts.iloc[:336],
                ["h2_produced_kg_per_h", "h2_to_ptmeoh_kg_per_h", "tank_soc_kg_h2"],
                "Hydrogen production, PtMeOH feed, and tank state of charge — first two weeks",
                "H2 / SOC [kg or kg/h]",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            line_profile(
                ts.iloc[:336],
                ["methanol_t_per_h", "unmet_h2_kg_per_h"],
                "Methanol production and unmet H2 — first two weeks",
                "Production / unmet H2",
            ),
            use_container_width=True,
        )
        st.subheader("Traceable hourly results preview")
        st.dataframe(ts.head(48), use_container_width=True)

with tab3:
    if optimization is None:
        st.info("Press 'Run optimization' to generate the design ranking.")
    else:
        left, right = st.columns([1.0, 1.6])

        with left:
            st.subheader("Recommended configuration")
            st.dataframe(
                optimization.best_row.to_frame(name="value"),
                use_container_width=True,
            )
            st.success(
                "Recommended design balances CAPEX, renewable utilization, PtMeOH stability, and storage buffering under the chosen scenario."
            )
            st.info(
                "CAPEX vs stability: larger electrolyzers and larger tanks reduce unmet H2 risk, but they can increase capital intensity and leave more underutilized capacity if the renewable profile is not strong enough."
            )

        with right:
            st.plotly_chart(
                heatmap(optimization.results, z_col="lcomeoh_usd_per_t_meoh"),
                use_container_width=True,
            )

        st.subheader("Ranked shortlist")
        st.dataframe(
            optimization.results.sort_values(
                ["lcomeoh_usd_per_t_meoh", "warning_count"]
            ).head(12),
            use_container_width=True,
        )

with tab4:
    if sensitivities is None:
        st.info("Press 'Run sensitivities' to generate the tornado view.")
    else:
        st.plotly_chart(tornado(sensitivities), use_container_width=True)
        st.dataframe(sensitivities, use_container_width=True)
