from __future__ import annotations

from io import BytesIO
import json
import logging
from pathlib import Path
import re
from typing import Callable, Optional
from zipfile import BadZipFile, ZipFile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from application.case_runner import CaseRunner
from infrastructure.model_registry import ModelRegistry
from presentation.plotting import heatmap, line_profile, tornado

PROJECT_ROOT = Path(__file__).resolve().parent
PERSIST_ROOT = PROJECT_ROOT / "user_data"
PROFILE_STORE = PERSIST_ROOT / "renewable_profiles"
MODEL_ARCHIVE_STORE = PERSIST_ROOT / "model_archives"

SCENARIO_PRICE_DEFAULTS_USD_PER_KWH = {
    "optimistic": 35.0 / 1000.0,
    "moderate": 55.0 / 1000.0,
    "pessimistic": 80.0 / 1000.0,
}

ProgressCallback = Optional[Callable[[str, int, int], None]]

runner = CaseRunner(PROJECT_ROOT)
registry = ModelRegistry(PROJECT_ROOT)
st.sidebar.code(
    f"catalog_path={registry.catalog_path}\n"
    f"catalog_exists={registry.catalog_path.exists()}\n"
    f"model_registry_file={ModelRegistry.__module__}\n"
    f"libraries={registry.get_library_names()}\n"
    f"models_const={registry.get_models_by_library('variable_h2_constant_co2')[:5]}",
    language="text",
)

st.set_page_config(page_title="PtMeOH Sizing Tool V2", layout="wide")
st.title("PtMeOH Plant Sizing Tool — Version 2")
st.caption(
    "Annual deterministic simulator with surrogate governance, renewable-profile diagnostics, "
    "pre-run consistency checks, execution monitoring, and expanded operational observability."
)


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text).strip())
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


def get_scenario_price_default(scenario_name: str) -> float:
    runner_cfg = getattr(runner, "scenario_config", None)
    if isinstance(runner_cfg, dict):
        try:
            return float(runner_cfg[scenario_name]["electricity_price_usd_per_mwh"]) / 1000.0
        except Exception:
            pass
    return SCENARIO_PRICE_DEFAULTS_USD_PER_KWH.get(
        scenario_name, SCENARIO_PRICE_DEFAULTS_USD_PER_KWH["moderate"]
    )


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
    lowered = [str(c).lower() for c in columns]
    for kw in keywords:
        for idx, col in enumerate(lowered):
            if kw in col:
                return idx
    return 0


def infer_profile_meta_from_normalized(
    df: pd.DataFrame,
    source_name: str,
    source_kind: str,
    raw_rows: int | None = None,
    unit_mode: str | None = None,
    expanded_to_hourly: bool | None = None,
) -> dict:
    if df is None or df.empty:
        return {
            "source_name": source_name,
            "source_kind": source_kind,
            "raw_rows": raw_rows or 0,
            "normalized_rows": 0,
            "median_step_h": None,
            "looks_daily": False,
            "expanded_to_hourly": bool(expanded_to_hourly),
            "unit_mode": unit_mode,
        }

    diffs_h = (
        df["timestamp"].diff().dropna().dt.total_seconds().div(3600.0)
        if len(df) > 1
        else pd.Series(dtype=float)
    )
    median_step_h = float(diffs_h.median()) if not diffs_h.empty else 1.0
    looks_daily = median_step_h >= 23.0

    return {
        "source_name": source_name,
        "source_kind": source_kind,
        "raw_rows": int(raw_rows or len(df)),
        "normalized_rows": int(len(df)),
        "median_step_h": median_step_h,
        "looks_daily": looks_daily,
        "expanded_to_hourly": bool(expanded_to_hourly),
        "unit_mode": unit_mode,
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
    }


def normalize_renewable_profile(
    raw_profile_df: pd.DataFrame,
    timestamp_col: str,
    renewable_col: str,
    unit_mode: str,
    expand_daily_to_hourly: bool = True,
) -> tuple[pd.DataFrame, dict]:
    df = raw_profile_df[[timestamp_col, renewable_col]].copy()
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

    expanded_to_hourly = False
    if looks_daily and expand_daily_to_hourly:
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
        expanded_to_hourly = True
    else:
        out = df[["timestamp", "renewable_power_mw"]].copy()

    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    meta = {
        "source_name": "uploaded_profile",
        "source_kind": "uploaded",
        "raw_rows": int(len(df)),
        "normalized_rows": int(len(out)),
        "median_step_h": median_step_h,
        "looks_daily": bool(looks_daily),
        "expanded_to_hourly": bool(expanded_to_hourly),
        "unit_mode": unit_mode,
        "start": str(out["timestamp"].min()),
        "end": str(out["timestamp"].max()),
    }
    return out, meta


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


class StreamlitTelemetryHandler(logging.Handler):
    def __init__(self, emit_fn):
        super().__init__(level=logging.INFO)
        self.emit_fn = emit_fn
        self._streamlit_telemetry = True

    def emit(self, record):
        try:
            self.emit_fn(self.format(record))
        except Exception:
            pass


def ensure_runtime_feedback_state() -> None:
    if "telemetry_lines" not in st.session_state:
        st.session_state["telemetry_lines"] = []


def clear_telemetry() -> None:
    st.session_state["telemetry_lines"] = []


def append_telemetry_line(message: str, max_lines: int = 250) -> None:
    ensure_runtime_feedback_state()
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
    line = f"{timestamp} | {message}"
    lines = st.session_state["telemetry_lines"]
    if not lines or lines[-1] != line:
        lines.append(line)
    if len(lines) > max_lines:
        st.session_state["telemetry_lines"] = lines[-max_lines:]


def get_telemetry_text(last_n: int = 40) -> str:
    ensure_runtime_feedback_state()
    lines = st.session_state.get("telemetry_lines", [])
    if not lines:
        return "No telemetry messages yet."
    return "\n".join(lines[-last_n:])


def attach_streamlit_telemetry_handler() -> None:
    ensure_runtime_feedback_state()

    candidate_loggers = []
    for logger_obj in [
        getattr(runner, "logger", None),
        getattr(getattr(runner, "engine", None), "logger", None),
        getattr(getattr(runner, "optimizer", None), "logger", None),
        getattr(getattr(runner, "sensitivity", None), "logger", None),
        logging.getLogger("ptmeoh_tool"),
        logging.getLogger("ptmeoh_tool.simulation"),
        logging.getLogger("ptmeoh_tool.optimization"),
        logging.getLogger("ptmeoh_tool.sensitivity"),
    ]:
        if logger_obj is not None and logger_obj not in candidate_loggers:
            candidate_loggers.append(logger_obj)

    for logger_obj in candidate_loggers:
        if any(getattr(h, "_streamlit_telemetry", False) for h in logger_obj.handlers):
            continue
        handler = StreamlitTelemetryHandler(append_telemetry_line)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger_obj.addHandler(handler)
        logger_obj.setLevel(logging.INFO)


def format_progress_label(stage: str, current: int, total: int) -> tuple[float, str]:
    safe_total = max(int(total), 1)
    safe_current = max(0, min(int(current), safe_total))
    pct = 100.0 * safe_current / safe_total
    stage_key = str(stage).lower().strip()

    if stage_key == "simulation":
        hours_done = safe_current
        hours_total = safe_total
        days_done = hours_done / 24.0
        days_total = hours_total / 24.0
        label = (
            f"Annual simulation | {pct:5.1f}% | "
            f"{hours_done:,}/{hours_total:,} simulated h | "
            f"{days_done:,.2f}/{days_total:,.2f} days"
        )
        return pct, label

    if stage_key == "optimization":
        label = f"Optimization | {pct:5.1f}% | {safe_current:,}/{safe_total:,} cases evaluated"
        return pct, label

    if stage_key == "sensitivity":
        label = f"Sensitivity | {pct:5.1f}% | {safe_current:,}/{safe_total:,} perturbations"
        return pct, label

    label = f"{stage} | {pct:5.1f}% | {safe_current:,}/{safe_total:,}"
    return pct, label


def run_simulation(case, progress_callback: ProgressCallback = None):
    if hasattr(runner, "run_simulation"):
        try:
            return runner.run_simulation(case, progress_callback=progress_callback)
        except TypeError:
            pass

    try:
        return runner.engine.run(case, progress_callback=progress_callback)
    except TypeError:
        append_telemetry_line(
            "WARNING | The simulation engine does not accept progress_callback; it will run without fine-grained backend progress."
        )
        return runner.engine.run(case)


def run_optimization(case, progress_callback: ProgressCallback = None):
    if hasattr(runner, "run_optimization"):
        try:
            return runner.run_optimization(case, progress_callback=progress_callback)
        except TypeError:
            pass

    try:
        return runner.optimizer.run(case, progress_callback=progress_callback)
    except TypeError:
        append_telemetry_line(
            "WARNING | The optimizer does not accept progress_callback; it will run without fine-grained backend progress."
        )
        return runner.optimizer.run(case)


def run_sensitivities(case, progress_callback: ProgressCallback = None):
    if hasattr(runner, "run_sensitivity"):
        try:
            return runner.run_sensitivity(case, progress_callback=progress_callback)
        except TypeError:
            pass

    if hasattr(runner, "sensitivity"):
        try:
            return runner.sensitivity.run(case, progress_callback=progress_callback)
        except TypeError:
            return runner.sensitivity.run(case)

    raise AttributeError("No sensitivity runner was found.")


def build_profile_diagnostics(profile_df: pd.DataFrame, electrolyzer_power_mw: float, min_load_fraction: float) -> dict:
    peak_mw = float(profile_df["renewable_power_mw"].max()) if not profile_df.empty else 0.0
    mean_mw = float(profile_df["renewable_power_mw"].mean()) if not profile_df.empty else 0.0
    annual_energy_gwh_equiv = float(profile_df["renewable_power_mw"].sum() / 1000.0) if not profile_df.empty else 0.0
    min_load_mw = float(electrolyzer_power_mw * min_load_fraction)
    below_min_mask = (
        profile_df["renewable_power_mw"] < min_load_mw if not profile_df.empty else pd.Series(dtype=bool)
    )
    below_min_fraction = float(below_min_mask.mean()) if len(below_min_mask) > 0 else 0.0
    zero_power_fraction = float((profile_df["renewable_power_mw"] <= 1e-9).mean()) if not profile_df.empty else 0.0
    severe_scale_mismatch = peak_mw < 0.20 * float(electrolyzer_power_mw)
    likely_non_operational = below_min_fraction > 0.50

    return {
        "peak_mw": peak_mw,
        "mean_mw": mean_mw,
        "annual_energy_gwh_equiv": annual_energy_gwh_equiv,
        "electrolyzer_nominal_power_mw": float(electrolyzer_power_mw),
        "electrolyzer_min_load_fraction": float(min_load_fraction),
        "electrolyzer_min_load_mw": min_load_mw,
        "below_min_load_fraction": below_min_fraction,
        "zero_power_fraction": zero_power_fraction,
        "severe_scale_mismatch": severe_scale_mismatch,
        "likely_non_operational": likely_non_operational,
    }


def collect_bundle_issues(filtered_catalog: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if filtered_catalog is None or filtered_catalog.empty:
        issues.append("No model bundles were detected for the selected surrogate library.")
        return issues

    if "ready_for_runtime" in filtered_catalog.columns:
        incomplete = filtered_catalog[filtered_catalog["ready_for_runtime"] != True].copy()
        for _, row in incomplete.iterrows():
            model_name = row.get("model_name", "unknown_model")
            missing = row.get("missing_files", "")
            if pd.isna(missing):
                missing = ""
            issues.append(f"{model_name}: missing runtime artifacts -> {missing or 'unspecified'}")

    if "model_name" in filtered_catalog.columns and "ready_for_runtime" in filtered_catalog.columns:
        prod_mask = filtered_catalog["model_name"].astype(str).eq("Model_Prod_MeOH")
        if prod_mask.any():
            prod_ready = bool(filtered_catalog.loc[prod_mask, "ready_for_runtime"].iloc[0])
            if not prod_ready:
                issues.append(
                    "Model_Prod_MeOH is incomplete; methanol output may rely on fallback estimation rather than the dedicated surrogate."
                )
        else:
            issues.append("Model_Prod_MeOH is not listed in the selected surrogate library catalog.")

    return issues


def build_pre_run_messages(case, profile_meta: dict | None, profile_diag: dict, filtered_catalog: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    infos: list[str] = []

    if case.ptmeoh.target_h2_feed_kg_per_h > case.ptmeoh.max_h2_feed_kg_per_h:
        errors.append(
            "Target H2 feed is greater than PtMeOH maximum intake. Reduce the target or increase the maximum intake before running."
        )

    if profile_meta and profile_meta.get("looks_daily") and profile_meta.get("expanded_to_hourly"):
        infos.append(
            f"Loaded renewable profile was detected as daily data and expanded from {profile_meta.get('raw_rows')} rows to {profile_meta.get('normalized_rows')} hourly rows."
        )

    if profile_diag["severe_scale_mismatch"]:
        warnings.append(
            f"Renewable peak power ({profile_diag['peak_mw']:.3f} MW) is far below electrolyzer nominal power ({profile_diag['electrolyzer_nominal_power_mw']:.3f} MW)."
        )

    if profile_diag["likely_non_operational"]:
        warnings.append(
            f"Renewable availability is below electrolyzer minimum load ({profile_diag['electrolyzer_min_load_mw']:.3f} MW) during "
            f"{100.0 * profile_diag['below_min_load_fraction']:.1f}% of the profile, so the electrolyzer may remain off for much of the year."
        )

    if profile_diag["mean_mw"] < 0.10 * profile_diag["electrolyzer_nominal_power_mw"]:
        warnings.append(
            f"Average renewable power ({profile_diag['mean_mw']:.3f} MW) is very low relative to electrolyzer nominal power "
            f"({profile_diag['electrolyzer_nominal_power_mw']:.3f} MW)."
        )

    bundle_issues = collect_bundle_issues(filtered_catalog)
    warnings.extend(bundle_issues)

    return errors, warnings, infos


def build_daily_mean_chart(ts: pd.DataFrame, columns: list[str], title: str, y_title: str):
    if ts.empty:
        return go.Figure()
    tmp = ts.copy()
    tmp["date"] = pd.to_datetime(tmp["timestamp"]).dt.date
    agg = tmp.groupby("date", as_index=False)[columns].mean(numeric_only=True)
    fig = go.Figure()
    palette = ["#0b7285", "#2f9e44", "#4dabf7", "#6b7f86", "#8ca6ad"]
    for idx, col in enumerate(columns):
        if col in agg.columns:
            fig.add_trace(
                go.Scatter(
                    x=agg["date"],
                    y=agg[col],
                    mode="lines",
                    name=col,
                    line=dict(color=palette[idx % len(palette)], width=2),
                )
            )
    fig.update_layout(
        title=title,
        paper_bgcolor="#f4f8f8",
        plot_bgcolor="white",
        yaxis_title=y_title,
        legend_orientation="h",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_diagnostic_event_chart(ts: pd.DataFrame):
    if ts.empty:
        return go.Figure()
    tmp = ts.copy()
    tmp["date"] = pd.to_datetime(tmp["timestamp"]).dt.date
    daily = tmp.groupby("date", as_index=False).agg(
        unmet_h2_kg_per_d=("unmet_h2_kg_per_h", "sum"),
        curtailed_power_mwh_per_d=("curtailed_power_mw", "sum"),
        out_of_domain_hours=("surrogate_all_models_in_domain", lambda s: int((1 - s).sum())),
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily["date"],
            y=daily["unmet_h2_kg_per_d"],
            name="unmet_h2_kg_per_d",
            marker_color="#a12c7b",
        )
    )
    fig.add_trace(
        go.Bar(
            x=daily["date"],
            y=daily["curtailed_power_mwh_per_d"],
            name="curtailed_power_mwh_per_d",
            marker_color="#da7101",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Daily diagnostic events — unmet H2 and renewable curtailment",
        paper_bgcolor="#f4f8f8",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_balance_bar_chart(ts: pd.DataFrame):
    if ts.empty:
        return go.Figure()

    records = [
        {"metric": "Renewable available [MWh/y]", "value": float(ts["renewable_power_mw"].sum())},
        {"metric": "Renewable used [MWh/y]", "value": float(ts["renewable_used_mw"].sum())},
        {"metric": "Electrolyzer power [MWh/y]", "value": float(ts["power_to_electrolyzer_mw"].sum())},
        {"metric": "Downstream aux power [MWh/y]", "value": float(ts["downstream_aux_power_mw"].sum())},
        {"metric": "Curtailed power [MWh/y]", "value": float(ts["curtailed_power_mw"].sum())},
        {"metric": "H2 produced [kg/y]", "value": float(ts["h2_produced_kg_per_h"].sum())},
        {"metric": "H2 to PtMeOH [kg/y]", "value": float(ts["h2_to_ptmeoh_kg_per_h"].sum())},
        {"metric": "Unmet H2 [kg/y]", "value": float(ts["unmet_h2_kg_per_h"].sum())},
    ]
    df = pd.DataFrame(records)
    fig = px.bar(
        df,
        x="value",
        y="metric",
        orientation="h",
        title="Annual balance overview",
        color_discrete_sequence=["#0b7285"],
    )
    fig.update_layout(
        paper_bgcolor="#f4f8f8",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    last_error: Exception | None = None
    for engine in ["openpyxl", "xlsxwriter"]:
        output = BytesIO()
        try:
            with pd.ExcelWriter(output, engine=engine) as writer:
                df.to_excel(writer, index=False, sheet_name=sheet_name)
            output.seek(0)
            return output.getvalue()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Excel export failed because no supported writer engine is available. Last error: {last_error}"
    )


def build_kpi_export_dataframe(kpis: dict) -> pd.DataFrame:
    main_kpis = [
        ("annual_methanol_t", kpis.get("annual_methanol_t")),
        ("electrolyzer_full_load_hours_h", kpis.get("electrolyzer_full_load_hours_h")),
        ("ptmeoh_utilization_factor", kpis.get("ptmeoh_utilization_factor")),
        ("renewable_utilization_fraction", kpis.get("renewable_utilization_fraction")),
        ("lcomeoh_usd_per_t_meoh", kpis.get("lcomeoh_usd_per_t_meoh")),
        ("npv_usd", kpis.get("npv_usd")),
    ]

    diagnostic_kpis = [
        ("surrogate_out_of_domain_fraction", kpis.get("surrogate_out_of_domain_fraction")),
        ("curtailment_fraction", kpis.get("curtailment_fraction")),
        ("h2_not_supplied_t", kpis.get("h2_not_supplied_t")),
        ("runtime_models_fraction", kpis.get("runtime_models_fraction")),
        ("tank_empty_hours", kpis.get("tank_empty_hours")),
        ("tank_full_hours", kpis.get("tank_full_hours")),
        ("curtailed_hours", kpis.get("curtailed_hours")),
        ("annual_total_electricity_mwh", kpis.get("annual_total_electricity_mwh")),
    ]

    rows = []
    for metric_name, value in main_kpis:
        rows.append(
            {
                "group": "main_kpi",
                "metric": metric_name,
                "value": value,
            }
        )

    for metric_name, value in diagnostic_kpis:
        rows.append(
            {
                "group": "diagnostic_kpi",
                "metric": metric_name,
                "value": value,
            }
        )

    return pd.DataFrame(rows)


def build_case_inputs_export_dataframe(
    case_payload: dict,
    effective_profile_meta: dict,
    profile_diag: dict,
) -> pd.DataFrame:
    rows = []

    for key, value in case_payload.items():
        rows.append(
            {
                "group": "case_input",
                "parameter": key,
                "value": value,
            }
        )

    for key, value in effective_profile_meta.items():
        rows.append(
            {
                "group": "profile_meta",
                "parameter": key,
                "value": value,
            }
        )

    for key, value in profile_diag.items():
        rows.append(
            {
                "group": "profile_diagnostic",
                "parameter": key,
                "value": value,
            }
        )

    return pd.DataFrame(rows)


render_flash_message()

active_profile_df = st.session_state.get("renewable_profile_df")
active_profile_meta = st.session_state.get("renewable_profile_meta")
active_profile_name = st.session_state.get("renewable_profile_name", "default_synthetic_profile")

with st.sidebar:
    st.header("Case inputs")

    persist_assets_value = st.session_state.get("persist_assets", True)

    library_names = registry.get_library_names() or [
        "variable_h2_constant_co2",
        "variable_h2_variable_co2",
    ]
    

    scenario_name = st.selectbox(
        "Techno-economic scenario",
        ["optimistic", "moderate", "pessimistic"],
        index=1,
    )
    
    default_electricity_price_usd_per_kwh = get_scenario_price_default(scenario_name)
    electricity_price_usd_per_kwh = st.number_input(
        "Electricity price [USD/kWh]",
        min_value=0.0,
        value=float(default_electricity_price_usd_per_kwh),
        step=0.001,
        format="%.4f",
        help="This value overrides the electricity-price assumption of the selected scenario for the current case.",
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
        library_names,
        index=0,
    )

    target_h2_kg_per_h = st.number_input(
        "Target H2 feed to PtMeOH [kg/h]",
        min_value=0.0,
        value=1850.0,
        step=10.0,
        format="%.3f",
        help=(
            "Hourly H2 delivery target for the methanol train. If renewable production is insufficient, "
            "the tank discharges first and any remaining deficit becomes unmet H2."
        ),
    )

    max_h2_feed_kg_per_h = st.number_input(
        "Maximum PtMeOH H2 intake [kg/h]",
        min_value=0.0,
        value=2200.0,
        step=10.0,
        format="%.3f",
        help="Hard upper bound on how much H2 the downstream PtMeOH train can physically absorb.",
    )

    with st.expander("Explain downstream H2 variables", expanded=False):
        st.markdown(
            """
- **Target H2 feed to PtMeOH [kg/h]**: hourly H2 delivery target for the methanol train.
- **Maximum PtMeOH H2 intake [kg/h]**: downstream processing ceiling that caps actual H2 intake.
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
        st.session_state["renewable_profile_df"] = None
        st.session_state["renewable_profile_meta"] = None
        st.session_state["renewable_profile_name"] = "default_synthetic_profile"
        active_profile_df = None
        active_profile_meta = None
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
                expand_daily_to_hourly = st.checkbox(
                    "Expand daily profile to hourly resolution when daily data is detected",
                    value=True,
                )

                st.dataframe(raw_profile_df.head(12), use_container_width=True)

                if st.button("Load this renewable profile", use_container_width=True):
                    normalized_profile, profile_meta = normalize_renewable_profile(
                        raw_profile_df=raw_profile_df,
                        timestamp_col=str(timestamp_col),
                        renewable_col=str(renewable_col),
                        unit_mode=str(unit_mode),
                        expand_daily_to_hourly=bool(expand_daily_to_hourly),
                    )
                    profile_meta["source_name"] = uploaded_profile_file.name
                    profile_meta["source_kind"] = "uploaded"

                    st.session_state["renewable_profile_df"] = normalized_profile
                    st.session_state["renewable_profile_meta"] = profile_meta
                    st.session_state["renewable_profile_name"] = uploaded_profile_file.name

                    save_profile_assets(
                        uploaded_profile_file,
                        normalized_profile,
                        uploaded_profile_file.name,
                        persist_assets_value,
                    )
                    flash_message(
                        "success",
                        f"Renewable profile loaded with {len(normalized_profile)} rows from '{uploaded_profile_file.name}'.",
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
                profile_meta = infer_profile_meta_from_normalized(
                    df=normalized_profile,
                    source_name=saved_profile_name,
                    source_kind="saved",
                    raw_rows=len(normalized_profile),
                    expanded_to_hourly=False,
                )
                st.session_state["renewable_profile_df"] = normalized_profile
                st.session_state["renewable_profile_meta"] = profile_meta
                st.session_state["renewable_profile_name"] = saved_profile_name
                flash_message(
                    "success",
                    f"Saved renewable profile '{saved_profile_name}' loaded with {len(normalized_profile)} rows.",
                )
                st.rerun()

    active_profile_df = st.session_state.get("renewable_profile_df")
    active_profile_meta = st.session_state.get("renewable_profile_meta")
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

    catalog_df = registry.discover_packages()
    catalog_has_library_col = (
            isinstance(catalog_df, pd.DataFrame)
            and not catalog_df.empty
            and "library" in catalog_df.columns
        )

    filtered_catalog = (
            catalog_df.loc[catalog_df["library"].astype(str) == str(surrogate_library)].copy()
            if catalog_has_library_col
            else pd.DataFrame()
        )

    model_names = registry.get_models_by_library(str(surrogate_library)) or []

    with st.expander("Upload surrogate model", expanded=False):
            if not isinstance(catalog_df, pd.DataFrame) or catalog_df.empty:
                st.warning("Model registry returned no package catalog.")
            elif "library" not in catalog_df.columns:
                st.error("Model registry catalog does not contain the required 'library' column.")
            elif filtered_catalog.empty:
                st.warning(
                    f"No configured model names were found for the selected surrogate library: "
                    f"{surrogate_library}"
                )
            else:
                preferred_cols = [
                    "model_name",
                    "joblib_found",
                    "py_found",
                    "txt_found",
                    "ready_for_runtime",
                    "ready_for_qa",
                    "missing_files",
                ]
                present_cols = [c for c in preferred_cols if c in filtered_catalog.columns]
                st.dataframe(
                    filtered_catalog[present_cols],
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "Upload one ZIP per model. The archive is flattened into "
                "models/packages/<library>/<model>/ so the runtime can find "
                ".joblib, .py and .txt directly."
            )

            if model_names:
                target_model_name = st.selectbox(
                    "Target model bundle",
                    model_names,
                    key=f"target_model_{surrogate_library}",
                )

                uploaded_model_zip = st.file_uploader(
                    "Upload model ZIP",
                    type=["zip"],
                    accept_multiple_files=False,
                    key=f"zip_uploader_{surrogate_library}_{target_model_name}",
                    help=(
                        "Select the ZIP file for the chosen bundle. "
                        "The archive will be extracted into the target model folder."
                    ),
                )

                if st.button(
                    "Assign selected model ZIP to model bundle",
                    use_container_width=True,
                    key=f"assign_model_zip_{surrogate_library}_{target_model_name}",
                ):
                    if uploaded_model_zip is None:
                        flash_message("error", "Select a ZIP file before pressing upload.")
                        st.rerun()

                    try:
                        summary = install_model_zip(
                            PROJECT_ROOT,
                            str(surrogate_library),
                            str(target_model_name),
                            uploaded_model_zip,
                            st.session_state.get("persist_assets", True),
                        )
                        flash_message(
                            "success",
                            f"ZIP extracted into {summary['target_dir']} with "
                            f"{summary['written_count']} file(s): "
                            f"{', '.join(summary['written_files'])}",
                        )
                        st.rerun()
                    except BadZipFile:
                        flash_message("error", "The uploaded file is not a valid ZIP archive.")
                        st.rerun()
                    except Exception as exc:
                        flash_message("error", f"Upload failed: {exc}")
                        st.rerun()
            else:
                st.info(
                    f"No model names are currently registered for the selected library: "
                    f"{surrogate_library}"
                )

    st.toggle(
    "Save uploaded model ZIPs and renewable profile for future iterations",
    value=st.session_state.get("persist_assets", True),
    key="persist_assets",
    help="When enabled, uploaded model archives and normalized renewable profiles are written to disk under user_data/.",
        )

    confirm_bundle = st.checkbox(
    "I confirm that the detected model folders and file sets correspond to the intended surrogate library for this run.",
        value=not filtered_catalog.empty,)

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
    "module_count": int(module_count),
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

effective_profile_df = case.renewable_profile.copy()
effective_profile_meta = active_profile_meta
if effective_profile_meta is None:
    effective_profile_meta = infer_profile_meta_from_normalized(
        df=effective_profile_df,
        source_name="default_synthetic_profile",
        source_kind="synthetic",
        raw_rows=len(effective_profile_df),
        expanded_to_hourly=False,
    )

profile_diag = build_profile_diagnostics(
    profile_df=effective_profile_df,
    electrolyzer_power_mw=float(case.electrolyzer.nominal_power_mw),
    min_load_fraction=float(case.electrolyzer.min_load_fraction),
)

pre_errors, pre_warnings, pre_infos = build_pre_run_messages(
    case=case,
    profile_meta=effective_profile_meta,
    profile_diag=profile_diag,
    filtered_catalog=filtered_catalog,
)

run_disabled = len(pre_errors) > 0

attach_streamlit_telemetry_handler()
ensure_runtime_feedback_state()

st.subheader("Execution monitor")
progress_bar = st.progress(0.0, text="Idle | 0.0%")
progress_caption = st.empty()

with st.expander("Real-time telemetry", expanded=True):
    telemetry_placeholder = st.empty()
    telemetry_placeholder.code(get_telemetry_text(), language="text")


def refresh_runtime_panels() -> None:
    telemetry_placeholder.code(get_telemetry_text(), language="text")


def start_runtime_feedback(title: str) -> None:
    clear_telemetry()
    append_telemetry_line(f"UI | {title}")
    progress_bar.progress(0.0, text=f"{title} | 0.0%")
    progress_caption.info(title)
    refresh_runtime_panels()


def finish_runtime_feedback(title: str) -> None:
    append_telemetry_line(f"UI | {title} completed")
    progress_bar.progress(1.0, text=f"{title} | 100.0%")
    progress_caption.success(title)
    refresh_runtime_panels()


def fail_runtime_feedback(title: str, exc: Exception) -> None:
    append_telemetry_line(f"ERROR | {title} | {exc}")
    progress_caption.error(f"{title}: {exc}")
    refresh_runtime_panels()


def ui_progress(stage: str, current: int, total: int) -> None:
    pct, label = format_progress_label(stage, current, total)
    progress_bar.progress(min(max(pct / 100.0, 0.0), 1.0), text=label)
    progress_caption.caption(label)
    append_telemetry_line(f"PROGRESS | {label}")
    refresh_runtime_panels()


action_col1, action_col2, action_col3, action_col4 = st.columns(4)

with action_col1:
    if st.button("Run annual simulation", use_container_width=True, disabled=run_disabled):
        try:
            start_runtime_feedback("Annual simulation started")
            append_telemetry_line("UI | Building annual engine call")
            st.session_state["simulation"] = run_simulation(
                case,
                progress_callback=ui_progress,
            )
            finish_runtime_feedback("Annual simulation completed")
        except Exception as exc:
            fail_runtime_feedback("Annual simulation failed", exc)
            st.exception(exc)

with action_col2:
    if st.button("Run optimization", use_container_width=True, disabled=run_disabled):
        try:
            start_runtime_feedback("Optimization started")
            append_telemetry_line("UI | Launching grid search")
            st.session_state["optimization"] = run_optimization(
                case,
                progress_callback=ui_progress,
            )
            finish_runtime_feedback("Optimization completed")
        except Exception as exc:
            fail_runtime_feedback("Optimization failed", exc)
            st.exception(exc)

with action_col3:
    if st.button("Run sensitivities", use_container_width=True, disabled=run_disabled):
        try:
            start_runtime_feedback("Sensitivity analysis started")
            append_telemetry_line("UI | Launching sensitivity analysis")
            st.session_state["sensitivities"] = run_sensitivities(
                case,
                progress_callback=ui_progress,
            )
            finish_runtime_feedback("Sensitivity analysis completed")
        except Exception as exc:
            fail_runtime_feedback("Sensitivity analysis failed", exc)
            st.exception(exc)

with action_col4:
    if st.button("Run all", use_container_width=True, disabled=run_disabled):
        try:
            start_runtime_feedback("Run all started")
            append_telemetry_line("UI | Running annual simulation")
            st.session_state["simulation"] = run_simulation(
                case,
                progress_callback=ui_progress,
            )

            append_telemetry_line("UI | Running optimization")
            st.session_state["optimization"] = run_optimization(
                case,
                progress_callback=ui_progress,
            )

            append_telemetry_line("UI | Running sensitivities")
            st.session_state["sensitivities"] = run_sensitivities(
                case,
                progress_callback=ui_progress,
            )

            finish_runtime_feedback("Run all completed")
        except Exception as exc:
            fail_runtime_feedback("Run all failed", exc)
            st.exception(exc)

simulation = st.session_state.get("simulation")
optimization = st.session_state.get("optimization")
sensitivities = st.session_state.get("sensitivities")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Inputs", "Annual Simulation", "Techno-Economic Optimum", "Sensitivities"]
)

with tab1:
    c1, c2, c3 = st.columns([1.0, 1.15, 1.25])

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

        st.subheader("Pre-run gates")
        if pre_errors:
            for msg in pre_errors:
                st.error(msg)
        else:
            st.success("No blocking pre-run issues detected.")

        for msg in pre_warnings:
            st.warning(msg)

        for msg in pre_infos:
            st.info(msg)

    with c2:
        st.subheader("Model bundle checklist")
        if filtered_catalog.empty:
            st.warning("No bundles detected for the selected library.")
        else:
            preferred_cols = [
                "model_name",
                "ready_for_runtime",
                "ready_for_qa",
                "missing_files",
            ]
            present_cols = [c for c in preferred_cols if c in filtered_catalog.columns]
            st.dataframe(filtered_catalog[present_cols], use_container_width=True, hide_index=True)

            if "ready_for_runtime" in filtered_catalog.columns:
                incomplete = filtered_catalog[filtered_catalog["ready_for_runtime"] != True].copy()
                if not incomplete.empty:
                    st.error("Incomplete runtime bundles detected.")
                    st.dataframe(
                        incomplete[[c for c in ["model_name", "missing_files"] if c in incomplete.columns]],
                        use_container_width=True,
                        hide_index=True,
                    )

    with c3:
        st.subheader("Renewable profile summary")
        st.write(
            {
                "source_kind": effective_profile_meta.get("source_kind"),
                "source_name": effective_profile_meta.get("source_name"),
                "raw_rows": effective_profile_meta.get("raw_rows"),
                "normalized_rows": effective_profile_meta.get("normalized_rows"),
                "median_step_h": effective_profile_meta.get("median_step_h"),
                "looks_daily": effective_profile_meta.get("looks_daily"),
                "expanded_to_hourly": effective_profile_meta.get("expanded_to_hourly"),
                "unit_mode": effective_profile_meta.get("unit_mode"),
                "start": effective_profile_meta.get("start"),
                "end": effective_profile_meta.get("end"),
            }
        )

        st.subheader("Profile-operability diagnostics")
        st.write(
            {
                "peak_mw": round(profile_diag["peak_mw"], 6),
                "mean_mw": round(profile_diag["mean_mw"], 6),
                "annual_energy_gwh_equiv": round(profile_diag["annual_energy_gwh_equiv"], 6),
                "electrolyzer_nominal_power_mw": round(profile_diag["electrolyzer_nominal_power_mw"], 6),
                "electrolyzer_min_load_mw": round(profile_diag["electrolyzer_min_load_mw"], 6),
                "below_min_load_fraction": round(profile_diag["below_min_load_fraction"], 6),
                "zero_power_fraction": round(profile_diag["zero_power_fraction"], 6),
            }
        )

        st.dataframe(effective_profile_df.head(24), use_container_width=True)

with tab2:
    if simulation is None:
        st.info("Press 'Run annual simulation' to generate results.")
    else:
        for warning in getattr(simulation, "warnings", []):
            st.warning(warning)

        kpis = getattr(simulation, "kpis", {})
        ts = getattr(simulation, "time_series", pd.DataFrame())

        export_stub = (
            f"ptmeoh_{slugify(scenario_name)}_{slugify(operating_mode)}_{slugify(active_profile_name)}"
        )

        kpi_export_df = build_kpi_export_dataframe(kpis)
        case_inputs_export_df = build_case_inputs_export_dataframe(
            case_payload=case_payload,
            effective_profile_meta=effective_profile_meta,
            profile_diag=profile_diag,
        )

        kpi_excel_bytes = dataframe_to_excel_bytes(
            kpi_export_df,
            sheet_name="annual_simulation_kpis",
        )
        case_inputs_excel_bytes = dataframe_to_excel_bytes(
            case_inputs_export_df,
            sheet_name="case_inputs",
        )

        st.subheader("Main KPIs")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Annual MeOH [t/y]", f"{kpis.get('annual_methanol_t', float('nan')):,.3f}")
        m2.metric("Electrolyzer FLH [h/y]", f"{kpis.get('electrolyzer_full_load_hours_h', float('nan')):,.1f}")
        m3.metric("PtMeOH utilization [-]", f"{kpis.get('ptmeoh_utilization_factor', float('nan')):.4f}")
        m4.metric("Renewable utilization [-]", f"{kpis.get('renewable_utilization_fraction', float('nan')):.4f}")
        m5.metric("LCOMeOH [USD/t]", f"{kpis.get('lcomeoh_usd_per_t_meoh', float('nan')):,.3f}")
        m6.metric("NPV [USD]", f"{kpis.get('npv_usd', float('nan')):,.1f}")

        st.subheader("Diagnostic KPIs")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Surrogate out-of-domain [-]", f"{kpis.get('surrogate_out_of_domain_fraction', float('nan')):.4f}")
        d2.metric("Curtailment fraction [-]", f"{kpis.get('curtailment_fraction', float('nan')):.4f}")
        d3.metric("Unmet H2 [t/y]", f"{kpis.get('h2_not_supplied_t', float('nan')):,.4f}")
        d4.metric("Runtime models fraction [-]", f"{kpis.get('runtime_models_fraction', float('nan')):.4f}")

        d5, d6, d7, d8 = st.columns(4)
        d5.metric("Tank empty hours [h/y]", f"{kpis.get('tank_empty_hours', float('nan')):,.1f}")
        d6.metric("Tank full hours [h/y]", f"{kpis.get('tank_full_hours', float('nan')):,.1f}")
        d7.metric("Curtailed hours [h/y]", f"{kpis.get('curtailed_hours', float('nan')):,.1f}")
        d8.metric("Annual total electricity [MWh/y]", f"{kpis.get('annual_total_electricity_mwh', float('nan')):,.3f}")

        with st.expander("Export annual simulation review files", expanded=False):
            st.caption(
                "Download the KPI summary and the exact case-input snapshot used for this simulation run."
            )

            export_col1, export_col2 = st.columns(2)

            with export_col1:
                st.download_button(
                    label="Download KPI summary (.xlsx)",
                    data=kpi_excel_bytes,
                    file_name=f"{export_stub}_annual_simulation_kpis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            with export_col2:
                st.download_button(
                    label="Download case inputs (.xlsx)",
                    data=case_inputs_excel_bytes,
                    file_name=f"{export_stub}_case_inputs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            st.dataframe(kpi_export_df, use_container_width=True, hide_index=True)

        if not ts.empty:
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

            st.plotly_chart(
                build_daily_mean_chart(
                    ts,
                    ["renewable_power_mw", "power_to_electrolyzer_mw", "downstream_aux_power_mw"],
                    "Daily mean power profile — full year",
                    "Power [MW]",
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                build_daily_mean_chart(
                    ts,
                    ["h2_produced_kg_per_h", "h2_to_ptmeoh_kg_per_h", "unmet_h2_kg_per_h"],
                    "Daily mean H2 profile — full year",
                    "H2 [kg/h]",
                ),
                use_container_width=True,
            )

            st.plotly_chart(build_diagnostic_event_chart(ts), use_container_width=True)
            st.plotly_chart(build_balance_bar_chart(ts), use_container_width=True)

            with st.expander("Surrogate and runtime diagnostics", expanded=False):
                surrogate_info = getattr(simulation, "surrogate_info", {})
                model_summary = getattr(simulation, "model_summary", pd.DataFrame())
                st.write(surrogate_info)
                if isinstance(model_summary, pd.DataFrame) and not model_summary.empty:
                    st.dataframe(model_summary, use_container_width=True, hide_index=True)

            with st.expander("Traceable hourly results preview", expanded=False):
                st.dataframe(ts.head(96), use_container_width=True)

            with st.expander("Potentially problematic timesteps", expanded=False):
                flagged_cols = [
                    "timestamp",
                    "renewable_power_mw",
                    "power_to_electrolyzer_mw",
                    "h2_produced_kg_per_h",
                    "h2_to_ptmeoh_kg_per_h",
                    "unmet_h2_kg_per_h",
                    "curtailed_power_mw",
                    "tank_soc_kg_h2",
                    "surrogate_all_models_in_domain",
                    "power_deficit_mw",
                ]
                flagged_cols = [c for c in flagged_cols if c in ts.columns]

                unmet_series = ts["unmet_h2_kg_per_h"] if "unmet_h2_kg_per_h" in ts.columns else pd.Series(0, index=ts.index)
                curtailed_series = ts["curtailed_power_mw"] if "curtailed_power_mw" in ts.columns else pd.Series(0, index=ts.index)
                in_domain_series = ts["surrogate_all_models_in_domain"] if "surrogate_all_models_in_domain" in ts.columns else pd.Series(1, index=ts.index)
                deficit_series = ts["power_deficit_mw"] if "power_deficit_mw" in ts.columns else pd.Series(0, index=ts.index)

                flagged = ts[
                    (unmet_series > 0)
                    | (curtailed_series > 0)
                    | (in_domain_series == 0)
                    | (deficit_series > 0)
                ]

                st.dataframe(flagged[flagged_cols].head(200), use_container_width=True)

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
            st.info(
                "Use the shortlist together with the diagnostic KPIs. A mathematically best row is not automatically a trustworthy row if "
                "it still shows high out-of-domain fraction, severe curtailment, or low utilization."
            )

        with right:
            st.plotly_chart(
                heatmap(optimization.results, z_col="lcomeoh_usd_per_t_meoh"),
                use_container_width=True,
            )

        st.subheader("Ranked shortlist")
        shortlist = optimization.results.sort_values(
            ["lcomeoh_usd_per_t_meoh", "warning_count"]
        ).copy()
        priority_cols = [
            "case_name",
            "electrolyzer_power_mw",
            "module_count",
            "storage_kg_h2",
            "target_h2_kg_per_h",
            "annual_methanol_t",
            "ptmeoh_utilization_factor",
            "renewable_utilization_fraction",
            "curtailment_fraction",
            "surrogate_out_of_domain_fraction",
            "warning_count",
            "feasible",
            "lcomeoh_usd_per_t_meoh",
            "npv_usd",
        ]
        priority_cols = [c for c in priority_cols if c in shortlist.columns]
        st.dataframe(shortlist[priority_cols].head(20), use_container_width=True)

with tab4:
    if sensitivities is None:
        st.info("Press 'Run sensitivities' to generate the tornado view.")
    else:
        st.plotly_chart(tornado(sensitivities), use_container_width=True)
        st.dataframe(sensitivities, use_container_width=True)
