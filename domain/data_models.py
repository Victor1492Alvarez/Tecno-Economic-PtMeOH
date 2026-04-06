from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

@dataclass
class ScenarioEconomicInputs:
    electricity_price_usd_per_mwh: float
    co2_price_usd_per_t: float
    methanol_price_usd_per_t: float
    discount_rate: float
    project_years: int
    capex_multiplier: float = 1.0
    opex_multiplier: float = 1.0

@dataclass
class ElectrolyzerInputs:
    nominal_power_mw: float
    module_size_mw: float
    min_load_fraction: float
    specific_energy_kwh_per_kg_h2: float
    capex_usd_per_kw: float
    fixed_opex_fraction: float

@dataclass
class StorageInputs:
    enabled: bool
    usable_capacity_kg_h2: float
    initial_soc_fraction: float
    max_charge_kg_per_h: Optional[float] = None
    max_discharge_kg_per_h: Optional[float] = None
    capex_usd_per_kg_h2: float = 20.0

@dataclass
class PtMeOHInputs:
    operating_mode: str
    surrogate_library: str
    target_h2_feed_kg_per_h: float
    max_h2_feed_kg_per_h: float
    methanol_yield_t_meoh_per_t_h2: float = 7.95
    unmet_h2_warning_threshold: float = 0.10
    curtailment_warning_threshold: float = 0.35
    utilization_warning_threshold: float = 0.70

@dataclass
class OptimizationInputs:
    electrolyzer_power_grid_mw: List[float]
    storage_grid_kg_h2: List[float]
    target_h2_grid_kg_per_h: List[float]
    module_count_grid: List[int]
    objective: str = "min_lcomeoh"
    min_ptmeoh_utilization: float = 0.70
    max_unmet_h2_fraction: float = 0.10
    max_curtailment_fraction: float = 0.35

@dataclass
class CaseInputs:
    scenario_name: str
    economic: ScenarioEconomicInputs
    electrolyzer: ElectrolyzerInputs
    storage: StorageInputs
    ptmeoh: PtMeOHInputs
    optimization: OptimizationInputs
    renewable_profile: pd.DataFrame
    time_step_h: float = 1.0
    case_name: str = "base_case"

@dataclass
class SimulationArtifacts:
    time_series: pd.DataFrame
    kpis: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    surrogate_info: Dict[str, Any] = field(default_factory=dict)
    model_summary: pd.DataFrame = field(default_factory=pd.DataFrame)

@dataclass
class GridSearchArtifacts:
    results: pd.DataFrame
    feasible_results: pd.DataFrame
    best_row: pd.Series

