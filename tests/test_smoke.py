from pathlib import Path
from application.case_runner import CaseRunner


def test_case_runner_smoke():
    root = Path(__file__).resolve().parents
    runner = CaseRunner(root)
    case = runner.build_case(
        scenario_name="moderate",
        electrolyzer_power_mw=50.0,
        module_count=4,
        storage_enabled=True,
        storage_kg_h2=10000.0,
        operating_mode="quasi_base_load",
        surrogate_library="variable_h2_constant_co2",
        target_h2_kg_per_h=1000.0,
        max_h2_feed_kg_per_h=1200.0,
        renewable_peak_power_mw=120.0,
    )
    result = runner.engine.run(case)
    assert "annual_methanol_t" in result.kpis

