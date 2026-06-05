from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any

from domain.data_models import CaseInputs
from domain.simulation_engine import SimulationEngine
from domain.optimizer_grid import GridOptimizer
from domain.sensitivity_analysis import SensitivityAnalyzer

ProgressCallback = Callable[[str, int, int], None] | None


class CaseRunner:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.engine = SimulationEngine(self.project_root)
        self.optimizer = GridOptimizer(self.engine)
        self.sensitivity = SensitivityAnalyzer(self.engine)
        self.logger = self.engine.logger

    def run_simulation(
        self,
        case: CaseInputs,
        progress_callback: ProgressCallback = None,
    ):
        self.logger.info("Running base simulation")
        return self.engine.run(case, progress_callback=progress_callback)

    def run_optimization(
        self,
        case: CaseInputs,
        progress_callback: ProgressCallback = None,
    ):
        self.logger.info("Running optimization")
        return self.optimizer.run(case, progress_callback=progress_callback)

    def run_sensitivity(self, case: CaseInputs):
        self.logger.info("Running sensitivity analysis")
        return self.sensitivity.run(case)

    def run_all(
        self,
        case: CaseInputs,
        run_optimization: bool = True,
        run_sensitivity: bool = True,
        progress_callback: ProgressCallback = None,
    ) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        outputs["simulation"] = self.run_simulation(case, progress_callback=progress_callback)

        if run_optimization:
            outputs["optimization"] = self.run_optimization(case, progress_callback=progress_callback)
        else:
            self.logger.info("Optimization skipped by user")
            outputs["optimization"] = None

        if run_sensitivity:
            outputs["sensitivity"] = self.run_sensitivity(case)
        else:
            self.logger.info("Sensitivity analysis skipped by user")
            outputs["sensitivity"] = None

        return outputs
