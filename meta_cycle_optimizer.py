import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import math


class MetaCycleOptimizer:
    def __init__(
        self,
        initial_frequency: float = 1.0 / 500,
        initial_compensation: float = 1.0,
        frequency_lr: float = 1e-4,
        compensation_lr: float = 1e-3,
        performance_threshold: float = 0.02,
        meta_update_interval: int = 5,
        frequency_bounds: Tuple[float, float] = (1.0 / 2000, 1.0 / 50),
        compensation_bounds: Tuple[float, float] = (0.1, 10.0),
        ema_decay: float = 0.9,
        higher_is_better: bool = False,
    ):
        self.frequency = initial_frequency
        self.compensation_strength = initial_compensation
        self.frequency_lr = frequency_lr
        self.compensation_lr = compensation_lr
        self.performance_threshold = performance_threshold
        self.meta_update_interval = meta_update_interval
        self.frequency_bounds = frequency_bounds
        self.compensation_bounds = compensation_bounds
        self.ema_decay = ema_decay
        self.higher_is_better = higher_is_better

        self._reversal_count = 0
        self._pending_pre_metric: Optional[float] = None
        self._degradation_buffer: List[float] = []
        self._baseline_performance: Optional[float] = None
        self._ema_performance: Optional[float] = None
        self._meta_step = 0
        self._performance_history: List[float] = []
        self._frequency_history: List[float] = [initial_frequency]
        self._compensation_history: List[float] = [initial_compensation]
        self._degradation_history: List[float] = []

    def _compute_degradation(self, post_metric: float, pre_metric: float) -> float:
        if self.higher_is_better:
            if pre_metric <= 0:
                return 0.0
            relative_drop = (pre_metric - post_metric) / pre_metric
            return max(0.0, relative_drop)
        else:
            if pre_metric <= 0:
                return 0.0
            relative_rise = (post_metric - pre_metric) / pre_metric
            return max(0.0, relative_rise)

    def record_pre_reversal_metric(self, metric: float) -> None:
        self._pending_pre_metric = metric

    def record_post_reversal_metric(self, metric: float) -> None:
        if self._pending_pre_metric is not None:
            degradation = self._compute_degradation(metric, self._pending_pre_metric)
            self._degradation_buffer.append(degradation)
            self._pending_pre_metric = None

        self._reversal_count += 1

        if self._reversal_count % self.meta_update_interval == 0 and self._degradation_buffer:
            self._meta_update()

    def record_baseline_performance(self, performance: float) -> None:
        if self._baseline_performance is None:
            self._baseline_performance = performance
        self._ema_performance = (
            performance
            if self._ema_performance is None
            else self.ema_decay * self._ema_performance + (1 - self.ema_decay) * performance
        )
        self._performance_history.append(performance)

    def _meta_update(self) -> None:
        if not self._degradation_buffer:
            return

        avg_degradation = float(np.mean(self._degradation_buffer[-self.meta_update_interval:]))
        self._degradation_history.append(avg_degradation)

        freq_delta = self._compute_frequency_delta(avg_degradation)
        comp_delta = self._compute_compensation_delta(avg_degradation)

        self.frequency = float(
            np.clip(
                self.frequency + freq_delta,
                self.frequency_bounds[0],
                self.frequency_bounds[1],
            )
        )
        self.compensation_strength = float(
            np.clip(
                self.compensation_strength + comp_delta,
                self.compensation_bounds[0],
                self.compensation_bounds[1],
            )
        )

        self._frequency_history.append(self.frequency)
        self._compensation_history.append(self.compensation_strength)
        self._meta_step += 1

    def _compute_frequency_delta(self, degradation: float) -> float:
        threshold = self.performance_threshold
        freq_range = self.frequency_bounds[1] - self.frequency_bounds[0]

        if degradation > threshold * 3.0:
            return -freq_range * 0.15
        elif degradation > threshold * 2.0:
            return -freq_range * 0.10
        elif degradation > threshold * 1.5:
            return -freq_range * 0.05
        elif degradation > threshold:
            return -freq_range * 0.02
        elif degradation > threshold * 0.7:
            return 0.0
        elif degradation > threshold * 0.3:
            return freq_range * 0.005
        else:
            return freq_range * 0.01

    def _compute_compensation_delta(self, degradation: float) -> float:
        threshold = self.performance_threshold
        comp_range = self.compensation_bounds[1] - self.compensation_bounds[0]

        if degradation > threshold * 3.0:
            return comp_range * 0.20
        elif degradation > threshold * 2.0:
            return comp_range * 0.12
        elif degradation > threshold * 1.5:
            return comp_range * 0.06
        elif degradation > threshold:
            return comp_range * 0.03
        elif degradation > threshold * 0.7:
            return comp_range * 0.005
        elif degradation > threshold * 0.3:
            return -comp_range * 0.005
        else:
            return -comp_range * 0.01

    def get_adjustment_direction(self) -> str:
        if len(self._degradation_history) < 2:
            return "initializing"
        recent = self._degradation_history[-1]
        threshold = self.performance_threshold
        if recent > threshold * 2:
            return "emergency_brake"
        elif recent > threshold:
            return "conservative"
        elif recent > threshold * 0.5:
            return "stable"
        else:
            return "exploring"

    def check_functional_robustness(self, current_metric: float, baseline_metric: float) -> bool:
        degradation = self._compute_degradation(current_metric, baseline_metric)
        return degradation <= self.performance_threshold

    def get_meta_state(self) -> Dict:
        return {
            "frequency": self.frequency,
            "compensation_strength": self.compensation_strength,
            "meta_step": self._meta_step,
            "reversal_count": self._reversal_count,
            "baseline_performance": self._baseline_performance,
            "ema_performance": self._ema_performance,
            "adjustment_mode": self.get_adjustment_direction(),
            "last_degradation": self._degradation_history[-1] if self._degradation_history else None,
        }

    def state_dict(self) -> Dict:
        return {
            "frequency": self.frequency,
            "compensation_strength": self.compensation_strength,
            "frequency_lr": self.frequency_lr,
            "compensation_lr": self.compensation_lr,
            "performance_threshold": self.performance_threshold,
            "meta_update_interval": self.meta_update_interval,
            "frequency_bounds": self.frequency_bounds,
            "compensation_bounds": self.compensation_bounds,
            "ema_decay": self.ema_decay,
            "higher_is_better": self.higher_is_better,
            "reversal_count": self._reversal_count,
            "pending_pre_metric": self._pending_pre_metric,
            "degradation_buffer": self._degradation_buffer,
            "baseline_performance": self._baseline_performance,
            "ema_performance": self._ema_performance,
            "meta_step": self._meta_step,
            "performance_history": self._performance_history,
            "frequency_history": self._frequency_history,
            "compensation_history": self._compensation_history,
            "degradation_history": self._degradation_history,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.frequency = state["frequency"]
        self.compensation_strength = state["compensation_strength"]
        self.frequency_lr = state["frequency_lr"]
        self.compensation_lr = state["compensation_lr"]
        self.performance_threshold = state["performance_threshold"]
        self.meta_update_interval = state["meta_update_interval"]
        self.frequency_bounds = state["frequency_bounds"]
        self.compensation_bounds = state["compensation_bounds"]
        self.ema_decay = state["ema_decay"]
        self.higher_is_better = state.get("higher_is_better", False)
        self._reversal_count = state["reversal_count"]
        self._pending_pre_metric = state.get("pending_pre_metric", None)
        self._degradation_buffer = state.get("degradation_buffer", [])
        self._baseline_performance = state["baseline_performance"]
        self._ema_performance = state["ema_performance"]
        self._meta_step = state["meta_step"]
        self._performance_history = state["performance_history"]
        self._frequency_history = state["frequency_history"]
        self._compensation_history = state["compensation_history"]
        self._degradation_history = state.get("degradation_history", [])
