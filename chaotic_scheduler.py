import torch
import numpy as np
from typing import Optional, List, Dict
import math


class ChaoticTrainingScheduler:
    def __init__(
        self,
        seed: int = 42,
        base_interval: int = 500,
        min_interval: int = 100,
        max_interval: int = 2000,
        chaos_strength: float = 0.3,
        warmup_steps: int = 200,
        initial_frequency: float = 1.0 / 500,
    ):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.chaos_strength = chaos_strength
        self.warmup_steps = warmup_steps
        self.current_frequency = initial_frequency
        self.current_step = 0
        self.next_reversal_step = self._compute_next_reversal(0)
        self.reversal_history: List[int] = []
        self.loss_spike_history: List[float] = []
        self.frequency_schedule: List[float] = []

    def _compute_next_reversal(self, from_step: int) -> int:
        if from_step < self.warmup_steps:
            first_after_warmup = self.warmup_steps + self._sample_interval()
            return first_after_warmup

        interval = self._sample_interval()
        return from_step + interval

    def _sample_interval(self) -> int:
        chaos_factor = 1.0 + self.chaos_strength * (self.rng.randn())
        interval = int(self.base_interval * chaos_factor / max(self.current_frequency * self.base_interval, 0.01))
        interval = max(self.min_interval, min(self.max_interval, interval))
        return interval

    def should_reverse(self, step: int) -> bool:
        self.current_step = step
        if step >= self.next_reversal_step:
            self.reversal_history.append(step)
            self.next_reversal_step = self._compute_next_reversal(step)
            return True
        return False

    def update_frequency(self, new_frequency: float) -> None:
        self.current_frequency = float(np.clip(new_frequency, 1.0 / self.max_interval, 1.0 / self.min_interval))
        self.frequency_schedule.append(self.current_frequency)

    def record_loss_spike(self, spike_magnitude: float) -> None:
        self.loss_spike_history.append(spike_magnitude)

    def get_reversal_stats(self) -> Dict:
        if len(self.reversal_history) < 2:
            return {
                "total_reversals": len(self.reversal_history),
                "mean_interval": self.base_interval,
                "std_interval": 0.0,
                "current_frequency": self.current_frequency,
            }
        intervals = np.diff(self.reversal_history)
        return {
            "total_reversals": len(self.reversal_history),
            "mean_interval": float(np.mean(intervals)),
            "std_interval": float(np.std(intervals)),
            "current_frequency": self.current_frequency,
        }

    def predict_reversal_steps(self, n_ahead: int = 10) -> List[int]:
        rng_copy = np.random.RandomState(self.seed)
        rng_copy.set_state(self.rng.get_state())
        steps = []
        current = self.current_step
        freq = self.current_frequency
        for _ in range(n_ahead):
            chaos_factor = 1.0 + self.chaos_strength * rng_copy.randn()
            interval = int(self.base_interval * chaos_factor / max(freq * self.base_interval, 0.01))
            interval = max(self.min_interval, min(self.max_interval, interval))
            current += interval
            steps.append(current)
        return steps

    def state_dict(self) -> Dict:
        return {
            "seed": self.seed,
            "base_interval": self.base_interval,
            "min_interval": self.min_interval,
            "max_interval": self.max_interval,
            "chaos_strength": self.chaos_strength,
            "warmup_steps": self.warmup_steps,
            "current_frequency": self.current_frequency,
            "current_step": self.current_step,
            "next_reversal_step": self.next_reversal_step,
            "reversal_history": self.reversal_history,
            "loss_spike_history": self.loss_spike_history,
            "rng_state": self.rng.get_state(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self.current_frequency = state["current_frequency"]
        self.current_step = state["current_step"]
        self.next_reversal_step = state["next_reversal_step"]
        self.reversal_history = state["reversal_history"]
        self.loss_spike_history = state["loss_spike_history"]
        self.rng.set_state(state["rng_state"])
