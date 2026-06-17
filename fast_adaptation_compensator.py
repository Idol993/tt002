import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class FastAdaptationCompensator:
    def __init__(
        self,
        compensation_strength: float = 1.0,
        kl_temperature: float = 1.0,
        ema_momentum: float = 0.9,
        max_compensation_epochs: int = 50,
        compensation_lr: float = 0.01,
        min_adaptation_steps: int = 3,
        kl_aware_scaling: bool = True,
        max_kl_scale: float = 3.0,
        grad_clip: float = 1.0,
    ):
        self.compensation_strength = compensation_strength
        self.kl_temperature = kl_temperature
        self.ema_momentum = ema_momentum
        self.max_compensation_epochs = max_compensation_epochs
        self.compensation_lr = compensation_lr
        self.min_adaptation_steps = min_adaptation_steps
        self.kl_aware_scaling = kl_aware_scaling
        self.max_kl_scale = max_kl_scale
        self.grad_clip = grad_clip

        self._pre_reversal_stats: Optional[Dict[str, torch.Tensor]] = None
        self._post_reversal_stats: Optional[Dict[str, torch.Tensor]] = None
        self._baseline_loss: Optional[float] = None
        self._ema_loss: Optional[float] = None
        self._target_distribution: Optional[Dict[str, torch.Tensor]] = None
        self._current_kl: float = 0.0
        self._kl_history: List[float] = []
        self._adaptation_history: List[Dict] = []

    def capture_pre_reversal_state(
        self,
        module_a_output: torch.Tensor,
        module_b_output: torch.Tensor,
        combined_output: torch.Tensor,
        current_loss: float,
    ) -> None:
        with torch.no_grad():
            self._pre_reversal_stats = {
                "module_a_mean": module_a_output.mean(dim=0).detach().clone(),
                "module_a_std": module_a_output.std(dim=0).detach().clone() + 1e-8,
                "module_b_mean": module_b_output.mean(dim=0).detach().clone(),
                "module_b_std": module_b_output.std(dim=0).detach().clone() + 1e-8,
                "combined_mean": combined_output.mean(dim=0).detach().clone(),
                "combined_std": combined_output.std(dim=0).detach().clone() + 1e-8,
            }
            self._target_distribution = {
                "mean": combined_output.mean(dim=0).detach().clone(),
                "std": combined_output.std(dim=0).detach().clone() + 1e-8,
            }
            self._baseline_loss = current_loss
            if self._ema_loss is None:
                self._ema_loss = current_loss
            else:
                self._ema_loss = self.ema_momentum * self._ema_loss + (1 - self.ema_momentum) * current_loss

    def capture_post_reversal_state(
        self,
        module_a_output: torch.Tensor,
        module_b_output: torch.Tensor,
        combined_output: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            self._post_reversal_stats = {
                "module_a_mean": module_a_output.mean(dim=0).detach().clone(),
                "module_a_std": module_a_output.std(dim=0).detach().clone() + 1e-8,
                "module_b_mean": module_b_output.mean(dim=0).detach().clone(),
                "module_b_std": module_b_output.std(dim=0).detach().clone() + 1e-8,
                "combined_mean": combined_output.mean(dim=0).detach().clone(),
                "combined_std": combined_output.std(dim=0).detach().clone() + 1e-8,
            }

        kl_div = self._compute_kl_divergence()
        self._current_kl = self._compute_combined_kl()
        return kl_div

    def _compute_kl_divergence(self) -> float:
        if self._pre_reversal_stats is None or self._post_reversal_stats is None:
            return 0.0

        kl_values = []
        for key_prefix in ["module_a", "module_b", "combined"]:
            pre_mean = self._pre_reversal_stats[f"{key_prefix}_mean"]
            pre_std = self._pre_reversal_stats[f"{key_prefix}_std"]
            post_mean = self._post_reversal_stats[f"{key_prefix}_mean"]
            post_std = self._post_reversal_stats[f"{key_prefix}_std"]

            var_pre = (pre_std * self.kl_temperature) ** 2
            var_post = (post_std * self.kl_temperature) ** 2

            kl = 0.5 * (
                (var_pre / var_post).log().mean()
                + (var_post / var_pre).mean()
                + ((pre_mean - post_mean) ** 2 / var_pre).mean()
                - 1
            )
            kl_values.append(max(kl.item(), 0.0))

        total_kl = sum(kl_values) / len(kl_values)
        self._kl_history.append(total_kl)
        return total_kl

    def _compute_combined_kl(self) -> float:
        if self._pre_reversal_stats is None or self._post_reversal_stats is None:
            return 0.0

        pre_mean = self._pre_reversal_stats["combined_mean"]
        pre_std = self._pre_reversal_stats["combined_std"]
        post_mean = self._post_reversal_stats["combined_mean"]
        post_std = self._post_reversal_stats["combined_std"]

        var_pre = (pre_std * self.kl_temperature) ** 2
        var_post = (post_std * self.kl_temperature) ** 2

        kl = 0.5 * (
            (var_pre / var_post).log().mean()
            + (var_post / var_pre).mean()
            + ((pre_mean - post_mean) ** 2 / var_pre).mean()
            - 1
        )
        return max(kl.item(), 0.0)

    def compute_kl_alignment_loss(
        self,
        current_output: torch.Tensor,
    ) -> torch.Tensor:
        if self._target_distribution is None:
            return torch.tensor(0.0, device=current_output.device)

        current_mean = current_output.mean(dim=0)
        current_std = current_output.std(dim=0) + 1e-8

        target_mean = self._target_distribution["mean"]
        target_std = self._target_distribution["std"]

        var_current = (current_std * self.kl_temperature) ** 2
        var_target = (target_std * self.kl_temperature) ** 2

        kl_loss = 0.5 * (
            (var_target / var_current).log().mean()
            + (var_current / var_target).mean()
            + ((target_mean - current_mean) ** 2 / var_target).mean()
            - 1
        )

        return kl_loss

    def compute_compensation_gradient(
        self,
        dual_wrapper,
        sample_input: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], float]:
        if self._target_distribution is None:
            all_params = dual_wrapper.get_all_params()
            return [torch.zeros_like(p) for p in all_params], 0.0

        output = dual_wrapper(sample_input)
        kl_loss = self.compute_kl_alignment_loss(output)

        all_params = dual_wrapper.get_all_params()
        if kl_loss.requires_grad:
            kl_grads = torch.autograd.grad(kl_loss, all_params, retain_graph=True, allow_unused=True)
        else:
            kl_grads = [torch.zeros_like(p) for p in all_params]

        kl_grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(kl_grads, all_params)]

        kl_magnitude = kl_loss.item()
        if self.kl_aware_scaling and self._current_kl > 0:
            scale_factor = 1.0 + min(self._current_kl, self.max_kl_scale)
        else:
            scale_factor = 1.0

        effective_scale = self.compensation_strength * scale_factor

        compensation_grads = [-effective_scale * g for g in kl_grads]

        if self.grad_clip > 0:
            total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in compensation_grads))
            if total_norm > self.grad_clip:
                clip_coef = self.grad_clip / (total_norm + 1e-6)
                compensation_grads = [g * clip_coef for g in compensation_grads]

        return compensation_grads, kl_magnitude

    def apply_compensation(
        self,
        module_params: List[torch.Tensor],
        compensation_grads: List[torch.Tensor],
    ) -> None:
        with torch.no_grad():
            for param, comp_grad in zip(module_params, compensation_grads):
                param.add_(comp_grad * self.compensation_lr)

    def update_ema_loss(self, loss_val: float) -> None:
        if self._ema_loss is None:
            self._ema_loss = loss_val
        else:
            self._ema_loss = self.ema_momentum * self._ema_loss + (1 - self.ema_momentum) * loss_val

    def run_fast_adaptation(
        self,
        dual_wrapper,
        sample_input: torch.Tensor,
        loss_fn,
        target: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> Dict:
        if n_steps is None:
            n_steps = self.max_compensation_epochs

        n_steps = max(n_steps, self.min_adaptation_steps)

        adaptation_losses = []
        kl_values = []

        for step in range(n_steps):
            output = dual_wrapper(sample_input)
            loss = loss_fn(output, target)
            adaptation_losses.append(loss.item())

            comp_grads, current_kl = self.compute_compensation_gradient(dual_wrapper, sample_input)
            kl_values.append(current_kl)

            all_params = dual_wrapper.get_all_params()
            self.apply_compensation(all_params, comp_grads)

            if step > self.min_adaptation_steps and abs(adaptation_losses[-1] - adaptation_losses[-2]) < 1e-5:
                break

        with torch.no_grad():
            final_output = dual_wrapper(sample_input)
            final_loss = loss_fn(final_output, target).item()

        initial_loss = adaptation_losses[0] if adaptation_losses else 0.0
        loss_reduction = initial_loss - final_loss if len(adaptation_losses) > 0 else 0.0
        initial_kl = kl_values[0] if kl_values else 0.0
        final_kl = kl_values[-1] if kl_values else 0.0
        kl_reduction = initial_kl - final_kl

        result = {
            "adaptation_steps": len(adaptation_losses),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": loss_reduction,
            "loss_reduction_pct": (loss_reduction / (initial_loss + 1e-8) * 100) if initial_loss > 0 else 0.0,
            "initial_kl": initial_kl,
            "final_kl": final_kl,
            "kl_reduction": kl_reduction,
            "compensation_strength_used": self.compensation_strength,
        }

        self._adaptation_history.append(result)
        return result

    def update_compensation_strength(self, new_strength: float) -> None:
        self.compensation_strength = float(max(0.0, min(10.0, new_strength)))

    def get_last_adaptation_stats(self) -> Optional[Dict]:
        if not self._adaptation_history:
            return None
        return self._adaptation_history[-1]

    def state_dict(self) -> Dict:
        return {
            "compensation_strength": self.compensation_strength,
            "kl_temperature": self.kl_temperature,
            "ema_momentum": self.ema_momentum,
            "max_compensation_epochs": self.max_compensation_epochs,
            "compensation_lr": self.compensation_lr,
            "min_adaptation_steps": self.min_adaptation_steps,
            "kl_aware_scaling": self.kl_aware_scaling,
            "max_kl_scale": self.max_kl_scale,
            "grad_clip": self.grad_clip,
            "ema_loss": self._ema_loss,
            "baseline_loss": self._baseline_loss,
            "current_kl": self._current_kl,
            "kl_history": self._kl_history,
            "adaptation_history": self._adaptation_history,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.compensation_strength = state["compensation_strength"]
        self.kl_temperature = state["kl_temperature"]
        self.ema_momentum = state["ema_momentum"]
        self.max_compensation_epochs = state["max_compensation_epochs"]
        self.compensation_lr = state["compensation_lr"]
        self.min_adaptation_steps = state["min_adaptation_steps"]
        self.kl_aware_scaling = state.get("kl_aware_scaling", True)
        self.max_kl_scale = state.get("max_kl_scale", 3.0)
        self.grad_clip = state.get("grad_clip", 1.0)
        self._ema_loss = state["ema_loss"]
        self._baseline_loss = state["baseline_loss"]
        self._current_kl = state.get("current_kl", 0.0)
        self._kl_history = state["kl_history"]
        self._adaptation_history = state.get("adaptation_history", [])
