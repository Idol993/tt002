import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Tuple
from collections import defaultdict

from .dual_module_wrapper import DualModuleWrapper
from .chaotic_scheduler import ChaoticTrainingScheduler
from .fast_adaptation_compensator import FastAdaptationCompensator
from .meta_cycle_optimizer import MetaCycleOptimizer


class SupersymmetricTrainer:
    def __init__(
        self,
        dual_wrapper: DualModuleWrapper,
        scheduler: ChaoticTrainingScheduler,
        compensator: FastAdaptationCompensator,
        meta_optimizer: MetaCycleOptimizer,
        base_optimizer_a: torch.optim.Optimizer,
        base_optimizer_b: torch.optim.Optimizer,
        loss_fn: Callable,
        robustness_threshold: float = 0.02,
        log_interval: int = 100,
    ):
        self.dual_wrapper = dual_wrapper
        self.scheduler = scheduler
        self.compensator = compensator
        self.meta_optimizer = meta_optimizer
        self.opt_a = base_optimizer_a
        self.opt_b = base_optimizer_b
        self.loss_fn = loss_fn
        self.robustness_threshold = robustness_threshold
        self.log_interval = log_interval

        self._global_step = 0
        self._reversal_count = 0
        self._training_log: List[Dict] = []
        self._reversal_log: List[Dict] = []
        self._best_loss: Optional[float] = None
        self._reference_performance: Optional[float] = None

    def train_step(self, batch_input: torch.Tensor, batch_target: torch.Tensor) -> Dict:
        self._global_step += 1
        step_info: Dict = {"step": self._global_step}

        should_reverse = self.scheduler.should_reverse(self._global_step)

        if should_reverse:
            reversal_info = self._execute_reversal(batch_input, batch_target)
            step_info["reversal"] = reversal_info

        output, intermediate = self.dual_wrapper.forward_with_intermediate(batch_input)
        loss = self.loss_fn(output, batch_target)

        self.opt_a.zero_grad()
        self.opt_b.zero_grad()
        loss.backward()

        if self.compensator._compensation_gradient_cache is not None:
            all_params = self.dual_wrapper.get_all_params()
            comp_grads = self.compensator._compensation_gradient_cache
            with torch.no_grad():
                for param, cg in zip(all_params, comp_grads):
                    if param.grad is not None:
                        param.grad.add_(cg)

        self.opt_a.step()
        self.opt_b.step()

        self.compensator._ema_loss = (
            loss.item()
            if self.compensator._ema_loss is None
            else self.compensator.ema_momentum * self.compensator._ema_loss
            + (1 - self.compensator.ema_momentum) * loss.item()
        )

        if self._reference_performance is None:
            self._reference_performance = loss.item()

        step_info["loss"] = loss.item()
        step_info["is_swapped"] = self.dual_wrapper.is_swapped
        step_info["frequency"] = self.meta_optimizer.frequency
        step_info["compensation_strength"] = self.meta_optimizer.compensation_strength

        if self._global_step % self.log_interval == 0:
            self._training_log.append(step_info)

        return step_info

    def _execute_reversal(
        self, batch_input: torch.Tensor, batch_target: torch.Tensor
    ) -> Dict:
        self._reversal_count += 1
        reversal_info: Dict = {"reversal_number": self._reversal_count, "step": self._global_step}

        with torch.no_grad():
            out_a, out_b = self.dual_wrapper.forward_separate(batch_input)
            combined_out, _ = self.dual_wrapper.forward_with_intermediate(batch_input)
            pre_loss = self.loss_fn(combined_out, batch_target).item()

        self.compensator.capture_pre_reversal_state(out_a, out_b, combined_out, pre_loss)
        self.meta_optimizer.record_pre_reversal_loss(pre_loss)
        reversal_info["pre_reversal_loss"] = pre_loss

        param_swap_info = self.dual_wrapper.swap_parameters()
        opt_swap_info = self.dual_wrapper.swap_optimizer_states(self.opt_a, self.opt_b)
        reversal_info["swapped"] = True

        with torch.no_grad():
            out_a_post, out_b_post = self.dual_wrapper.forward_separate(batch_input)
            combined_out_post, _ = self.dual_wrapper.forward_with_intermediate(batch_input)

        kl_div = self.compensator.capture_post_reversal_state(out_a_post, out_b_post, combined_out_post)
        reversal_info["kl_divergence"] = kl_div

        post_loss = self.loss_fn(combined_out_post, batch_target).item()
        reversal_info["post_reversal_loss"] = post_loss
        self.meta_optimizer.record_post_reversal_loss(post_loss)

        loss_spike = abs(post_loss - pre_loss) / (abs(pre_loss) + 1e-8)
        self.scheduler.record_loss_spike(loss_spike)
        reversal_info["loss_spike_ratio"] = loss_spike

        if loss_spike > self.compensator.spike_threshold:
            adaptation_result = self.compensator.run_fast_adaptation(
                self.dual_wrapper,
                batch_input,
                self.loss_fn,
                batch_target,
            )
            reversal_info["adaptation"] = adaptation_result

        self._update_meta_cycle(loss_spike, post_loss)

        reversal_info["meta_frequency"] = self.meta_optimizer.frequency
        reversal_info["meta_compensation"] = self.meta_optimizer.compensation_strength

        self._reversal_log.append(reversal_info)
        return reversal_info

    def _update_meta_cycle(self, loss_spike: float, post_reversal_loss: float) -> None:
        current_performance = post_reversal_loss

        self.meta_optimizer.record_baseline_performance(current_performance)

        self.scheduler.update_frequency(self.meta_optimizer.frequency)
        self.compensator.update_compensation_strength(self.meta_optimizer.compensation_strength)

    def evaluate_robustness(self, eval_dataloader) -> Dict:
        self.dual_wrapper.eval()
        total_loss_normal = 0.0
        total_loss_swapped = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_input, batch_target in eval_dataloader:
                out_normal, _ = self.dual_wrapper.forward_with_intermediate(batch_input)
                loss_normal = self.loss_fn(out_normal, batch_target).item()
                total_loss_normal += loss_normal

                current_swap_state = self.dual_wrapper.is_swapped
                self.dual_wrapper._swapped = not current_swap_state
                out_swapped, _ = self.dual_wrapper.forward_with_intermediate(batch_input)
                loss_swapped = self.loss_fn(out_swapped, batch_target).item()
                total_loss_swapped += loss_swapped

                self.dual_wrapper._swapped = current_swap_state
                n_batches += 1

        self.dual_wrapper.train()

        avg_normal = total_loss_normal / max(n_batches, 1)
        avg_swapped = total_loss_swapped / max(n_batches, 1)
        performance_drop = abs(avg_swapped - avg_normal) / (avg_normal + 1e-8)
        is_robust = performance_drop <= self.robustness_threshold

        return {
            "avg_loss_normal": avg_normal,
            "avg_loss_swapped": avg_swapped,
            "performance_drop": performance_drop,
            "performance_drop_pct": performance_drop * 100,
            "is_robust": is_robust,
            "robustness_threshold_pct": self.robustness_threshold * 100,
            "supersymmetry_achieved": is_robust,
        }

    def train_epoch(self, dataloader, epoch: int = 0) -> Dict:
        self.dual_wrapper.train()
        epoch_losses = []
        epoch_reversals = 0

        for batch_input, batch_target in dataloader:
            step_info = self.train_step(batch_input, batch_target)
            epoch_losses.append(step_info["loss"])
            if "reversal" in step_info:
                epoch_reversals += 1

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        return {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "reversals_this_epoch": epoch_reversals,
            "total_reversals": self._reversal_count,
            "total_steps": self._global_step,
            "current_frequency": self.meta_optimizer.frequency,
            "current_compensation": self.meta_optimizer.compensation_strength,
        }

    def get_training_summary(self) -> Dict:
        return {
            "total_steps": self._global_step,
            "total_reversals": self._reversal_count,
            "scheduler_stats": self.scheduler.get_reversal_stats(),
            "meta_state": self.meta_optimizer.get_meta_state(),
            "compensator_kl_history": self.compensator._kl_history,
            "reversal_log": self._reversal_log,
        }

    def state_dict(self) -> Dict:
        return {
            "global_step": self._global_step,
            "reversal_count": self._reversal_count,
            "dual_wrapper": self.dual_wrapper.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "compensator": self.compensator.state_dict(),
            "meta_optimizer": self.meta_optimizer.state_dict(),
            "opt_a": self.opt_a.state_dict(),
            "opt_b": self.opt_b.state_dict(),
            "best_loss": self._best_loss,
            "reference_performance": self._reference_performance,
        }

    def load_state_dict(self, state: Dict) -> None:
        self._global_step = state["global_step"]
        self._reversal_count = state["reversal_count"]
        self.dual_wrapper.load_state_dict(state["dual_wrapper"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.compensator.load_state_dict(state["compensator"])
        self.meta_optimizer.load_state_dict(state["meta_optimizer"])
        self.opt_a.load_state_dict(state["opt_a"])
        self.opt_b.load_state_dict(state["opt_b"])
        self._best_loss = state["best_loss"]
        self._reference_performance = state["reference_performance"]
