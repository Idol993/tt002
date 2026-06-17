import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tt002 import (
    DualModuleWrapper,
    ChaoticTrainingScheduler,
    FastAdaptationCompensator,
    MetaCycleOptimizer,
    SupersymmetricTrainer,
)


class IsomorphicModule(nn.Module):
    def __init__(self, dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassificationHead(nn.Module):
    def __init__(self, dim: int = 64, n_classes: int = 10):
        super().__init__()
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SupersymmetricModel(nn.Module):
    def __init__(self, dim: int = 64, hidden_dim: int = 128, n_classes: int = 10):
        super().__init__()
        self.syntax = IsomorphicModule(dim, hidden_dim)
        self.semantics = IsomorphicModule(dim, hidden_dim)
        self.classifier = ClassificationHead(dim, n_classes)

    def forward_dual(self, x: torch.Tensor, wrapper: DualModuleWrapper) -> torch.Tensor:
        h = wrapper(x)
        return self.classifier(h)


def create_synthetic_data(n_samples: int = 2000, input_dim: int = 64, n_classes: int = 10):
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    return TensorDataset(X, y)


class RobustTrainer:
    def __init__(self):
        torch.manual_seed(42)

        self.dim = 64
        self.hidden_dim = 128
        self.n_classes = 10
        self.robustness_threshold = 0.02

        self.model = SupersymmetricModel(self.dim, self.hidden_dim, self.n_classes)

        self.dual_wrapper = DualModuleWrapper(
            self.model.syntax, self.model.semantics, name_a="syntax", name_b="semantics"
        )

        self.scheduler = ChaoticTrainingScheduler(
            seed=42,
            base_interval=15,
            min_interval=5,
            max_interval=40,
            chaos_strength=0.3,
            warmup_steps=20,
        )

        self.compensator = FastAdaptationCompensator(
            compensation_strength=1.5,
            kl_temperature=1.0,
            ema_momentum=0.9,
            max_compensation_epochs=25,
            compensation_lr=0.01,
            min_adaptation_steps=5,
            kl_aware_scaling=True,
            max_kl_scale=2.5,
            grad_clip=1.0,
        )

        self.meta_optimizer = MetaCycleOptimizer(
            initial_frequency=1.0 / 15,
            initial_compensation=1.0,
            performance_threshold=0.02,
            meta_update_interval=2,
            frequency_bounds=(1.0 / 100, 1.0 / 10),
            compensation_bounds=(0.5, 5.0),
            higher_is_better=False,
        )

        self.opt_a = torch.optim.Adam(self.model.syntax.parameters(), lr=1e-3)
        self.opt_b = torch.optim.Adam(self.model.semantics.parameters(), lr=1e-3)
        self.opt_classifier = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-3)

        self.loss_fn = nn.CrossEntropyLoss()

        self._global_step = 0
        self._reversal_count = 0
        self._reversal_log: List[Dict] = []

    def _full_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.model.forward_dual(x, self.dual_wrapper)
        return self.loss_fn(logits, y)

    def _loss_fn_wrapper(self, combined: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self.model.classifier(combined)
        return self.loss_fn(logits, target)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        self._global_step += 1
        info = {"step": self._global_step}

        should_reverse = self.scheduler.should_reverse(self._global_step)
        if should_reverse:
            reversal_info = self._execute_reversal(x, y)
            info["reversal"] = reversal_info
            self._reversal_log.append(reversal_info)

        self.opt_a.zero_grad()
        self.opt_b.zero_grad()
        self.opt_classifier.zero_grad()

        loss = self._full_loss(x, y)
        loss.backward()

        self.opt_a.step()
        self.opt_b.step()
        self.opt_classifier.step()

        self.compensator.update_ema_loss(loss.item())

        info["loss"] = loss.item()
        info["is_swapped"] = self.dual_wrapper.is_swapped
        return info

    def _execute_reversal(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        self._reversal_count += 1
        info = {
            "reversal_number": self._reversal_count,
            "step": self._global_step,
        }

        with torch.no_grad():
            out_a, out_b = self.dual_wrapper.forward_separate(x)
            combined = self.dual_wrapper(x)
            pre_logits = self.model.classifier(combined)
            pre_loss = self.loss_fn(pre_logits, y).item()

        self.compensator.capture_pre_reversal_state(out_a, out_b, combined, pre_loss)
        self.meta_optimizer.record_pre_reversal_metric(pre_loss)
        info["pre_reversal_loss"] = pre_loss

        self.dual_wrapper.swap_parameters()
        self.dual_wrapper.swap_optimizer_states(self.opt_a, self.opt_b)
        info["swapped"] = True

        with torch.no_grad():
            out_a_post, out_b_post = self.dual_wrapper.forward_separate(x)
            combined_post = self.dual_wrapper(x)

        kl_div = self.compensator.capture_post_reversal_state(out_a_post, out_b_post, combined_post)
        info["kl_divergence"] = kl_div

        with torch.no_grad():
            post_logits = self.model.classifier(combined_post)
            post_loss_before_adapt = self.loss_fn(post_logits, y).item()
        info["post_reversal_loss_before_adapt"] = post_loss_before_adapt

        loss_degradation = (post_loss_before_adapt - pre_loss) / (pre_loss + 1e-8)
        info["loss_degradation_pct"] = loss_degradation * 100

        adaptation_result = self.compensator.run_fast_adaptation(
            self.dual_wrapper, x, self._loss_fn_wrapper, y, n_steps=15
        )
        info["adaptation"] = adaptation_result
        info["post_reversal_loss_after_adapt"] = adaptation_result["final_loss"]

        effective_degradation = (adaptation_result["final_loss"] - pre_loss) / (pre_loss + 1e-8)
        info["effective_degradation_pct"] = max(0.0, effective_degradation) * 100

        self.meta_optimizer.record_post_reversal_metric(adaptation_result["final_loss"])

        self.scheduler.update_frequency(self.meta_optimizer.frequency)
        self.compensator.update_compensation_strength(self.meta_optimizer.compensation_strength)

        info["meta_frequency"] = self.meta_optimizer.frequency
        info["meta_compensation"] = self.meta_optimizer.compensation_strength
        info["meta_mode"] = self.meta_optimizer.get_adjustment_direction()

        return info

    def evaluate_robustness(self, dataloader) -> dict:
        self.model.eval()
        total_loss_normal = 0.0
        total_loss_swapped = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, y in dataloader:
                logits_normal = self.model.forward_dual(x, self.dual_wrapper)
                loss_normal = self.loss_fn(logits_normal, y).item()
                total_loss_normal += loss_normal

                saved_state = self.dual_wrapper._swapped
                self.dual_wrapper._swapped = not saved_state
                logits_swapped = self.model.forward_dual(x, self.dual_wrapper)
                loss_swapped = self.loss_fn(logits_swapped, y).item()
                total_loss_swapped += loss_swapped
                self.dual_wrapper._swapped = saved_state

                n_batches += 1

        self.model.train()

        avg_normal = total_loss_normal / max(n_batches, 1)
        avg_swapped = total_loss_swapped / max(n_batches, 1)

        if self.meta_optimizer.higher_is_better:
            if avg_normal <= 0:
                raw_change = 0.0
            else:
                raw_change = (avg_normal - avg_swapped) / avg_normal
        else:
            if avg_normal <= 0:
                raw_change = 0.0
            else:
                raw_change = (avg_swapped - avg_normal) / avg_normal

        performance_degradation = max(0.0, raw_change)
        is_robust = performance_degradation <= self.robustness_threshold

        if raw_change < -0.001:
            change_label = f"IMPROVED by {-raw_change*100:.2f}%"
        elif raw_change <= 0.001:
            change_label = "no change"
        else:
            change_label = f"DEGRADED by {raw_change*100:.2f}%"

        return {
            "avg_loss_normal": avg_normal,
            "avg_loss_swapped": avg_swapped,
            "raw_change": raw_change,
            "performance_degradation": performance_degradation,
            "performance_degradation_pct": performance_degradation * 100,
            "change_label": change_label,
            "is_robust": is_robust,
            "robustness_threshold_pct": self.robustness_threshold * 100,
            "supersymmetry_achieved": is_robust,
            "metric_direction": "lower_is_better (loss)" if not self.meta_optimizer.higher_is_better else "higher_is_better (accuracy)",
        }

    def train_epoch(self, dataloader, epoch: int = 0, verbose: bool = True) -> dict:
        self.model.train()
        losses = []
        reversals_this_epoch = 0

        for x, y in dataloader:
            info = self.train_step(x, y)
            losses.append(info["loss"])

            if "reversal" in info:
                reversals_this_epoch += 1
                if verbose:
                    r = info["reversal"]
                    a = r["adaptation"]
                    print(f"  [REVERSAL #{r['reversal_number']}] step={r['step']}")
                    print(f"    pre_loss={r['pre_reversal_loss']:.4f}  ->  post_loss={r['post_reversal_loss_before_adapt']:.4f}")
                    print(f"    KL divergence: {r['kl_divergence']:.6f}")
                    print(f"    Degradation before adapt: {r['loss_degradation_pct']:.2f}%")
                    print(f"    Adaptation: {a['adaptation_steps']} steps, "
                          f"loss: {a['initial_loss']:.4f} -> {a['final_loss']:.4f} "
                          f"(-{a['loss_reduction_pct']:.1f}%)")
                    print(f"    Effective degradation: {r['effective_degradation_pct']:.2f}%")
                    print(f"    Meta mode: {r['meta_mode']} | "
                          f"freq={r['meta_frequency']:.5f} | "
                          f"comp={r['meta_compensation']:.4f}")

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        return {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "reversals_this_epoch": reversals_this_epoch,
            "total_reversals": self._reversal_count,
            "total_steps": self._global_step,
            "current_frequency": self.meta_optimizer.frequency,
            "current_compensation": self.meta_optimizer.compensation_strength,
        }


def main():
    trainer = RobustTrainer()

    dataset = create_synthetic_data(n_samples=1000, input_dim=trainer.dim, n_classes=trainer.n_classes)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    eval_dataset = create_synthetic_data(n_samples=500, input_dim=trainer.dim, n_classes=trainer.n_classes)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    print("=" * 80)
    print("  Supersymmetric Training - Full Recovery Loop Demo")
    print("=" * 80)
    print()
    print(f"  Module arch: dim={trainer.dim} <-> hidden={trainer.hidden_dim}")
    print(f"  Classes: {trainer.n_classes}")
    print(f"  Robustness threshold: {trainer.robustness_threshold*100:.0f}%")
    print(f"  Higher is better: {trainer.meta_optimizer.higher_is_better}")
    print()

    base_epochs = 4
    max_recovery_epochs = 10
    recovery_count = 0
    converged = False

    print(f"--- Phase 1: Initial Training ({base_epochs} epochs) ---")
    print()

    for epoch in range(base_epochs):
        result = trainer.train_epoch(dataloader, epoch=epoch, verbose=True)
        robustness = trainer.evaluate_robustness(eval_dataloader)

        print(f"Epoch {epoch+1}/{base_epochs} Summary:")
        print(f"  Avg Loss:      {result['avg_loss']:.4f}")
        print(f"  Reversals:     {result['reversals_this_epoch']} (total: {result['total_reversals']})")
        print(f"  Normal loss:   {robustness['avg_loss_normal']:.4f}")
        print(f"  Swapped loss:  {robustness['avg_loss_swapped']:.4f}")
        print(f"  Change:        {robustness['change_label']}")
        print(f"  Degradation:   {robustness['performance_degradation_pct']:.2f}% "
              f"({'PASS' if robustness['is_robust'] else 'FAIL'}, threshold: {robustness['robustness_threshold_pct']:.0f}%)")
        print(f"  Meta freq:     {result['current_frequency']:.5f}")
        print(f"  Meta comp:     {result['current_compensation']:.4f}")
        print(f"  Meta mode:     {trainer.meta_optimizer.get_adjustment_direction()}")
        print()

    print(f"--- Phase 2: Recovery Training (max {max_recovery_epochs} epochs) ---")
    print()

    final_robustness = trainer.evaluate_robustness(eval_dataloader)

    if final_robustness["is_robust"]:
        print("Already robust! Skipping recovery phase.")
        converged = True
    else:
        for recovery_epoch in range(max_recovery_epochs):
            recovery_count += 1

            print(f"Recovery epoch {recovery_count}/{max_recovery_epochs}")
            print(f"  Current drop: {final_robustness['performance_degradation_pct']:.2f}% "
                  f"(target: {final_robustness['robustness_threshold_pct']:.0f}%)")
            print(f"  Meta mode: {trainer.meta_optimizer.get_adjustment_direction()}")

            current_drop = final_robustness["performance_degradation"]
            current_mode = trainer.meta_optimizer.get_adjustment_direction()

            if current_drop > trainer.robustness_threshold * 3:
                print("  Action: EMERGENCY BRAKE - slashing frequency, boosting compensation")
                freq_low = trainer.meta_optimizer.frequency_bounds[0]
                trainer.meta_optimizer.frequency = max(freq_low, trainer.meta_optimizer.frequency * 0.5)
                trainer.meta_optimizer.compensation_strength = min(
                    trainer.meta_optimizer.compensation_bounds[1],
                    trainer.meta_optimizer.compensation_strength * 1.5
                )
            elif current_drop > trainer.robustness_threshold * 1.5:
                print("  Action: CONSERVATIVE - lowering frequency, raising compensation")
                trainer.meta_optimizer.frequency = max(
                    trainer.meta_optimizer.frequency_bounds[0],
                    trainer.meta_optimizer.frequency * 0.8
                )
                trainer.meta_optimizer.compensation_strength = min(
                    trainer.meta_optimizer.compensation_bounds[1],
                    trainer.meta_optimizer.compensation_strength * 1.2
                )
            elif current_drop <= trainer.robustness_threshold * 0.5:
                print("  Action: EXPLORING - carefully increasing frequency")
                trainer.meta_optimizer.frequency = min(
                    trainer.meta_optimizer.frequency_bounds[1],
                    trainer.meta_optimizer.frequency * 1.1
                )

            trainer.scheduler.update_frequency(trainer.meta_optimizer.frequency)
            trainer.compensator.update_compensation_strength(trainer.meta_optimizer.compensation_strength)

            print(f"  New freq: {trainer.meta_optimizer.frequency:.5f} | "
                  f"New comp: {trainer.meta_optimizer.compensation_strength:.4f}")

            result = trainer.train_epoch(dataloader, epoch=base_epochs + recovery_epoch, verbose=False)
            final_robustness = trainer.evaluate_robustness(eval_dataloader)

            print(f"  After epoch: drop={final_robustness['performance_degradation_pct']:.2f}% "
                  f"({'PASS' if final_robustness['is_robust'] else 'FAIL'})")
            print()

            if final_robustness["is_robust"]:
                print(f"*** ROBUSTNESS ACHIEVED after {recovery_count} recovery epochs! ***")
                converged = True
                break

    print("=" * 80)
    print("  FINAL RESULTS")
    print("=" * 80)
    print(f"  Total steps:           {trainer._global_step}")
    print(f"  Total reversals:       {trainer._reversal_count}")
    print(f"  Recovery epochs:       {recovery_count}")
    print(f"  Converged:             {converged}")
    print()
    print(f"  Normal loss:           {final_robustness['avg_loss_normal']:.4f}")
    print(f"  Swapped loss:          {final_robustness['avg_loss_swapped']:.4f}")
    print(f"  Change:              {final_robustness['change_label']}")
    print(f"  Degradation:         {final_robustness['performance_degradation_pct']:.2f}%")
    print(f"  Threshold:             {final_robustness['robustness_threshold_pct']:.0f}%")
    print(f"  Supersymmetry:         {final_robustness['supersymmetry_achieved']}")
    print()
    print(f"  Final frequency:       {trainer.meta_optimizer.frequency:.5f}")
    print(f"  Final compensation:    {trainer.meta_optimizer.compensation_strength:.4f}")
    print(f"  Meta mode:             {trainer.meta_optimizer.get_adjustment_direction()}")
    print()

    if not converged:
        print("  *** DIAGNOSIS: Training did NOT converge to supersymmetry ***")
        print(f"  - Final drop ({final_robustness['performance_degradation_pct']:.1f}%) > "
              f"threshold ({final_robustness['robustness_threshold_pct']:.0f}%)")
        print(f"  - Total recovery epochs exhausted ({max_recovery_epochs})")
        last_mode = trainer.meta_optimizer.get_adjustment_direction()
        if last_mode in ["emergency_brake", "conservative"]:
            print(f"  - Meta mode: {last_mode} (system was still trying to stabilize)")
            print(f"  - Possible causes:")
            print(f"    1. Modules are too specialized to swap roles")
            print(f"    2. Compensation strength may be insufficient")
            print(f"    3. Reversal frequency may still be too high for this task")
            print(f"    4. Need more training data or stronger regularization")
        elif last_mode == "stable":
            print(f"  - Meta mode: {last_mode} (stabilized but not meeting threshold)")
            print(f"  - The system reached a stable state but below the 2% target")
        else:
            print(f"  - Meta mode: {last_mode}")
    else:
        print("  *** SUPERSYMMETRY ACHIEVED: Modules have developed swap-invariant representations ***")

    print()

    print("  Per-reversal summary (first 5 and last 5):")
    logs = trainer._reversal_log
    display_logs = logs[:5] + (["..."] if len(logs) > 10 else []) + logs[-5:] if len(logs) > 5 else logs
    for i, r in enumerate(display_logs):
        if isinstance(r, str):
            print(f"    {r}")
            continue
        a = r["adaptation"]
        print(f"    #{r['reversal_number']}: step={r['step']}  "
              f"KL={r['kl_divergence']:.4f}  "
              f"degrad={r['effective_degradation_pct']:.1f}%  "
              f"adapt_steps={a['adaptation_steps']}  "
              f"loss_reduct={a['loss_reduction_pct']:.1f}%  "
              f"mode={r['meta_mode']}")

    print()


if __name__ == "__main__":
    main()
