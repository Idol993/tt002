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

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(42)

    dim = 64
    hidden_dim = 128

    module_a = IsomorphicModule(dim, hidden_dim)
    module_b = IsomorphicModule(dim, hidden_dim)

    dual_wrapper = DualModuleWrapper(module_a, module_b, name_a="module_a", name_b="module_b")

    scheduler = ChaoticTrainingScheduler(
        seed=42,
        base_interval=15,
        min_interval=5,
        max_interval=40,
        chaos_strength=0.3,
        warmup_steps=20,
    )

    compensator = FastAdaptationCompensator(
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

    meta_optimizer = MetaCycleOptimizer(
        initial_frequency=1.0 / 15,
        initial_compensation=1.0,
        performance_threshold=0.02,
        meta_update_interval=2,
        frequency_bounds=(1.0 / 100, 1.0 / 10),
        compensation_bounds=(0.5, 5.0),
        higher_is_better=False,
    )

    opt_a = torch.optim.Adam(module_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(module_b.parameters(), lr=1e-3)

    loss_fn = nn.MSELoss()

    trainer = SupersymmetricTrainer(
        dual_wrapper=dual_wrapper,
        scheduler=scheduler,
        compensator=compensator,
        meta_optimizer=meta_optimizer,
        base_optimizer_a=opt_a,
        base_optimizer_b=opt_b,
        loss_fn=loss_fn,
        robustness_threshold=0.02,
        higher_is_better=False,
        adaptation_steps_per_reversal=15,
    )

    print("=" * 70)
    print("  SupersymmetricTrainer - Validation Script")
    print("=" * 70)
    print()
    print(f"  Module arch: dim={dim} <-> hidden={hidden_dim}")
    print(f"  Robustness threshold: 2%")
    print(f"  Higher is better: False (loss)")
    print(f"  Adaptation steps per reversal: 15")
    print()

    X = torch.randn(800, dim)
    Y = torch.randn(800, dim) * 0.5
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    X_eval = torch.randn(200, dim)
    Y_eval = torch.randn(200, dim) * 0.5
    eval_dataset = TensorDataset(X_eval, Y_eval)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    print("--- Training for 4 epochs ---")
    print()

    for epoch in range(4):
        result = trainer.train_epoch(dataloader, epoch=epoch, verbose=True)
        robustness = trainer.evaluate_robustness(eval_dataloader)

        print(f"Epoch {epoch+1}/4 Summary:")
        print(f"  Avg Loss:      {result['avg_loss']:.4f}")
        print(f"  Reversals:     {result['reversals_this_epoch']} (total: {result['total_reversals']})")
        print(f"  Normal loss:   {robustness['avg_loss_normal']:.4f}")
        print(f"  Swapped loss:  {robustness['avg_loss_swapped']:.4f}")
        print(f"  Change:        {robustness['change_label']}")
        print(f"  Degradation:   {robustness['performance_degradation_pct']:.2f}% "
              f"({'PASS' if robustness['is_robust'] else 'FAIL'}, "
              f"threshold: {robustness['robustness_threshold_pct']:.0f}%)")
        print(f"  Meta freq:     {result['current_frequency']:.5f}")
        print(f"  Meta comp:     {result['current_compensation']:.4f}")
        print(f"  Meta mode:     {trainer.meta_optimizer.get_adjustment_direction()}")
        print()

    final_robustness = trainer.evaluate_robustness(eval_dataloader)

    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    summary = trainer.get_training_summary()
    print(f"  Total steps:           {summary['total_steps']}")
    print(f"  Total reversals:       {summary['total_reversals']}")
    print(f"  Degradation:           {final_robustness['performance_degradation_pct']:.2f}%")
    print(f"  Change:                {final_robustness['change_label']}")
    print(f"  Metric direction:      {final_robustness['metric_direction']}")
    print(f"  Is robust:             {final_robustness['is_robust']}")
    print()

    print("  Per-reversal details:")
    for r in summary["reversal_log"]:
        a = r["adaptation"]
        print(f"    #{r['reversal_number']}: step={r['step']}  "
              f"KL={r['kl_divergence']:.4f}  "
              f"degrad={r['effective_degradation_pct']:.1f}%  "
              f"adapt_steps={a['adaptation_steps']}  "
              f"loss_red={a['loss_reduction_pct']:.1f}%  "
              f"mode={r['meta_mode']}")

    print()

    if final_robustness["is_robust"]:
        print("  *** SUPERSYMMETRY ACHIEVED ***")
    else:
        print(f"  Degradation ({final_robustness['performance_degradation_pct']:.2f}%) "
              f"exceeds threshold ({final_robustness['robustness_threshold_pct']:.0f}%)")
        mode = trainer.meta_optimizer.get_adjustment_direction()
        print(f"  Current meta mode: {mode}")
        print("  Would enter recovery training in a full training loop.")

    print()


if __name__ == "__main__":
    main()
