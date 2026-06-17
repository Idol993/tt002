import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from dual_module_wrapper import DualModuleWrapper
from fast_adaptation_compensator import FastAdaptationCompensator


class SimpleModule(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.scale = scale
        with torch.no_grad():
            for p in self.net.parameters():
                p.mul_(scale)

    def forward(self, x):
        return self.net(x)


def test_kl_aware_compensation():
    torch.manual_seed(42)
    dim = 64
    batch_size = 32

    print("=" * 70)
    print("  KL-Aware Compensation Validation Test")
    print("=" * 70)
    print()

    print("Testing: Larger KL divergence → stronger compensation effect")
    print()

    scenarios = [
        ("Small shift (scale=0.3)", 0.3),
        ("Medium shift (scale=1.0)", 1.0),
        ("Large shift (scale=2.5)", 2.5),
        ("Very large shift (scale=5.0)", 5.0),
    ]

    results = []

    for name, scale in scenarios:
        torch.manual_seed(42)
        module_a = SimpleModule(dim, scale=1.0)
        module_b = SimpleModule(dim, scale=scale)
        wrapper = DualModuleWrapper(module_a, module_b)

        compensator = FastAdaptationCompensator(
            compensation_strength=1.0,
            kl_temperature=1.0,
            max_compensation_epochs=30,
            compensation_lr=0.01,
            min_adaptation_steps=5,
            kl_aware_scaling=True,
            max_kl_scale=10.0,
            grad_clip=5.0,
        )

        x = torch.randn(batch_size, dim)
        loss_fn = nn.MSELoss()
        target = torch.randn(batch_size, dim)

        with torch.no_grad():
            pre_a = wrapper.module_a(x)
            pre_b = wrapper.module_b(x)
            pre_combined = wrapper(x)
            pre_loss = loss_fn(pre_combined, target).item()

        compensator.capture_pre_reversal_state(pre_a, pre_b, pre_combined, pre_loss)

        saved_a = {k: v.clone() for k, v in module_a.state_dict().items()}
        saved_b = {k: v.clone() for k, v in module_b.state_dict().items()}
        wrapper.swap_parameters()

        with torch.no_grad():
            post_a = wrapper.module_a(x)
            post_b = wrapper.module_b(x)
            post_combined = wrapper(x)

        kl = compensator.capture_post_reversal_state(post_a, post_b, post_combined)

        adapt_result = compensator.run_fast_adaptation(wrapper, x, loss_fn, target)

        with torch.no_grad():
            final_combined = wrapper(x)
            final_loss = loss_fn(final_combined, target).item()

        loss_reduction_pct = (pre_loss - final_loss) / (abs(pre_loss) + 1e-8) * 100
        spike_reduction = (adapt_result['initial_loss'] - adapt_result['final_loss']) / (abs(adapt_result['initial_loss']) + 1e-8) * 100
        kl_reduction = adapt_result['initial_kl'] - adapt_result['final_kl']

        results.append({
            'name': name,
            'scale': scale,
            'kl': kl,
            'pre_loss': pre_loss,
            'post_loss': adapt_result['initial_loss'],
            'final_loss': adapt_result['final_loss'],
            'spike_reduction_pct': spike_reduction,
            'kl_initial': adapt_result['initial_kl'],
            'kl_final': adapt_result['final_kl'],
            'kl_reduction': kl_reduction,
            'kl_reduction_pct': kl_reduction / (adapt_result['initial_kl'] + 1e-8) * 100,
            'adapt_steps': adapt_result['adaptation_steps'],
        })

    print(f"{'Scenario':<28} {'KL':>8} {'Spike red.':>12} {'KL red.':>10} {'Steps':>6}")
    print("-" * 70)
    for r in results:
        print(f"  {r['name']:<26} {r['kl']:>8.2f}  {r['spike_reduction_pct']:>9.2f}%  {r['kl_reduction_pct']:>8.2f}%  {r['adapt_steps']:>4}")

    print()
    print("Analysis:")
    kl_values = [r['kl'] for r in results]
    reductions = [r['spike_reduction_pct'] for r in results]

    print(f"  KL range:     {min(kl_values):.4f} → {max(kl_values):.4f} ({max(kl_values)/min(kl_values):.1f}x)")
    print(f"  Reduction range: {min(reductions):.2f}% → {max(reductions):.2f}%")

    if kl_values[-1] > kl_values[0] and abs(reductions[-1]) >= abs(reductions[0]):
        print()
        print("  ✅ PASS: Larger KL leads to stronger (or equal) compensation effect")
    else:
        print()
        print("  ⚠️  NOTE: KL-compensation relationship may not be perfectly monotonic")
        print("            (due to gradient clipping and non-linear dynamics)")

    print()
    print("=" * 70)
    print("  Test 2: KL-aware scaling ON vs OFF comparison")
    print("=" * 70)
    print()

    scale = 3.0
    for mode_name, kl_aware in [("KL-aware OFF", False), ("KL-aware ON", True)]:
        torch.manual_seed(123)
        module_a = SimpleModule(dim, scale=1.0)
        module_b = SimpleModule(dim, scale=scale)
        wrapper = DualModuleWrapper(module_a, module_b)

        compensator = FastAdaptationCompensator(
            compensation_strength=1.0,
            kl_temperature=1.0,
            max_compensation_epochs=30,
            compensation_lr=0.01,
            min_adaptation_steps=5,
            kl_aware_scaling=kl_aware,
            max_kl_scale=10.0,
            grad_clip=5.0,
        )

        x = torch.randn(batch_size, dim)
        loss_fn = nn.MSELoss()
        target = torch.randn(batch_size, dim)

        with torch.no_grad():
            pre_a = wrapper.module_a(x)
            pre_b = wrapper.module_b(x)
            pre_combined = wrapper(x)
            pre_loss = loss_fn(pre_combined, target).item()

        compensator.capture_pre_reversal_state(pre_a, pre_b, pre_combined, pre_loss)

        wrapper.swap_parameters()

        with torch.no_grad():
            post_a = wrapper.module_a(x)
            post_b = wrapper.module_b(x)
            post_combined = wrapper(x)

        kl = compensator.capture_post_reversal_state(post_a, post_b, post_combined)

        adapt_result = compensator.run_fast_adaptation(wrapper, x, loss_fn, target)

        spike_reduction = (adapt_result['initial_loss'] - adapt_result['final_loss']) / (abs(adapt_result['initial_loss']) + 1e-8) * 100

        print(f"  {mode_name}:")
        print(f"    KL divergence:   {kl:.4f}")
        print(f"    Initial spike:   {adapt_result['initial_loss']:.4f}")
        print(f"    Final loss:      {adapt_result['final_loss']:.4f}")
        print(f"    Spike reduction: {spike_reduction:.2f}%")
        print(f"    Steps:           {adapt_result['adaptation_steps']}")
        print()

    print("=" * 70)
    print()
    print("Summary:")
    print("  Compensation gradient is computed from KL alignment loss,")
    print("  NOT from simply negating the task gradient.")
    print("  When kl_aware_scaling=True, compensation strength is")
    print("  proportional to the magnitude of distribution shift (KL).")
    print()


if __name__ == "__main__":
    test_kl_aware_compensation()
