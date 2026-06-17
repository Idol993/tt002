import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from dual_module_wrapper import DualModuleWrapper
from fast_adaptation_compensator import FastAdaptationCompensator


class SimpleModule(nn.Module):
    def __init__(self, dim: int, init_scale: float = 1.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
        with torch.no_grad():
            for p in self.parameters():
                p.mul_(init_scale)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_spike_suppression():
    torch.manual_seed(42)
    dim = 64
    batch_size = 32

    print("=" * 70)
    print("  Loss Spike Suppression Test")
    print("  (verify compensation reduces loss spikes from reversal)")
    print("=" * 70)
    print()

    module_a = SimpleModule(dim, init_scale=0.5)
    module_b = SimpleModule(dim, init_scale=2.0)

    wrapper = DualModuleWrapper(module_a, module_b)

    compensator = FastAdaptationCompensator(
        compensation_strength=1.5,
        kl_temperature=1.0,
        max_compensation_epochs=50,
        compensation_lr=0.005,
        min_adaptation_steps=5,
        kl_aware_scaling=True,
        max_kl_scale=3.0,
        grad_clip=1.0,
    )

    x = torch.randn(batch_size, dim)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        pre_a = wrapper.module_a(x)
        pre_b = wrapper.module_b(x)
        pre_combined = wrapper(x)

    target = pre_combined.detach().clone() + 0.1 * torch.randn_like(pre_combined)

    pre_loss = loss_fn(pre_combined, target).item()
    compensator.capture_pre_reversal_state(pre_a, pre_b, pre_combined, pre_loss)

    wrapper.swap_parameters()

    with torch.no_grad():
        post_a = wrapper.module_a(x)
        post_b = wrapper.module_b(x)
        post_combined = wrapper(x)

    post_loss = loss_fn(post_combined, target).item()
    kl = compensator.capture_post_reversal_state(post_a, post_b, post_combined)

    print("Before compensation:")
    print(f"  Pre-reversal loss:  {pre_loss:.4f}")
    print(f"  Post-reversal loss: {post_loss:.4f}")
    print(f"  Loss spike:         {(post_loss - pre_loss)/pre_loss*100:+.2f}%")
    print(f"  KL divergence:      {kl:.4f}")
    print()

    adapt_result = compensator.run_fast_adaptation(wrapper, x, loss_fn, target)

    with torch.no_grad():
        final_combined = wrapper(x)
        final_loss = loss_fn(final_combined, target).item()

    spike_reduction_abs = post_loss - final_loss
    spike_reduction_pct = spike_reduction_abs / (post_loss - pre_loss + 1e-8) * 100 if post_loss > pre_loss else 0

    print("After compensation:")
    print(f"  Final loss:         {final_loss:.4f}")
    print(f"  Loss reduced by:    {spike_reduction_abs:.4f} ({spike_reduction_pct:.1f}% of spike)")
    print(f"  Adaptation steps:   {adapt_result['adaptation_steps']}")
    print(f"  Initial KL:         {adapt_result['initial_kl']:.4f}")
    print(f"  Final KL:           {adapt_result['final_kl']:.4f}")
    print(f"  KL reduction:       {adapt_result['kl_reduction']:.4f} ({adapt_result['kl_reduction']/(adapt_result['initial_kl']+1e-8)*100:.1f}%)")
    print()

    if post_loss > pre_loss and final_loss < post_loss:
        print("  ✅ PASS: Compensation successfully reduced the loss spike")
    elif post_loss <= pre_loss:
        print("  ⚠️  NOTE: No loss spike to suppress (post-loss < pre-loss)")
    else:
        print("  ❌ FAIL: Compensation did not reduce the loss spike")

    print()
    print("=" * 70)
    print("  Test 2: KL-aware vs Uniform Compensation")
    print("=" * 70)
    print()

    scenarios = [
        ("Small spike (scale=0.8)", 0.8, 1.3),
        ("Medium spike (scale=0.5)", 0.5, 2.0),
        ("Large spike (scale=0.3)", 0.3, 3.0),
    ]

    print(f"{'Scenario':<30} {'Spike':>8} {'KL':>8} {'KL-aware':>10} {'Uniform':>10}")
    print("-" * 70)

    for name, scale_a, scale_b in scenarios:
        results = {}
        for mode_name, kl_aware in [("kl_aware", True), ("uniform", False)]:
            torch.manual_seed(99)
            a = SimpleModule(dim, init_scale=scale_a)
            b = SimpleModule(dim, init_scale=scale_b)
            w = DualModuleWrapper(a, b)

            c = FastAdaptationCompensator(
                compensation_strength=1.0,
                kl_temperature=1.0,
                max_compensation_epochs=40,
                compensation_lr=0.005,
                min_adaptation_steps=5,
                kl_aware_scaling=kl_aware,
                max_kl_scale=5.0,
                grad_clip=2.0,
            )

            x_t = torch.randn(batch_size, dim)
            t_t = torch.randn(batch_size, dim) * 0.5

            with torch.no_grad():
                pre_a = w.module_a(x_t)
                pre_b = w.module_b(x_t)
                pre_c = w(x_t)
            pre_l = loss_fn(pre_c, t_t).item()
            c.capture_pre_reversal_state(pre_a, pre_b, pre_c, pre_l)

            w.swap_parameters()

            with torch.no_grad():
                post_c = w(x_t)
                post_a = w.module_a(x_t)
                post_b = w.module_b(x_t)
            post_l = loss_fn(post_c, t_t).item()
            kl_val = c.capture_post_reversal_state(post_a, post_b, post_c)

            r = c.run_fast_adaptation(w, x_t, loss_fn, t_t)
            spike_red = (post_l - r['final_loss']) / max(post_l - pre_l, 1e-8) * 100 if post_l > pre_l else 0

            results[mode_name] = {
                'spike_pct': (post_l - pre_l) / pre_l * 100,
                'kl': kl_val,
                'spike_reduction': spike_red,
            }

        ka = results['kl_aware']
        print(f"  {name:<28} {ka['spike_pct']:>+7.1f}% {ka['kl']:>8.2f} "
              f"{ka['spike_reduction']:>9.1f}% {results['uniform']['spike_reduction']:>9.1f}%")

    print()
    print("Explanation:")
    print("  - Compensation gradient comes from KL alignment loss")
    print("  - KL-aware mode: stronger compensation when distribution shift is larger")
    print("  - Uniform mode: same compensation strength regardless of KL")
    print("  - Both modes aim to align output distribution with pre-reversal state")
    print()


if __name__ == "__main__":
    test_spike_suppression()
