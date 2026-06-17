import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tt002 import DualModuleWrapper, FastAdaptationCompensator


class TestModule(nn.Module):
    def __init__(self, dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(123)
    dim = 32
    batch_size = 64

    module_a = TestModule(dim, dim * 2)
    module_b = TestModule(dim, dim * 2)

    with torch.no_grad():
        for param in module_b.parameters():
            param.add_(torch.randn_like(param) * 0.5)

    wrapper = DualModuleWrapper(module_a, module_b)

    compensator = FastAdaptationCompensator(
        compensation_strength=5.0,
        kl_temperature=1.0,
        max_compensation_epochs=50,
        compensation_lr=0.05,
        min_adaptation_steps=5,
        kl_aware_scaling=True,
    )

    x = torch.randn(batch_size, dim)

    print("=" * 70)
    print("  KL Compensation Validation Test")
    print("=" * 70)
    print()

    with torch.no_grad():
        out_a_pre, out_b_pre = wrapper.forward_separate(x)
        combined_pre = wrapper(x)

    dummy_loss_pre = combined_pre.mean().item()
    compensator.capture_pre_reversal_state(out_a_pre, out_b_pre, combined_pre, dummy_loss_pre)

    print(f"Before swap:")
    print(f"  Module A output mean: {out_a_pre.mean().item():.4f}, std: {out_a_pre.std().item():.4f}")
    print(f"  Module B output mean: {out_b_pre.mean().item():.4f}, std: {out_b_pre.std().item():.4f}")
    print(f"  Combined output mean: {combined_pre.mean().item():.4f}, std: {combined_pre.std().item():.4f}")
    print()

    wrapper.swap_parameters()

    with torch.no_grad():
        out_a_post, out_b_post = wrapper.forward_separate(x)
        combined_post = wrapper(x)

    kl_div = compensator.capture_post_reversal_state(out_a_post, out_b_post, combined_post)

    print(f"After swap (before adaptation):")
    print(f"  Module A output mean: {out_a_post.mean().item():.4f}, std: {out_a_post.std().item():.4f}")
    print(f"  Module B output mean: {out_b_post.mean().item():.4f}, std: {out_b_post.std().item():.4f}")
    print(f"  Combined output mean: {combined_post.mean().item():.4f}, std: {combined_post.std().item():.4f}")
    print(f"  KL divergence:       {kl_div:.6f}")
    print()

    def dummy_loss_fn(output, target=None):
        return ((output - combined_pre.detach()) ** 2).mean()

    print(f"Running fast adaptation ({compensator.max_compensation_epochs} steps max)...")
    print()

    result = compensator.run_fast_adaptation(
        wrapper, x, dummy_loss_fn, None, n_steps=50
    )

    with torch.no_grad():
        out_a_final, out_b_final = wrapper.forward_separate(x)
        combined_final = wrapper(x)

    final_kl = 0.5 * (
        ((combined_pre.std() ** 2) / (combined_final.std() ** 2 + 1e-8)).log().mean()
        + ((combined_final.std() ** 2) / (combined_pre.std() ** 2 + 1e-8)).mean()
        + ((combined_pre.mean() - combined_final.mean()) ** 2 / (combined_pre.std() ** 2 + 1e-8)).mean()
        - 1
    ).item()

    print(f"After adaptation:")
    print(f"  Combined output mean: {combined_final.mean().item():.4f}, std: {combined_final.std().item():.4f}")
    print(f"  Target mean:          {combined_pre.mean().item():.4f}, std: {combined_pre.std().item():.4f}")
    print(f"  Final KL to target:   {max(0, final_kl):.6f}")
    print()
    print(f"Adaptation stats:")
    print(f"  Steps:               {result['adaptation_steps']}")
    print(f"  Initial loss:        {result['initial_loss']:.6f}")
    print(f"  Final loss:          {result['final_loss']:.6f}")
    print(f"  Loss reduction:      {result['loss_reduction']:.6f} ({result['loss_reduction_pct']:.1f}%)")
    print(f"  Initial KL:          {result['initial_kl']:.6f}")
    print(f"  Final KL:            {result['final_kl']:.6f}")
    print(f"  KL reduction:        {result['kl_reduction']:.6f}")
    print()

    kl_improvement = (kl_div - max(0, final_kl)) / (kl_div + 1e-8) * 100
    print(f"  KL improvement:      {kl_improvement:.1f}%")
    print()

    if kl_improvement > 10:
        print("  ✅ PASS: KL divergence was significantly reduced by compensation")
    else:
        print("  ⚠️  WARNING: KL reduction is small - check compensation strength")
    print()

    print(f"  Compensation strength used: {result['compensation_strength_used']:.2f}")
    print(f"  KL-aware scaling:           {compensator.kl_aware_scaling}")
    print()

    print("=" * 70)
    print("  Test 2: Larger distribution shift")
    print("=" * 70)
    print()

    module_c = TestModule(dim, dim * 2)
    with torch.no_grad():
        for param in module_c.parameters():
            param.copy_(torch.randn_like(param) * 2.0)

    wrapper2 = DualModuleWrapper(module_a, module_c)
    compensator2 = FastAdaptationCompensator(
        compensation_strength=8.0,
        kl_temperature=1.0,
        max_compensation_epochs=100,
        compensation_lr=0.02,
        min_adaptation_steps=10,
        kl_aware_scaling=True,
    )

    with torch.no_grad():
        out_a2, out_c2 = wrapper2.forward_separate(x)
        combined2 = wrapper2(x)
    compensator2.capture_pre_reversal_state(out_a2, out_c2, combined2, 1.0)

    pre_std = combined2.std().item()
    pre_mean = combined2.mean().item()

    wrapper2.swap_parameters()

    with torch.no_grad():
        out_a2_post, out_c2_post = wrapper2.forward_separate(x)
        combined2_post = wrapper2(x)

    kl2 = compensator2.capture_post_reversal_state(out_a2_post, out_c2_post, combined2_post)

    print(f"Pre-swap:  mean={pre_mean:.4f}, std={pre_std:.4f}")
    print(f"Post-swap: mean={combined2_post.mean().item():.4f}, std={combined2_post.std().item():.4f}")
    print(f"KL divergence: {kl2:.4f}")
    print()

    result2 = compensator2.run_fast_adaptation(wrapper2, x, dummy_loss_fn, None, n_steps=100)

    with torch.no_grad():
        combined2_final = wrapper2(x)

    print(f"After adaptation:")
    print(f"  mean={combined2_final.mean().item():.4f}, std={combined2_final.std().item():.4f}")
    print(f"  target mean={pre_mean:.4f}, target std={pre_std:.4f}")
    print(f"  Initial KL:  {kl2:.4f}")
    print(f"  Final KL:    {result2['final_kl']:.4f}")
    print(f"  KL reduced by: {((kl2 - result2['final_kl']) / (kl2 + 1e-8) * 100):.1f}%")
    print()

    if result2['loss_reduction_pct'] > 20:
        print("  ✅ PASS: Large shift also shows significant compensation effect")
    else:
        print("  ⚠️  NOTE: Compensation effect may be limited for extreme shifts")
    print()


if __name__ == "__main__":
    main()
