"""
Functional API for EvoOptimizer.

Provides a stateless functional interface for use with custom training loops
or torch.compile().
"""

from typing import List, Optional
import torch
from torch import Tensor


@torch.no_grad()
def evo_optimizer_step(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    *,
    lr: float,
    beta1: float = 0.8553,
    beta2: float = 0.9358,
    eps: float = 5.4e-9,
    weight_decay: float = 9.7e-4,
    alpha_sign: float = 0.7345,
    alpha_adam: float = 3.6352,
) -> None:
    r"""
    Functional API for EvoOptimizer step.

    This can be used with torch.compile() for additional speedup.

    Args:
        params: List of parameter tensors
        grads: List of gradient tensors (same order as params)
        exp_avgs: List of first moment buffers (same order as params)
        exp_avg_sqs: List of second moment buffers (same order as params)
        lr: Learning rate (with any scheduling already applied)
        beta1: Exponential decay rate for first moment (default: 0.8553)
        beta2: Exponential decay rate for second moment (default: 0.9358)
        eps: Numerical stability term (default: 5.4e-9)
        weight_decay: Decoupled weight decay (default: 9.7e-4)
        alpha_sign: Weight for sign-based gradient term (default: 0.7345)
        alpha_adam: Weight for Adam-style adaptive term (default: 3.6352)

    Example::

        >>> # Manual state management
        >>> exp_avgs = [torch.zeros_like(p) for p in model.parameters()]
        >>> exp_avg_sqs = [torch.zeros_like(p) for p in model.parameters()]
        >>> 
        >>> for step in range(num_steps):
        ...     loss = model(x).sum()
        ...     loss.backward()
        ...     
        ...     params = list(model.parameters())
        ...     grads = [p.grad for p in params]
        ...     
        ...     # Compute scheduled LR
        ...     lr = base_lr * get_lr_scale(step, warmup_steps, total_steps)
        ...     
        ...     evo_optimizer_step(
        ...         params, grads, exp_avgs, exp_avg_sqs,
        ...         lr=lr, weight_decay=1e-4
        ...     )
        ...     
        ...     for p in params:
        ...         p.grad = None

    Note:
        Scheduling (warmup, cosine decay) should be applied to `lr` before
        calling this function. Use `compute_lr_scale()` helper if needed.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]

        # Update moments (in-place)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Compute update
        denom = exp_avg_sq.sqrt().add_(eps)
        update = alpha_sign * grad.sign() + alpha_adam * exp_avg / denom

        # Apply weight decay and update
        if weight_decay != 0:
            param.add_(param, alpha=-lr * weight_decay)
        param.add_(update, alpha=-lr)


@torch.compile
def evo_optimizer_step_compiled(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    *,
    lr: float,
    beta1: float = 0.8553,
    beta2: float = 0.9358,
    eps: float = 5.4e-9,
    weight_decay: float = 9.7e-4,
    alpha_sign: float = 0.7345,
    alpha_adam: float = 3.6352,
) -> None:
    """
    Compiled version of evo_optimizer_step using torch.compile().

    Same interface as evo_optimizer_step but with automatic kernel fusion.
    Requires PyTorch 2.0+.
    """
    evo_optimizer_step(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        alpha_sign=alpha_sign,
        alpha_adam=alpha_adam,
    )


def compute_lr_scale(
    step: int,
    warmup_steps: int = 100,
    total_steps: Optional[int] = None,
    use_warmup: bool = True,
    use_cosine_decay: bool = True,
) -> float:
    """
    Compute the learning rate multiplier for a given step.

    Args:
        step: Current training step (0-indexed)
        warmup_steps: Number of warmup steps (default: 100)
        total_steps: Total training steps. If None, cosine decay is disabled.
        use_warmup: Whether to apply linear warmup (default: True)
        use_cosine_decay: Whether to apply cosine decay after warmup (default: True)

    Returns:
        Learning rate multiplier in [0, 1]

    Example::

        >>> base_lr = 1.2e-3
        >>> for step in range(10000):
        ...     scale = compute_lr_scale(step, warmup_steps=100, total_steps=10000)
        ...     current_lr = base_lr * scale
    """
    import math

    scale = 1.0

    # Linear warmup
    if use_warmup and step < warmup_steps:
        scale = (step + 1) / warmup_steps

    # Cosine decay after warmup
    elif use_cosine_decay and total_steps is not None:
        warmup = warmup_steps if use_warmup else 0
        if step >= warmup:
            progress = (step - warmup) / max(1, total_steps - warmup)
            progress = min(1.0, progress)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

    return scale
