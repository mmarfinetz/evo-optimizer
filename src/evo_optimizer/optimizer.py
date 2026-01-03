"""
EvoOptimizer: Evolved Deep Learning Optimizer

The evolved optimizer combines sign-based gradient terms with adaptive moment
estimation, using lower momentum coefficients than Adam (β₁=0.8553, β₂=0.9358),
and notably disables bias correction while enabling learning rate warmup and
cosine decay.

Discovered via genetic algorithm search over 50 generations, evaluated across
Fashion-MNIST, CIFAR-10, and MNIST. Outperforms Adam by 2.6% in aggregate
fitness and achieves 7.7% relative improvement on CIFAR-10.

Reference:
    Marfinetz, M. (2025). Evolving Deep Learning Optimizers. arXiv:2512.11853
"""

import math
from typing import Optional, Callable, Iterable, Tuple, List, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class EvoOptimizer(Optimizer):
    r"""
    Implements the Evolved Optimizer from arXiv:2512.11853.

    The update rule combines sign-based and adaptive moment terms:

    .. math::
        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
        \Delta\theta_t &= \alpha_{sign} \cdot \text{sign}(g_t) + \alpha_{adam} \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}

    Key differences from Adam:
        - Combines sign-based updates (like Lion) with adaptive moments
        - Lower momentum coefficients (β₁=0.8553, β₂=0.9358 vs 0.9/0.999)
        - No bias correction
        - Built-in warmup and cosine decay scheduling

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1.2e-3)
        betas: Coefficients for computing running averages of gradient and its square
            (default: (0.8553, 0.9358))
        eps: Term added to denominator for numerical stability (default: 5.4e-9)
        weight_decay: Decoupled weight decay coefficient (default: 9.7e-4)
        warmup_steps: Number of linear warmup steps (default: 100)
        total_steps: Total training steps for cosine decay. If None, decay is disabled
            (default: None)
        use_warmup: Whether to apply linear warmup (default: True)
        use_cosine_decay: Whether to apply cosine annealing after warmup (default: True)
        alpha_sign: Weight for sign-based gradient term (default: 0.7345)
        alpha_adam: Weight for Adam-style adaptive term (default: 3.6352)

    Example::

        >>> optimizer = EvoOptimizer(model.parameters(), lr=1e-3, total_steps=10000)
        >>> for step, (inputs, targets) in enumerate(dataloader):
        ...     optimizer.zero_grad()
        ...     loss = loss_fn(model(inputs), targets)
        ...     loss.backward()
        ...     optimizer.step(step=step)

    Note:
        The `step` parameter in the step() method is required for proper scheduling.
        If not provided, warmup and cosine decay are disabled for that step.

    .. _arXiv:2512.11853:
        https://arxiv.org/abs/2512.11853
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1.2e-3,
        betas: Tuple[float, float] = (0.8553, 0.9358),
        eps: float = 5.4e-9,
        weight_decay: float = 9.7e-4,
        warmup_steps: int = 100,
        total_steps: Optional[int] = None,
        use_warmup: bool = True,
        use_cosine_decay: bool = True,
        alpha_sign: float = 0.7345,
        alpha_adam: float = 3.6352,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            use_warmup=use_warmup,
            use_cosine_decay=use_cosine_decay,
            alpha_sign=alpha_sign,
            alpha_adam=alpha_adam,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("use_warmup", True)
            group.setdefault("use_cosine_decay", True)

    def _get_lr_scale(self, step: int, group: dict) -> float:
        """Compute learning rate multiplier for warmup and cosine decay."""
        scale = 1.0

        # Linear warmup
        if group["use_warmup"] and step < group["warmup_steps"]:
            scale = (step + 1) / group["warmup_steps"]

        # Cosine decay after warmup
        elif group["use_cosine_decay"] and group["total_steps"] is not None:
            warmup = group["warmup_steps"] if group["use_warmup"] else 0
            if step >= warmup:
                progress = (step - warmup) / max(1, group["total_steps"] - warmup)
                progress = min(1.0, progress)
                scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        return scale

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        step: Optional[int] = None,
    ) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
            step: Current training step (required for warmup/cosine decay scheduling).
                If None, scheduling is disabled for this step.

        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            alpha_sign = group["alpha_sign"]
            alpha_adam = group["alpha_adam"]

            # Apply LR schedule if step is provided
            if step is not None:
                lr_scale = self._get_lr_scale(step, group)
                lr = lr * lr_scale

            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

            _single_tensor_evo(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
                alpha_sign=alpha_sign,
                alpha_adam=alpha_adam,
            )

        return loss


class EvoOptimizerSimplified(EvoOptimizer):
    """
    Simplified alias for EvoOptimizer with identical functionality.

    This class exists for backwards compatibility and clarity.
    See EvoOptimizer for full documentation.
    """

    pass


def _single_tensor_evo(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    alpha_sign: float,
    alpha_adam: float,
):
    """Single tensor implementation of EvoOptimizer step."""
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]

        if grad.is_sparse:
            raise RuntimeError("EvoOptimizer does not support sparse gradients")

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Compute denominator (no bias correction)
        denom = exp_avg_sq.sqrt().add_(eps)

        # Combined update: sign term + adaptive term
        update = alpha_sign * grad.sign() + alpha_adam * exp_avg / denom

        # Decoupled weight decay
        if weight_decay != 0:
            param.add_(param, alpha=-lr * weight_decay)

        # Apply update
        param.add_(update, alpha=-lr)
