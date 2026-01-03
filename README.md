# EvoOptimizer

[![PyPI version](https://badge.fury.io/py/evo-optimizer.svg)](https://badge.fury.io/py/evo-optimizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2512.11853-b31b1b.svg)](https://arxiv.org/abs/2512.11853)

An evolved deep learning optimizer discovered via genetic algorithm search. Outperforms Adam by 2.6% in aggregate fitness and achieves 7.7% relative improvement on CIFAR-10.

## Installation

```bash
pip install evo-optimizer
```

## Quick Start

```python
from evo_optimizer import EvoOptimizer

model = YourModel()
optimizer = EvoOptimizer(model.parameters(), lr=1.2e-3, total_steps=10000)

for step, (inputs, targets) in enumerate(dataloader):
    optimizer.zero_grad()
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    optimizer.step(step=step)  # Pass step for warmup/cosine scheduling
```

## What Makes It Different?

EvoOptimizer was **discovered, not designed**. Using genetic algorithm search over 50 generations with 50 individuals, evaluated across Fashion-MNIST, CIFAR-10, and MNIST, this optimizer emerged with several surprising properties:

| Property | Adam | EvoOptimizer |
|----------|------|--------------|
| Update rule | Adaptive moments only | **Sign + Adaptive hybrid** |
| β₁ (momentum) | 0.9 | **0.8553** |
| β₂ (RMS) | 0.999 | **0.9358** |
| Bias correction | ✓ | ✗ |
| Built-in scheduling | ✗ | ✓ (warmup + cosine) |

The evolved update rule:

```
Δθ = α_sign · sign(g) + α_adam · (m / √v)
```

This combines the magnitude-insensitivity of Lion-style sign updates with Adam's adaptive moment scaling.

## API Reference

### EvoOptimizer

```python
EvoOptimizer(
    params,
    lr=1.2e-3,              # Learning rate
    betas=(0.8553, 0.9358), # Momentum coefficients (lower than Adam)
    eps=5.4e-9,             # Numerical stability
    weight_decay=9.7e-4,    # Decoupled weight decay
    warmup_steps=100,       # Linear warmup steps
    total_steps=None,       # Total steps for cosine decay
    use_warmup=True,        # Enable warmup
    use_cosine_decay=True,  # Enable cosine annealing
    alpha_sign=0.7345,      # Weight for sign(gradient) term
    alpha_adam=3.6352,      # Weight for adaptive moment term
)
```

### Functional API

For custom training loops or use with `torch.compile()`:

```python
from evo_optimizer import evo_optimizer_step
from evo_optimizer.functional import compute_lr_scale

# Initialize state
exp_avgs = [torch.zeros_like(p) for p in model.parameters()]
exp_avg_sqs = [torch.zeros_like(p) for p in model.parameters()]

for step in range(num_steps):
    loss = model(x).sum()
    loss.backward()
    
    # Compute scheduled LR
    lr = 1.2e-3 * compute_lr_scale(step, warmup_steps=100, total_steps=num_steps)
    
    evo_optimizer_step(
        list(model.parameters()),
        [p.grad for p in model.parameters()],
        exp_avgs,
        exp_avg_sqs,
        lr=lr,
    )
    
    model.zero_grad()
```

## Benchmark Results

From the original paper, evaluated with 1000 training steps per task:

| Optimizer | Fashion-MNIST | CIFAR-10 | MNIST | Overall |
|-----------|---------------|----------|-------|---------|
| SGD (momentum) | 0.8250 | 0.4642 | 0.9896 | 0.7596 |
| RMSProp | 0.9304 | 0.5980 | 1.0353 | 0.8546 |
| AdamW | 0.9283 | 0.6453 | 1.0358 | 0.8698 |
| Adam | 0.9296 | 0.6669 | 1.0360 | 0.8775 |
| **EvoOptimizer** | **0.9493** | **0.7080** | **1.0387** | **0.8987** |

## When to Use EvoOptimizer

**Good fit:**
- Vision tasks (CNNs, ViTs)
- Short-to-medium training runs (evolved on 500-1000 step budgets)
- When you want built-in scheduling without external schedulers

**Consider alternatives:**
- Very long training runs (>100k steps) - may want to tune further
- Extremely large models - not yet tested at scale
- Tasks very different from vision classification

## Comparison to Similar Optimizers

- **vs Adam**: Lower momentum, no bias correction, adds sign-based term
- **vs Lion**: Hybrid approach (Lion is pure sign-based)
- **vs AdamW**: Similar decoupled weight decay, but different update rule
- **vs Sophia**: EvoOptimizer doesn't use Hessian information

## Citation

```bibtex
@article{marfinetz2025evolving,
  title={Evolving Deep Learning Optimizers},
  author={Marfinetz, Mitchell},
  journal={arXiv preprint arXiv:2512.11853},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

Areas of interest:
- Benchmarks on additional tasks (NLP, RL, audio)
- Scaling experiments on larger models
- Hyperparameter sensitivity analysis
- Integration with popular training frameworks (PyTorch Lightning, HuggingFace)
