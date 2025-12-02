# Evolutionary Optimizer Discovery

This repository implements an **evolutionary search over optimizer algorithms** in PyTorch.  
Instead of tuning Adam/SGD hyperparameters, we evolve entire update rules using a genetic algorithm and evaluate them on real vision benchmarks.

The best evolved optimizer discovered in this run **outperforms SGD (momentum), RMSProp, Adam, and AdamW** on a 1000-step multi-task benchmark covering Fashion-MNIST, CIFAR-10, and MNIST.

---

## 1. Overview

Modern optimizers such as SGD, Adam, and AdamW are hand-designed.  
This project explores an alternative: **searching directly in the space of optimizer update rules**.

Each candidate optimizer is encoded as a small "genome" describing:

- A sequence of primitive update terms (e.g. gradient, momentum, Adam-style term, sign of gradient).
- Scalar weights for each term.
- Learning-rate and regularization hyperparameters (log-scaled).
- Flags controlling momentum, second moment, bias correction, gradient clipping, warmup, and cosine decay.

A genetic algorithm (GA) evolves these genomes based on how well they train neural networks on real datasets.

---

## 2. Method

### 2.1 Genome and Update Rule

Let `g_t` be the gradient, `m_t` the first-moment EMA, and `v_t` the second-moment EMA at step `t`.

We define a catalog of primitive terms:

| Term | Definition |
|------|------------|
| `GRAD` | `g_t` |
| `MOMENTUM` | `m_t` |
| `RMS_NORM` | `g_t / (sqrt(v_t) + ε)` |
| `ADAM_TERM` | `m_t / (sqrt(v_t) + ε)` |
| `SIGN_GRAD` | `sign(g_t)` |
| `SQRT_GRAD` | `g_t / (\|g_t\| + ε)` |
| `NESTEROV` | Nesterov-style momentum term |

An optimizer genome specifies:

- `term_types`: list of primitive types
- `alphas`: per-term scalar weights
- `lr_log10`: learning rate in log10 space
- `beta1`, `beta2`: EMA coefficients
- `eps_log10`: numerical stability epsilon (log10)
- `wd_log10`: decoupled weight decay (log10)
- Flags: `use_m`, `use_v`, `use_bias_correction`, `use_gradient_clipping`, `clip_value`, `use_warmup`, `warmup_steps`, `use_cosine_decay`

The resulting parameter update is:

```
Δw_t = η * Σ(α_k * T_k(g_t, m_t, v_t))
```

where `T_k` is one of the primitive terms and `η` is the scheduled learning rate (with optional warmup and cosine decay).

### 2.2 Evolutionary Search

The genetic algorithm operates on populations of genomes:

- **Selection:** tournament selection and elite carryover.
- **Crossover:** structural crossover on the `term_types` and `alphas` sequences, plus mixing of scalar hyperparameters and flags.
- **Mutation:** numeric perturbations of log-scaled parameters and alphas; structural mutations (add/remove/change term); and flipping of boolean flags. Mutation rates anneal over generations.
- **Checkpointing:** evolution state and partial generation progress are saved to disk so long runs can be resumed safely (e.g. after Colab disconnects).

---

## 3. Experimental Setup

All experiments in this run were conducted in PyTorch (2.9.0 + CUDA) on a single NVIDIA Tesla T4 (15.8 GB VRAM).

### 3.1 Tasks and Models

Each optimizer is evaluated on three supervised classification tasks:

| Dataset | Description | Model |
|---------|-------------|-------|
| **MNIST** | Grayscale digits (28×28) | SmallCNN |
| **Fashion-MNIST** | Grayscale clothing images (28×28) | SmallCNN |
| **CIFAR-10** | Color images (32×32) | Custom CNN |

For each dataset:
- Training split is subsampled to **15,000** examples for faster evaluation.
- Standard normalization and light augmentation are applied (including random crop and horizontal flip for CIFAR-10).

### 3.2 Evolution Configuration

The main evolution run uses:

- Population size: **50**
- Generations: **50**
- Tasks: `fashion_mnist`, `cifar10`, `mnist`
- Training budget per evaluation:
  - `max_steps_per_task` = **500**
  - `num_seeds` = **2**

Each genome is evaluated with up to ~3000 training steps (500 steps/task × 3 tasks × 2 seeds). Runs that diverge earlier are terminated and penalized.

### 3.3 Fitness Function

For each task and seed:

1. Train the model for up to `max_steps` mini-batches.
2. Compute validation accuracy.
3. Track recent training loss (last 50 steps).

Task-level fitness:
```
fitness_task ≈ val_accuracy + 0.05 * max(0, 1 - avg_final_loss)
```

Overall fitness for a genome is the mean task fitness across the three datasets and seeds.

---

## 4. Evolved Optimizer

The best optimizer discovered after 50 generations:

```python
genome = Genome(
    term_types=[4, 4, 3, 3],  # [SIGN_GRAD, SIGN_GRAD, ADAM_TERM, ADAM_TERM]
    alphas=[0.4121, 0.3224, 2.0030, 1.6322],
    lr_log10=-2.9199,         # lr ≈ 1.20e-3
    beta1=0.8553,
    beta2=0.9358,
    eps_log10=-8.2642,        # eps ≈ 5.4e-9
    wd_log10=-3.0112,         # weight decay ≈ 9.7e-4
    use_m=True,
    use_v=True,
    use_bias_correction=False,
    use_gradient_clipping=False,
    use_warmup=True,
    warmup_steps=100,
    use_cosine_decay=True,
)
```

This optimizer combines:

- Two **SIGN_GRAD** terms (magnitude-insensitive components)
- Two **ADAM_TERM** terms (Adam-style `m / sqrt(v)` components)

with relatively large positive weights on the Adam-like terms, warmup + cosine decay, and no bias correction. It is structurally different from AdamW and was discovered automatically by the evolutionary process.

---

## 5. Results

We compare the best evolved optimizer to standard baselines using a larger evaluation budget:

- **1000 training steps per task** (`max_steps = 1000`)
- **3 tasks** (Fashion-MNIST, CIFAR-10, MNIST)
- **3 seeds** per optimizer

Total budget: ~9000 training steps per optimizer.

### 5.1 Overall Fitness

| Optimizer | Overall Fitness |
|-----------|----------------:|
| SGD (momentum) | 0.7596 |
| RMSProp | 0.8546 |
| AdamW | 0.8698 |
| Adam | 0.8775 |
| **Evolved** | **0.8987** |

Relative to the strongest baseline (Adam):
- **+0.0211** absolute improvement over Adam
- **+0.0289** absolute improvement over AdamW

### 5.2 Per-Task Fitness

| Optimizer | Fashion-MNIST | CIFAR-10 | MNIST |
|-----------|-------------:|---------:|------:|
| SGD (momentum) | 0.8250 | 0.4642 | 0.9896 |
| Adam | 0.9296 | 0.6669 | 1.0360 |
| AdamW | 0.9283 | 0.6453 | 1.0358 |
| RMSProp | 0.9304 | 0.5980 | 1.0353 |
| **Evolved** | **0.9493** | **0.7080** | **1.0387** |

Key observations:

- **Fashion-MNIST:** evolved optimizer reaches 0.9493 vs Adam 0.9296 → ~+0.02 absolute improvement
- **CIFAR-10:** evolved optimizer reaches 0.7080 vs Adam 0.6669 → **+0.0411 vs Adam**, **+0.0627 vs AdamW** (largest gains)
- **MNIST:** evolved optimizer slightly but consistently outperforms all baselines

Within this 1000-step short-horizon training regime, the evolved optimizer is the top performer on **every dataset** tested.

> These are preliminary results from a single full evolution run. For a publication-grade study, additional seeds, held-out tasks, and multiple independent runs would be appropriate.

---

## 6. Repository Structure

```
evo-optimizer/
├── evolutionary_optimizer_colab.ipynb   # Main notebook with full implementation
├── README.md
└── checkpoints/                          # Evolution state and results (gitignored)
```

The notebook contains:
- Genome definition and primitive terms
- Optimizer implementation (`init_opt_state`, `optimizer_step`)
- Task models and data loading
- Fitness evaluation functions
- Genetic algorithm (selection, crossover, mutation)
- Evolution loop with checkpointing
- Visualization and baseline comparison
- Export of the best genome and results

---

## 7. Running the Code

### 7.1 In Google Colab

1. Open `evolutionary_optimizer_colab.ipynb` in Google Colab.
2. Enable GPU acceleration: *Runtime → Change runtime type → Hardware accelerator → GPU*.
3. Set `CHECKPOINT_DIR` to a directory on your Google Drive.
4. Run all cells.

The evolution loop will resume from an existing checkpoint if present. Checkpoints are saved periodically and at the end of each generation.

### 7.2 Locally

```bash
pip install torch torchvision matplotlib pandas
```

Then open the notebook in Jupyter or VS Code and configure `CHECKPOINT_DIR` to a local path.

A CUDA-enabled GPU is strongly recommended; CPU-only runs will be very slow.

---

## 8. Using the Evolved Optimizer

Once the `Genome` class and helper functions are defined (see notebook), use the evolved optimizer with any PyTorch model:

```python
model = YourModel().to(device)

genome = Genome(
    term_types=[4, 4, 3, 3],
    alphas=[0.41205802174892475, 0.32238621456272765,
            2.0030004279455267, 1.6322011630060542],
    lr_log10=-2.9198932483267144,
    beta1=0.8553020861538518,
    beta2=0.9357561078635277,
    eps_log10=-8.26423960220415,
    wd_log10=-3.0112238437153316,
    use_m=True,
    use_v=True,
    use_bias_correction=False,
    use_gradient_clipping=False,
    use_warmup=True,
    warmup_steps=100,
    use_cosine_decay=True,
)

opt_state = init_opt_state(model, genome)
criterion = torch.nn.CrossEntropyLoss()
total_steps = 1000

for step, (x, y) in enumerate(train_loader):
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer_step(model, opt_state, genome,
                   global_step=step, total_steps=total_steps)
```

---

## 9. Limitations and Future Work

**Current limitations:**
- Results are based on a single full evolution run and a limited task set
- The evaluation regime focuses on **short-horizon** training (up to 1000 steps), not full convergence
- Larger and more diverse task suites would provide a stronger test of generalization

**Planned extensions:**
- Additional tasks and longer training horizons
- Comparison with more recent optimizers (Lion, Sophia, Adafactor)
- Packaging evolved optimizers as standalone PyTorch optimizer classes
- Parallelizing evaluation across multiple GPUs

---

## 10. License

MIT License

---

## 11. Citation

If you use this code or results in academic work, please consider citing:

```bibtex
@misc{marfinetz2025evooptimizer,
  author = {Marfinetz, Mitchell},
  title = {Evolutionary Optimizer Discovery},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/mmarfinetz/evo-optimizer}
}
```
