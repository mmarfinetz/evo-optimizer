"""Tests for EvoOptimizer."""

import pytest
import torch
import torch.nn as nn


class TestEvoOptimizer:
    """Test suite for EvoOptimizer."""

    def test_import(self):
        """Test that the package imports correctly."""
        from evo_optimizer import EvoOptimizer, EvoOptimizerSimplified, evo_optimizer_step
        assert EvoOptimizer is not None
        assert EvoOptimizerSimplified is not None
        assert evo_optimizer_step is not None

    def test_basic_step(self):
        """Test basic optimization step."""
        from evo_optimizer import EvoOptimizer

        model = nn.Linear(10, 2)
        optimizer = EvoOptimizer(model.parameters())

        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))

        # Initial forward/backward
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()

        # Step should not raise
        optimizer.step(step=0)
        optimizer.zero_grad()

    def test_warmup_schedule(self):
        """Test that warmup schedule affects learning rate."""
        from evo_optimizer import EvoOptimizer

        model = nn.Linear(10, 2)
        optimizer = EvoOptimizer(
            model.parameters(),
            lr=1.0,
            warmup_steps=10,
            use_warmup=True,
            use_cosine_decay=False,
        )

        # At step 0, LR should be 0.1 (1/10)
        scale_0 = optimizer._get_lr_scale(0, optimizer.param_groups[0])
        assert abs(scale_0 - 0.1) < 1e-6

        # At step 4, LR should be 0.5 (5/10)
        scale_4 = optimizer._get_lr_scale(4, optimizer.param_groups[0])
        assert abs(scale_4 - 0.5) < 1e-6

        # At step 10, LR should be 1.0
        scale_10 = optimizer._get_lr_scale(10, optimizer.param_groups[0])
        assert abs(scale_10 - 1.0) < 1e-6

    def test_cosine_decay(self):
        """Test cosine decay schedule."""
        from evo_optimizer import EvoOptimizer

        model = nn.Linear(10, 2)
        optimizer = EvoOptimizer(
            model.parameters(),
            lr=1.0,
            warmup_steps=0,
            total_steps=100,
            use_warmup=False,
            use_cosine_decay=True,
        )

        # At step 0, LR should be ~1.0
        scale_0 = optimizer._get_lr_scale(0, optimizer.param_groups[0])
        assert abs(scale_0 - 1.0) < 1e-6

        # At step 50, LR should be ~0.5 (cosine midpoint)
        scale_50 = optimizer._get_lr_scale(50, optimizer.param_groups[0])
        assert abs(scale_50 - 0.5) < 0.01

        # At step 100, LR should be ~0.0
        scale_100 = optimizer._get_lr_scale(100, optimizer.param_groups[0])
        assert scale_100 < 0.01

    def test_weight_decay(self):
        """Test that weight decay is applied."""
        from evo_optimizer import EvoOptimizer

        model = nn.Linear(10, 2, bias=False)
        initial_weight = model.weight.clone()

        optimizer = EvoOptimizer(
            model.parameters(),
            lr=0.1,
            weight_decay=0.1,
            use_warmup=False,
            use_cosine_decay=False,
        )

        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))

        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step(step=0)

        # Weight should have changed
        assert not torch.allclose(model.weight, initial_weight)

    def test_multiple_param_groups(self):
        """Test optimizer with multiple parameter groups."""
        from evo_optimizer import EvoOptimizer

        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))

        optimizer = EvoOptimizer(
            [
                {"params": model[0].parameters(), "lr": 0.01},
                {"params": model[1].parameters(), "lr": 0.001},
            ]
        )

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[1]["lr"] == 0.001

    def test_state_initialization(self):
        """Test that optimizer state is initialized correctly."""
        from evo_optimizer import EvoOptimizer

        model = nn.Linear(10, 2)
        optimizer = EvoOptimizer(model.parameters())

        # State should be empty initially
        assert len(optimizer.state) == 0

        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))

        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step(step=0)

        # State should now have entries
        for p in model.parameters():
            if p.grad is not None:
                assert p in optimizer.state
                assert "exp_avg" in optimizer.state[p]
                assert "exp_avg_sq" in optimizer.state[p]

    def test_closure(self):
        """Test optimizer with closure."""
        from evo_optimizer import EvoOptimizer

        model = nn.Linear(10, 2)
        optimizer = EvoOptimizer(model.parameters())

        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))

        def closure():
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            return loss

        loss = optimizer.step(closure=closure, step=0)
        assert loss is not None
        assert isinstance(loss.item(), float)

    def test_sparse_grad_raises(self):
        """Test that sparse gradients raise an error."""
        from evo_optimizer import EvoOptimizer

        # Create embedding which produces sparse gradients
        embedding = nn.Embedding(100, 10, sparse=True)
        optimizer = EvoOptimizer(embedding.parameters())

        indices = torch.tensor([1, 2, 3])
        output = embedding(indices).sum()
        output.backward()

        with pytest.raises(RuntimeError, match="sparse"):
            optimizer.step(step=0)


class TestFunctionalAPI:
    """Test suite for functional API."""

    def test_evo_optimizer_step(self):
        """Test functional step."""
        from evo_optimizer import evo_optimizer_step

        params = [torch.randn(10, 10, requires_grad=True)]
        grads = [torch.randn(10, 10)]
        exp_avgs = [torch.zeros(10, 10)]
        exp_avg_sqs = [torch.zeros(10, 10)]

        initial_param = params[0].clone()

        evo_optimizer_step(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            lr=0.01,
        )

        # Parameter should have changed
        assert not torch.allclose(params[0], initial_param)

        # Moments should be non-zero
        assert exp_avgs[0].abs().sum() > 0
        assert exp_avg_sqs[0].abs().sum() > 0

    def test_compute_lr_scale(self):
        """Test LR scale computation."""
        from evo_optimizer.functional import compute_lr_scale

        # Test warmup
        assert abs(compute_lr_scale(0, warmup_steps=10, use_cosine_decay=False) - 0.1) < 1e-6
        assert abs(compute_lr_scale(9, warmup_steps=10, use_cosine_decay=False) - 1.0) < 1e-6

        # Test cosine decay
        scale = compute_lr_scale(50, warmup_steps=0, total_steps=100, use_warmup=False)
        assert abs(scale - 0.5) < 0.01


class TestConvergence:
    """Test convergence on simple problems."""

    def test_linear_regression(self):
        """Test convergence on linear regression."""
        from evo_optimizer import EvoOptimizer

        torch.manual_seed(42)

        # Generate data
        X = torch.randn(100, 5)
        true_w = torch.randn(5, 1)
        y = X @ true_w + 0.1 * torch.randn(100, 1)

        # Model
        model = nn.Linear(5, 1, bias=False)

        optimizer = EvoOptimizer(
            model.parameters(),
            lr=0.1,
            total_steps=200,
            warmup_steps=20,
        )

        initial_loss = None
        final_loss = None

        for step in range(200):
            optimizer.zero_grad()
            pred = model(X)
            loss = nn.functional.mse_loss(pred, y)

            if step == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step(step=step)
            final_loss = loss.item()

        # Should converge
        assert final_loss < initial_loss * 0.1

    def test_classification(self):
        """Test convergence on simple classification."""
        from evo_optimizer import EvoOptimizer

        torch.manual_seed(42)

        # Generate linearly separable data
        X_pos = torch.randn(50, 10) + 2
        X_neg = torch.randn(50, 10) - 2
        X = torch.cat([X_pos, X_neg], dim=0)
        y = torch.cat([torch.ones(50), torch.zeros(50)]).long()

        model = nn.Sequential(nn.Linear(10, 2))

        optimizer = EvoOptimizer(
            model.parameters(),
            lr=0.05,
            total_steps=100,
            warmup_steps=10,
        )

        for step in range(100):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(X), y)
            loss.backward()
            optimizer.step(step=step)

        # Check accuracy
        with torch.no_grad():
            preds = model(X).argmax(dim=1)
            accuracy = (preds == y).float().mean().item()

        assert accuracy > 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
