"""Tests for the TensorFlow/Keras EvoOptimizer implementation."""

import math
import os
import pathlib
import subprocess
import sys

import pytest


def test_existing_pytorch_imports_still_work():
    from evo_optimizer import EvoOptimizer, EvoOptimizerSimplified, evo_optimizer_step

    assert EvoOptimizer is not None
    assert EvoOptimizerSimplified is not None
    assert evo_optimizer_step is not None


def test_base_package_import_does_not_require_tensorflow():
    src_path = pathlib.Path(__file__).resolve().parents[1] / "src"
    script = """
import sys

import evo_optimizer
from evo_optimizer import EvoOptimizer, EvoOptimizerSimplified, evo_optimizer_step

assert EvoOptimizer is not None
assert EvoOptimizerSimplified is not None
assert evo_optimizer_step is not None
assert "tensorflow" not in sys.modules
assert not hasattr(evo_optimizer, "KerasEvoOptimizer")
print("ok")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.fixture()
def tf_keras_optimizer():
    tf = pytest.importorskip("tensorflow")
    np = pytest.importorskip("numpy")
    from evo_optimizer.keras import KerasEvoOptimizer

    tf.keras.backend.clear_session()
    tf.random.set_seed(123)
    return tf, np, KerasEvoOptimizer


def _small_model(tf):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(3,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(2),
        ]
    )


def test_keras_optimizer_imports_when_tensorflow_is_installed(tf_keras_optimizer):
    _, _, KerasEvoOptimizer = tf_keras_optimizer

    assert KerasEvoOptimizer is not None


def test_model_training_step_changes_parameters_and_creates_state(tf_keras_optimizer):
    tf, np, KerasEvoOptimizer = tf_keras_optimizer
    model = _small_model(tf)
    optimizer = KerasEvoOptimizer(
        learning_rate=1e-2,
        use_warmup=False,
        use_cosine_decay=False,
    )
    model.compile(optimizer=optimizer, loss="mse")

    x = np.ones((4, 3), dtype=np.float32)
    y = np.zeros((4, 2), dtype=np.float32)
    before = [variable.numpy().copy() for variable in model.trainable_variables]

    loss = model.train_on_batch(x, y)

    assert math.isfinite(float(loss))
    assert any(
        not np.allclose(old, variable.numpy())
        for old, variable in zip(before, model.trainable_variables)
    )
    assert int(optimizer.iterations.numpy()) == 1
    assert len(optimizer._momentums) == len(model.trainable_variables)
    assert len(optimizer._velocities) == len(model.trainable_variables)
    assert any(np.any(moment.numpy() != 0) for moment in optimizer._momentums)
    assert any(np.any(velocity.numpy() != 0) for velocity in optimizer._velocities)


def test_optimizer_works_under_tf_function(tf_keras_optimizer):
    tf, np, KerasEvoOptimizer = tf_keras_optimizer
    variable = tf.Variable([1.0, -1.0], dtype=tf.float32)
    optimizer = KerasEvoOptimizer(
        learning_rate=0.1,
        use_warmup=False,
        use_cosine_decay=False,
    )

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(variable * variable)
        gradients = tape.gradient(loss, [variable])
        optimizer.apply_gradients(zip(gradients, [variable]))
        return loss

    loss = train_step()

    assert math.isfinite(float(loss.numpy()))
    assert not np.allclose(variable.numpy(), np.array([1.0, -1.0], dtype=np.float32))
    assert int(optimizer.iterations.numpy()) == 1


def test_one_step_update_matches_numpy_reference_without_bias_correction(
    tf_keras_optimizer,
):
    tf, np, KerasEvoOptimizer = tf_keras_optimizer
    initial = np.array([1.0, -2.0], dtype=np.float32)
    gradient = np.array([0.5, -0.25], dtype=np.float32)
    learning_rate = 0.1
    beta_1 = 0.8
    beta_2 = 0.9
    epsilon = 1e-7
    weight_decay = 0.01
    alpha_sign = 0.7
    alpha_adam = 3.0

    variable = tf.Variable(initial)
    optimizer = KerasEvoOptimizer(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        alpha_sign=alpha_sign,
        alpha_adam=alpha_adam,
        use_warmup=False,
        use_cosine_decay=False,
    )

    optimizer.apply_gradients([(tf.constant(gradient), variable)])

    momentum = (1.0 - beta_1) * gradient
    velocity = (1.0 - beta_2) * np.square(gradient)
    update = alpha_sign * np.sign(gradient) + alpha_adam * momentum / (
        np.sqrt(velocity) + epsilon
    )
    expected = initial - learning_rate * weight_decay * initial - learning_rate * update

    bias_corrected_momentum = momentum / (1.0 - beta_1)
    bias_corrected_velocity = velocity / (1.0 - beta_2)
    bias_corrected_update = (
        alpha_sign * np.sign(gradient)
        + alpha_adam
        * bias_corrected_momentum
        / (np.sqrt(bias_corrected_velocity) + epsilon)
    )
    bias_corrected_expected = (
        initial
        - learning_rate * weight_decay * initial
        - learning_rate * bias_corrected_update
    )

    np.testing.assert_allclose(variable.numpy(), expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(variable.numpy(), bias_corrected_expected, rtol=1e-5)
    np.testing.assert_allclose(
        optimizer._momentums[0].numpy(),
        momentum,
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        optimizer._velocities[0].numpy(),
        velocity,
        rtol=1e-6,
        atol=1e-8,
    )


def test_decoupled_weight_decay_changes_parameter_with_zero_gradient(
    tf_keras_optimizer,
):
    tf, np, KerasEvoOptimizer = tf_keras_optimizer
    variable = tf.Variable([2.0], dtype=tf.float32)
    optimizer = KerasEvoOptimizer(
        learning_rate=0.1,
        weight_decay=0.2,
        use_warmup=False,
        use_cosine_decay=False,
    )

    optimizer.apply_gradients([(tf.constant([0.0], dtype=tf.float32), variable)])

    np.testing.assert_allclose(variable.numpy(), np.array([1.96], dtype=np.float32))


def test_warmup_and_cosine_schedule_scales(tf_keras_optimizer):
    _, np, KerasEvoOptimizer = tf_keras_optimizer
    optimizer = KerasEvoOptimizer(
        learning_rate=1.0,
        warmup_steps=10,
        total_steps=110,
        use_warmup=True,
        use_cosine_decay=True,
    )

    assert np.isclose(float(optimizer._get_learning_rate_scale(0).numpy()), 0.1)
    assert np.isclose(float(optimizer._get_learning_rate_scale(9).numpy()), 1.0)
    assert np.isclose(float(optimizer._get_learning_rate_scale(10).numpy()), 1.0)
    assert np.isclose(float(optimizer._get_learning_rate_scale(60).numpy()), 0.5)
    assert np.isclose(float(optimizer._get_learning_rate_scale(110).numpy()), 0.0)


def test_cosine_only_schedule_starts_at_first_step(tf_keras_optimizer):
    _, np, KerasEvoOptimizer = tf_keras_optimizer
    optimizer = KerasEvoOptimizer(
        learning_rate=1.0,
        warmup_steps=0,
        total_steps=100,
        use_warmup=False,
        use_cosine_decay=True,
    )

    assert np.isclose(float(optimizer._get_learning_rate_scale(0).numpy()), 1.0)
    assert np.isclose(float(optimizer._get_learning_rate_scale(50).numpy()), 0.5)
    assert np.isclose(float(optimizer._get_learning_rate_scale(100).numpy()), 0.0)


def test_warmup_only_and_disabled_schedules(tf_keras_optimizer):
    _, np, KerasEvoOptimizer = tf_keras_optimizer
    warmup_only = KerasEvoOptimizer(
        learning_rate=1.0,
        warmup_steps=5,
        use_warmup=True,
        use_cosine_decay=False,
    )
    disabled = KerasEvoOptimizer(
        learning_rate=1.0,
        use_warmup=False,
        use_cosine_decay=False,
    )

    assert np.isclose(float(warmup_only._get_learning_rate_scale(0).numpy()), 0.2)
    assert np.isclose(float(warmup_only._get_learning_rate_scale(4).numpy()), 1.0)
    assert np.isclose(float(warmup_only._get_learning_rate_scale(100).numpy()), 1.0)
    assert np.isclose(float(disabled._get_learning_rate_scale(0).numpy()), 1.0)
    assert np.isclose(float(disabled._get_learning_rate_scale(1000).numpy()), 1.0)


def test_external_schedule_rejected_when_internal_schedule_is_active(
    tf_keras_optimizer,
):
    tf, _, KerasEvoOptimizer = tf_keras_optimizer
    schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=10,
        decay_rate=0.9,
    )

    with pytest.raises(ValueError, match="internal warmup/cosine scheduling"):
        KerasEvoOptimizer(learning_rate=schedule)

    optimizer = KerasEvoOptimizer(
        learning_rate=schedule,
        use_warmup=False,
        use_cosine_decay=False,
    )
    assert optimizer is not None


def test_get_config_and_from_config_preserve_custom_parameters(tf_keras_optimizer):
    _, np, KerasEvoOptimizer = tf_keras_optimizer
    optimizer = KerasEvoOptimizer(
        learning_rate=2e-3,
        beta_1=0.7,
        beta_2=0.8,
        epsilon=1e-6,
        weight_decay=0.02,
        warmup_steps=7,
        total_steps=31,
        use_warmup=False,
        use_cosine_decay=False,
        alpha_sign=0.2,
        alpha_adam=0.3,
        name="custom_evo",
    )

    config = optimizer.get_config()
    restored = KerasEvoOptimizer.from_config(config)

    assert np.isclose(float(config["learning_rate"]), 2e-3)
    assert config["beta_1"] == 0.7
    assert config["beta_2"] == 0.8
    assert config["epsilon"] == 1e-6
    assert config["weight_decay"] == 0.02
    assert config["warmup_steps"] == 7
    assert config["total_steps"] == 31
    assert config["use_warmup"] is False
    assert config["use_cosine_decay"] is False
    assert config["alpha_sign"] == 0.2
    assert config["alpha_adam"] == 0.3
    assert config["name"] == "custom_evo"
    assert isinstance(restored, KerasEvoOptimizer)
    assert restored.beta_1 == optimizer.beta_1
    assert restored.beta_2 == optimizer.beta_2
    assert restored.evo_weight_decay == optimizer.evo_weight_decay


def test_keras_optimizer_serialization_round_trip(tf_keras_optimizer):
    tf, _, KerasEvoOptimizer = tf_keras_optimizer
    optimizer = KerasEvoOptimizer(
        learning_rate=2e-3,
        total_steps=20,
        warmup_steps=2,
        name="serialized_evo",
    )

    serialized = tf.keras.optimizers.serialize(optimizer)
    restored = tf.keras.optimizers.deserialize(serialized)

    assert isinstance(restored, KerasEvoOptimizer)
    assert restored.name == "serialized_evo"
    assert restored.total_steps == 20
    assert restored.warmup_steps == 2


def test_compiled_model_can_save_reload_and_continue_training(
    tf_keras_optimizer,
    tmp_path,
):
    tf, np, KerasEvoOptimizer = tf_keras_optimizer
    x = np.ones((8, 3), dtype=np.float32)
    y = np.zeros((8, 2), dtype=np.float32)
    model = _small_model(tf)
    model.compile(
        optimizer=KerasEvoOptimizer(
            learning_rate=1e-2,
            warmup_steps=1,
            total_steps=8,
        ),
        loss="mse",
    )
    model.fit(x, y, epochs=1, batch_size=2, verbose=0)

    iterations_before = int(model.optimizer.iterations.numpy())
    optimizer_state_before = [
        variable.numpy().copy() for variable in model.optimizer.variables
    ]
    save_path = tmp_path / "evo_model.keras"
    model.save(save_path)

    reloaded = tf.keras.models.load_model(save_path)
    assert isinstance(reloaded.optimizer, KerasEvoOptimizer)
    assert int(reloaded.optimizer.iterations.numpy()) == iterations_before
    for expected, actual in zip(optimizer_state_before, reloaded.optimizer.variables):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)

    loss = reloaded.train_on_batch(x[:2], y[:2])
    assert math.isfinite(float(loss))
    assert int(reloaded.optimizer.iterations.numpy()) == iterations_before + 1


def test_tf_checkpoint_restores_optimizer_iterations_and_moments(
    tf_keras_optimizer,
    tmp_path,
):
    tf, np, KerasEvoOptimizer = tf_keras_optimizer
    x = np.ones((4, 3), dtype=np.float32)
    y = np.zeros((4, 2), dtype=np.float32)
    model = _small_model(tf)
    model.compile(
        optimizer=KerasEvoOptimizer(
            learning_rate=1e-2,
            use_warmup=False,
            use_cosine_decay=False,
        ),
        loss="mse",
    )
    model.train_on_batch(x, y)
    optimizer_state_before = [
        variable.numpy().copy() for variable in model.optimizer.variables
    ]

    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    checkpoint_path = checkpoint.save(str(tmp_path / "ckpt"))

    restored_model = _small_model(tf)
    restored_model.compile(
        optimizer=KerasEvoOptimizer(
            learning_rate=1e-2,
            use_warmup=False,
            use_cosine_decay=False,
        ),
        loss="mse",
    )
    restored_model.train_on_batch(x, y)
    restored_checkpoint = tf.train.Checkpoint(
        model=restored_model,
        optimizer=restored_model.optimizer,
    )
    restored_checkpoint.restore(checkpoint_path).expect_partial()

    for expected, actual in zip(
        optimizer_state_before,
        restored_model.optimizer.variables,
    ):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("beta_name", ["beta_1", "beta_2"])
@pytest.mark.parametrize("value", [-0.1, 1.0])
def test_invalid_betas_raise_errors(tf_keras_optimizer, beta_name, value):
    _, _, KerasEvoOptimizer = tf_keras_optimizer

    with pytest.raises(ValueError, match=beta_name):
        KerasEvoOptimizer(**{beta_name: value})


def test_invalid_constructor_arguments_raise_clear_errors(tf_keras_optimizer):
    _, _, KerasEvoOptimizer = tf_keras_optimizer

    with pytest.raises(ValueError, match="learning_rate"):
        KerasEvoOptimizer(learning_rate=-1e-3)
    with pytest.raises(ValueError, match="epsilon"):
        KerasEvoOptimizer(epsilon=-1e-8)
    with pytest.raises(ValueError, match="weight_decay"):
        KerasEvoOptimizer(weight_decay=-1e-4)
    with pytest.raises(ValueError, match="warmup_steps"):
        KerasEvoOptimizer(warmup_steps=-1)
    with pytest.raises(ValueError, match="total_steps"):
        KerasEvoOptimizer(total_steps=0)
    with pytest.raises(ValueError, match="alpha_sign"):
        KerasEvoOptimizer(alpha_sign=float("nan"))


def test_sparse_gradients_raise_clear_error(tf_keras_optimizer):
    tf, _, KerasEvoOptimizer = tf_keras_optimizer
    embedding = tf.keras.layers.Embedding(16, 4)
    optimizer = KerasEvoOptimizer(
        learning_rate=1e-2,
        use_warmup=False,
        use_cosine_decay=False,
    )

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(embedding(tf.constant([1, 2, 3])))
    gradients = tape.gradient(loss, embedding.trainable_variables)

    assert isinstance(gradients[0], tf.IndexedSlices)
    with pytest.raises(NotImplementedError, match="sparse gradients"):
        optimizer.apply_gradients(zip(gradients, embedding.trainable_variables))
