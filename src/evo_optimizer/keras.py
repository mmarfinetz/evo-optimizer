"""TensorFlow/Keras implementation of EvoOptimizer."""

import math
from typing import Any, Optional

import tensorflow as tf
from tensorflow import keras

_SCHEDULE_ERROR = (
    "KerasEvoOptimizer internal warmup/cosine scheduling cannot be combined "
    "with an external Keras learning-rate schedule. Disable the Evo schedule "
    "or use a scalar base learning rate."
)


@keras.utils.register_keras_serializable(
    package="evo_optimizer",
    name="KerasEvoOptimizer",
)
class KerasEvoOptimizer(keras.optimizers.Optimizer):
    r"""Keras optimizer implementing the EvoOptimizer update rule.

    The optimizer mirrors the PyTorch EvoOptimizer algorithm: first and second
    moments are updated without bias correction, the parameter update combines a
    sign term and an Adam-style adaptive term, and weight decay is decoupled
    from the gradient update.

    If ``total_steps`` is ``None``, cosine decay is disabled even when
    ``use_cosine_decay`` is true.

    TensorFlow is an optional dependency for the package. Import this class from
    ``evo_optimizer.keras`` only when TensorFlow/Keras support is installed.
    """

    def __init__(
        self,
        learning_rate: Any = 1.2e-3,
        beta_1: float = 0.8553,
        beta_2: float = 0.9358,
        epsilon: float = 5.4e-9,
        weight_decay: float = 9.7e-4,
        warmup_steps: int = 100,
        total_steps: Optional[int] = None,
        use_warmup: bool = True,
        use_cosine_decay: bool = True,
        alpha_sign: float = 0.7345,
        alpha_adam: float = 3.6352,
        name: str = "KerasEvoOptimizer",
        **kwargs: Any,
    ):
        self._validate_learning_rate(
            learning_rate,
            use_warmup=use_warmup,
            use_cosine_decay=use_cosine_decay,
        )
        self._validate_hyperparameters(
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            alpha_sign=alpha_sign,
            alpha_adam=alpha_adam,
        )

        # Keras' base optimizer applies its own weight decay before
        # update_step() using self.learning_rate. EvoOptimizer needs decoupled
        # decay with the internally scheduled effective learning rate, so keep
        # the Evo coefficient separate and leave the base coefficient disabled.
        super().__init__(name=name, weight_decay=None, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.evo_weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.use_warmup = use_warmup
        self.use_cosine_decay = use_cosine_decay
        self.alpha_sign = alpha_sign
        self.alpha_adam = alpha_adam
        self._momentums = []
        self._velocities = []

    def build(self, var_list):
        """Create first- and second-moment state for each trainable variable."""
        if getattr(self, "_built", False):
            return
        super().build(var_list)

        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(self.add_variable_from_reference(var, "m"))
            self._velocities.append(self.add_variable_from_reference(var, "v"))
        self._built = True

    def _get_learning_rate_scale(self, step):
        """Return the internal warmup/cosine learning-rate multiplier.

        Args:
            step: Zero-based optimizer step, typically ``self.iterations``.

        Returns:
            A scalar Tensor in ``[0, 1]``.
        """
        step = tf.cast(step, tf.float32)
        scale = tf.ones((), dtype=tf.float32)

        if self.use_warmup and self.warmup_steps > 0:
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)
            warmup_scale = (step + 1.0) / warmup_steps
            scale = tf.where(step < warmup_steps, warmup_scale, scale)

        if self.use_cosine_decay and self.total_steps is not None:
            warmup = self.warmup_steps if self.use_warmup else 0
            decay_steps = max(1, self.total_steps - warmup)
            progress = (step - float(warmup)) / float(decay_steps)
            progress = tf.clip_by_value(progress, 0.0, 1.0)
            cosine_scale = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * progress))
            scale = tf.where(step >= float(warmup), cosine_scale, scale)

        return tf.maximum(scale, 0.0)

    def update_step(self, gradient, variable, learning_rate=None):
        """Apply one EvoOptimizer update to ``variable``."""
        if isinstance(gradient, tf.IndexedSlices):
            raise NotImplementedError(
                "KerasEvoOptimizer does not currently support sparse gradients "
                "(tf.IndexedSlices). Use dense gradients or a different "
                "optimizer for sparse embedding updates."
            )

        lr = self._effective_learning_rate(variable, learning_rate)
        gradient = tf.cast(gradient, variable.dtype)
        beta_1 = tf.cast(self.beta_1, variable.dtype)
        beta_2 = tf.cast(self.beta_2, variable.dtype)
        epsilon = tf.cast(self.epsilon, variable.dtype)
        alpha_sign = tf.cast(self.alpha_sign, variable.dtype)
        alpha_adam = tf.cast(self.alpha_adam, variable.dtype)

        var_index = self._variable_index(variable)
        momentum = self._momentums[var_index]
        velocity = self._velocities[var_index]

        momentum.assign_add((gradient - momentum) * (1.0 - beta_1))
        velocity.assign_add((tf.square(gradient) - velocity) * (1.0 - beta_2))

        update = (
            alpha_sign * tf.sign(gradient)
            + alpha_adam * momentum / (tf.sqrt(velocity) + epsilon)
        )

        if self.evo_weight_decay != 0:
            weight_decay = tf.cast(self.evo_weight_decay, variable.dtype)
            variable.assign_sub(lr * weight_decay * variable)
        variable.assign_sub(lr * update)

    def get_config(self):
        """Return a serializable optimizer configuration."""
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "weight_decay": self.evo_weight_decay,
                "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps,
                "use_warmup": self.use_warmup,
                "use_cosine_decay": self.use_cosine_decay,
                "alpha_sign": self.alpha_sign,
                "alpha_adam": self.alpha_adam,
            }
        )
        return config

    def _effective_learning_rate(self, variable, learning_rate=None):
        base_learning_rate = self.learning_rate
        if learning_rate is not None:
            base_learning_rate = learning_rate
        base_learning_rate = tf.cast(base_learning_rate, variable.dtype)
        scale = tf.cast(self._get_learning_rate_scale(self.iterations), variable.dtype)
        return base_learning_rate * scale

    def _variable_index(self, variable) -> int:
        if hasattr(self, "_index_dict"):
            return self._index_dict[self._var_key(variable)]
        get_variable_index = getattr(super(), "_get_variable_index", None)
        if get_variable_index is None:
            raise KeyError(f"Cannot find optimizer state for variable {variable.name}.")
        return get_variable_index(variable)

    @staticmethod
    def _validate_learning_rate(
        learning_rate: Any,
        *,
        use_warmup: bool,
        use_cosine_decay: bool,
    ) -> None:
        schedule_type = keras.optimizers.schedules.LearningRateSchedule
        if (use_warmup or use_cosine_decay) and (
            isinstance(learning_rate, schedule_type) or callable(learning_rate)
        ):
            raise ValueError(_SCHEDULE_ERROR)

        if isinstance(learning_rate, schedule_type) or callable(learning_rate):
            return

        try:
            lr_tensor = tf.convert_to_tensor(learning_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "learning_rate must be a non-negative scalar, scalar variable, "
                "or Keras learning-rate schedule."
            ) from exc

        if lr_tensor.shape.rank != 0:
            raise ValueError("learning_rate must be a scalar value.")

        lr_value = tf.get_static_value(lr_tensor)
        if lr_value is not None:
            try:
                lr_value = float(lr_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("learning_rate must be numeric.") from exc
            if not math.isfinite(lr_value):
                raise ValueError("learning_rate must be finite.")
            if lr_value < 0:
                raise ValueError("learning_rate must be non-negative.")

    @staticmethod
    def _validate_hyperparameters(
        *,
        beta_1: float,
        beta_2: float,
        epsilon: float,
        weight_decay: float,
        warmup_steps: int,
        total_steps: Optional[int],
        alpha_sign: float,
        alpha_adam: float,
    ) -> None:
        for name, value in (
            ("beta_1", beta_1),
            ("beta_2", beta_2),
            ("epsilon", epsilon),
            ("weight_decay", weight_decay),
            ("alpha_sign", alpha_sign),
            ("alpha_adam", alpha_adam),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite numeric value.")

        if not 0.0 <= beta_1 < 1.0:
            raise ValueError("beta_1 must be in the range [0, 1).")
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError("beta_2 must be in the range [0, 1).")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer.")
        if total_steps is not None and (
            not isinstance(total_steps, int) or total_steps <= 0
        ):
            raise ValueError("total_steps must be a positive integer or None.")
