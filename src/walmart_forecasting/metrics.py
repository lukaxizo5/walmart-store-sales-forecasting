from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def weighted_mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    is_holiday: ArrayLike,
    holiday_weight: float = 5.0,
) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    holiday_flags = np.asarray(is_holiday, dtype=bool)

    if actual.ndim != 1:
        actual = actual.reshape(-1)

    if predicted.ndim != 1:
        predicted = predicted.reshape(-1)

    if holiday_flags.ndim != 1:
        holiday_flags = holiday_flags.reshape(-1)

    if actual.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape. "
            f"Received {actual.shape} and {predicted.shape}."
        )

    if actual.shape != holiday_flags.shape:
        raise ValueError(
            "is_holiday must have the same shape as y_true. "
            f"Received {holiday_flags.shape} and {actual.shape}."
        )

    if actual.size == 0:
        raise ValueError("Metric inputs cannot be empty.")

    if holiday_weight <= 0:
        raise ValueError("holiday_weight must be positive.")

    if not np.isfinite(actual).all():
        raise ValueError("y_true contains non-finite values.")

    if not np.isfinite(predicted).all():
        raise ValueError("y_pred contains non-finite values.")

    weights = np.where(
        holiday_flags,
        holiday_weight,
        1.0,
    )

    absolute_errors = np.abs(actual - predicted)

    return float(
        np.sum(weights * absolute_errors) /
        np.sum(weights)
    )


def mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)

    if actual.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)

    if actual.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    return float(
        np.sqrt(np.mean((actual - predicted) ** 2))
    )