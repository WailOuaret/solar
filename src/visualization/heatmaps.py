from __future__ import annotations

import numpy as np


def normalize_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    min_value = values.min()
    max_value = values.max()
    if max_value <= min_value:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)

