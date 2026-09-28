from __future__ import annotations

import numpy as np


def f05(truth, prediction):
    truth, prediction = set(truth), set(prediction)
    if not truth:
        return float(not prediction)
    tp = len(truth & prediction)
    return 1.25 * tp / (1.25 * tp + .25 * len(truth - prediction) + len(prediction - truth))


def score(truth, prediction):
    values = [f05(t, prediction.get(q, [])) for q, t in truth.items()]
    return float(np.mean(values)) if values else 0.0


def bootstrap_delta(a, b, seed=2026, repeats=1000):
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if not len(delta):
        return {"mean": 0., "low": 0., "high": 0.}
    rng = np.random.default_rng(seed)
    # Multiplier bootstrap avoids allocating repeats x entity_count arrays.
    samples = []
    for _ in range(repeats):
        weights = rng.poisson(1, len(delta))
        samples.append(float(np.dot(delta, weights) / max(1, weights.sum())))
    low, high = np.quantile(samples, [.025, .975])
    return dict(mean=float(delta.mean()), low=float(low), high=float(high))
