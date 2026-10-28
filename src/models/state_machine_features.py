"""
Binary feature extractor for the state-machine idle detector.
Uses only raw 9-channel inertial signals (no feature engineering).
All features are lightweight O(n) statistics — suitable for
continuous real-time monitoring in the IDLE state.
"""

import numpy as np
from numpy.typing import NDArray


def extract_features(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute lightweight statistical features from a single window
    of raw inertial signals.

    Args:
        x: shape (9, 128) — one UCI HAR window (9 channels, 128 time steps)
           Channel order matches dataset.py:
             0-2: body_acc  (x, y, z)
             3-5: body_gyro (x, y, z)
             6-8: total_acc (x, y, z)

    Returns:
        feature vector of shape (n_features,)
    """
    body_acc  = x[0:3]   # (3, 128)
    body_gyro = x[3:6]
    total_acc = x[6:9]

    features = []

    # ── body_acc statistics ──────────────────────────────────────
    for axis in body_acc:
        features.append(float(np.std(axis)))          # std
        features.append(float(np.mean(np.abs(axis))))  # mean absolute deviation
        features.append(float(np.max(np.abs(axis))))  # peak amplitude

    # ── body_gyro statistics ───────────────────────────────────
    for axis in body_gyro:
        features.append(float(np.std(axis)))

    # gyro magnitude (rotation intensity)
    gyro_mag = np.sqrt(body_gyro[0]**2 + body_gyro[1]**2 + body_gyro[2]**2)
    features.append(float(np.mean(gyro_mag)))          # mean gyro magnitude
    features.append(float(np.std(gyro_mag)))            # std  gyro magnitude

    # ── total_acc statistics ───────────────────────────────────
    # Useful for detecting orientation changes (sitting → standing)
    for axis in total_acc:
        features.append(float(np.mean(axis)))          # mean (related to gravity orientation)
        features.append(float(np.std(axis)))           # std  (dynamic component)

    # ── cross-channel features ──────────────────────────────────
    # Total acceleration magnitude (should be ≈ 1g at rest)
    acc_mag = np.sqrt(total_acc[0]**2 + total_acc[1]**2 + total_acc[2]**2)
    features.append(float(np.mean(acc_mag)))           # mean acc magnitude
    features.append(float(np.std(acc_mag)))             # std  acc magnitude

    # SMA: Signal Magnitude Area
    sma = np.mean(np.abs(total_acc[0]) + np.abs(total_acc[1]) + np.abs(total_acc[2]))
    features.append(float(sma))

    # ── zero-crossing rate on body_acc_x (movement frequency proxy) ──
    zcr = np.sum(np.diff(np.sign(body_acc[0])) != 0) / 128.0
    features.append(float(zcr))

    # ── peak-to-mean ratio on body_acc magnitude ───────────────
    acc_mag_ba = np.sqrt(body_acc[0]**2 + body_acc[1]**2 + body_acc[2]**2)
    features.append(float(np.max(acc_mag_ba) / (np.mean(acc_mag_ba) + 1e-9)))

    return np.array(features, dtype=np.float64)


def extract_features_batch(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorised batch version of extract_features."""
    return np.stack([extract_features(x) for x in X], axis=0)
