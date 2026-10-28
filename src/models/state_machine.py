"""
HAR State Machine — three states + lightweight binary disturbance filter.

States
──────
  IDLE   — low-motion: SITTING / STANDING / LAYING
           No heavy model is running; only variance is monitored.
  CHECK  — transient: a variance spike was detected.
           The lightweight binary classifier decides whether it is
           a real activity transition or just a short disturbance.
  ACTIVE — high-motion: WALKING / WALKING_UPSTAIRS / WALKING_DOWNSTAIRS
           Full PatchTST + GAF model runs to classify the exact activity.

State transitions
─────────────────
  IDLE ──(acc_std > upper_th)──► CHECK
  CHECK ──(binary_model says ACTIVE)──► ACTIVE
  CHECK ──(binary_model says disturbance)──► IDLE
  ACTIVE ──(acc_std < lower_th)──► IDLE

The hysteresis gap (upper_th > lower_th) prevents oscillation at the
decision boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray


# ──────────────────────────────────────────────────────────────────────────────
#  State enum
# ──────────────────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE   = auto()
    CHECK  = auto()
    ACTIVE = auto()


LABEL_NAMES = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}

# IDLE labels in the UCI HAR 0-5 encoding
IDLE_LABELS = {3, 4, 5}          # SITTING, STANDING, LAYING
ACTIVE_LABELS = {0, 1, 2}        # WALKING, WALKING_UP, WALKING_DOWN


# ──────────────────────────────────────────────────────────────────────────────
#  Dataclass holding tuned thresholds (set after calibration on val data)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Thresholds:
    upper: float   # IDLE → CHECK
    lower: float   # ACTIVE → IDLE


# ──────────────────────────────────────────────────────────────────────────────
#  Main State Machine class
# ──────────────────────────────────────────────────────────────────────────────

class HARStateMachine:
    """
    Parameters
    ----------
    binary_model : sklearn RandomForestClassifier (already trained)
        Lightweight binary classifier: disturbance vs real activity.
        Input = feature vector from state_machine_features.py
        Output = 0 (disturbance / idle) or 1 (real activity)
    full_model : torch.nn.Module (already trained, on GPU/CPU)
        PatchTST + GAF dual-stream model returning logits (6 classes).
    thresholds : Thresholds
        Two thresholds that define the variance-based trigger.
    device : str
        'cuda' or 'cpu' for the full model.
    history_len : int
        Number of past windows to accumulate when entering ACTIVE state
        before running the full model (default 1 = instant).
    """

    def __init__(
        self,
        binary_model,
        full_model: torch.nn.Module,
        thresholds: Thresholds,
        device: str = "cuda",
        history_len: int = 1,
    ):
        self.binary_model = binary_model
        self.full_model   = full_model
        self.full_model.eval()
        self.device      = torch.device(device)
        self.full_model.to(self.device)

        self.upper_th = thresholds.upper
        self.lower_th = thresholds.lower
        self.history_len = history_len

        self._state       = State.IDLE
        self._acc_buffer  = []       # stores recent acc windows while in CHECK
        self._full_buffer = []       # stores recent windows while in ACTIVE

    @property
    def state(self) -> State:
        return self._state

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def acc_variance(window: NDArray[np.float64]) -> float:
        """Root-mean-square of body_acc std across 3 axes."""
        body_acc = window[0:3]                     # (3, 128)
        rms = np.sqrt(np.mean([np.std(axis)**2 for axis in body_acc]))
        return float(rms)

    # ── core step ────────────────────────────────────────────────────────────

    def step(
        self,
        raw_window: NDArray[np.float64],
        return_probs: bool = False,
    ) -> dict:
        """
        Process one incoming 128-point window.

        Args
        ----
        raw_window : (9, 128)
            Raw 9-channel inertial signal window (normalised or raw; the
            binary model was trained on raw windows, full model expects
            normalised windows).
        return_probs : bool
            If True, also return full-model class probabilities.

        Returns
        -------
        dict with keys:
            state            : current State after this step
            event            : str description of what just happened
            acc_variance     : acc variance at this step
            predicted_label  : human-readable activity name (or None)
            confidence       : float in (0,1) (or None)
        """
        var = self.acc_variance(raw_window)
        result: dict = {
            "state": self._state.name,
            "event": "none",
            "acc_variance": var,
            "predicted_label": None,
            "confidence": None,
        }

        # ── IDLE ──────────────────────────────────────────────────────────────
        if self._state == State.IDLE:
            if var > self.upper_th:
                self._state = State.CHECK
                self._acc_buffer = [raw_window]
                result["event"] = f"IDLE→CHECK (var={var:.4f} > {self.upper_th:.4f})"
            else:
                result["event"] = "IDLE (no change)"

        # ── CHECK ────────────────────────────────────────────────────────────
        elif self._state == State.CHECK:
            self._acc_buffer.append(raw_window)
            if len(self._acc_buffer) >= self.history_len:
                feat = self._stack_buffer(self._acc_buffer)
                pred = int(self.binary_model.predict(feat)[0])
                if pred == 1:                      # real activity
                    self._state = State.ACTIVE
                    self._full_buffer = list(self._acc_buffer)
                    result["event"] = f"CHECK→ACTIVE (binary=real_activity)"
                else:                               # disturbance
                    self._state = State.IDLE
                    self._acc_buffer = []
                    result["event"] = f"CHECK→IDLE (binary=disturbance)"

        # ── ACTIVE ───────────────────────────────────────────────────────────
        elif self._state == State.ACTIVE:
            self._full_buffer.append(raw_window)
            label, confidence = self._run_full_model(self._full_buffer[-1:])
            result["predicted_label"] = LABEL_NAMES[label]
            result["confidence"] = float(confidence)

            if var < self.lower_th:
                self._state = State.IDLE
                self._full_buffer = []
                result["event"] = f"ACTIVE→IDLE (var={var:.4f} < {self.lower_th:.4f})"
            else:
                result["event"] = "ACTIVE (full model running)"

        return result

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _stack_buffer(windows):
        """Stack a list of (9,128) windows into (1, N*9, 128) for the RF."""
        stacked = np.concatenate(windows, axis=0)   # (N*9, 128)
        from .state_machine_features import extract_features
        return extract_features(stacked).reshape(1, -1)

    def _run_full_model(self, windows):
        """Run the PatchTST+GAF model on the latest window(s)."""
        stacked = np.stack(windows, axis=0)          # (B, 9, 128)
        t = torch.from_numpy(stacked).float()
        mean = t.mean(dim=(0, 2), keepdim=True)
        std  = t.std(dim=(0, 2), keepdim=True)
        std  = torch.where(std < 1e-6, torch.ones_like(std), std)
        t = (t - mean) / std
        t = t.to(self.device)

        with torch.no_grad():
            logits = self.full_model(t)
            probs  = torch.softmax(logits, dim=1)
            conf,  preds = probs.max(dim=1)
        return int(preds[0].item()), float(conf[0].item())

    # ── batch evaluation (for simulation / evaluation runs) ───────────────────

    def evaluate_sequence(
        self,
        X: NDArray[np.float64],    # (N, 9, 128)
        y_true: NDArray[np.int64],  # (N,)  0-5 labels
    ) -> dict:
        """
        Run the state machine on a pre-recorded sequence of windows
        and collect metrics.

        Returns
        -------
        dict with keys:
            state_history    : list of State names
            event_history    : list of event strings
            true_labels      : ground-truth labels
            predicted_labels : predicted labels while ACTIVE (None elsewhere)
            confidences      : confidence scores while ACTIVE (None elsewhere)
            full_model_invocations : int
            idle_steps       : int
            active_steps     : int
            check_steps      : int
        """
        state_hist   = []
        event_hist   = []
        pred_hist    = []
        conf_hist    = []

        for i in range(len(X)):
            r = self.step(X[i])
            state_hist.append(r["state"])
            event_hist.append(r["event"])
            pred_hist.append(r.get("predicted_label"))
            conf_hist.append(r.get("confidence"))

        n_idle   = sum(1 for s in state_hist if s == "IDLE")
        n_check  = sum(1 for s in state_hist if s == "CHECK")
        n_active = sum(1 for s in state_hist if s == "ACTIVE")

        # Accuracy only over ACTIVE steps
        active_mask = [s == "ACTIVE" for s in state_hist]
        active_true = [LABEL_NAMES[int(y_true[i])] for i, m in enumerate(active_mask) if m]
        active_pred = [p for p, m in zip(pred_hist, active_mask) if m]
        active_conf = [c for c, m in zip(conf_hist, active_mask) if m]

        acc = np.mean([1 for t, p in zip(active_true, active_pred) if t == p]) \
              if active_pred else 0.0

        return {
            "state_history":             state_hist,
            "event_history":             event_hist,
            "true_labels":               [LABEL_NAMES[int(v)] for v in y_true],
            "predicted_labels":          pred_hist,
            "confidences":              conf_hist,
            "full_model_invocations":    n_active,
            "idle_steps":               n_idle,
            "check_steps":              n_check,
            "active_steps":             n_active,
            "active_accuracy":          float(acc),
            "active_true":              active_true,
            "active_pred":              active_pred,
            "active_conf":              active_conf,
        }

    # ── threshold calibration ────────────────────────────────────────────────

    @staticmethod
    def calibrate_thresholds(
        X: NDArray[np.float64],
        y_true: NDArray[np.int64],
    ) -> Thresholds:
        """
        Find optimal upper/lower thresholds by sweeping on validation data.
        Uses the gap between IDLE and ACTIVE to set hysteresis.
        """
        vars_ = np.array([HARStateMachine.acc_variance(x) for x in X])

        # Ground-truth labels
        idle_mask   = np.array([int(v) in IDLE_LABELS    for v in y_true])
        active_mask = np.array([int(v) in ACTIVE_LABELS for v in y_true])

        idle_vars   = vars_[idle_mask]
        active_vars = vars_[active_mask]

        if len(active_vars) == 0:
            upper, lower = 0.1416, 0.0418
            print("  [Warning] No active windows in calibration set; using defaults.")
        else:
            upper = float(np.percentile(active_vars, 5))
            lower = float(np.percentile(idle_vars, 95)) if len(idle_vars) > 0 else 0.0

        gap = max(0.01, (upper - lower) * 0.1)
        lower = lower - gap
        upper = upper + gap

        print(f"  Calibration → upper_th = {upper:.4f}, lower_th = {lower:.4f}")
        idle_n = len(idle_vars); act_n = len(active_vars)
        print(f"    idle  variance ({idle_n:>4}): mean={idle_vars.mean():.4f}  "
              f"std={idle_vars.std():.4f}  95%={np.percentile(idle_vars,95):.4f}")
        print(f"    active variance ({act_n:>4}): mean={active_vars.mean():.4f}  "
              f"std={active_vars.std():.4f}  5%={np.percentile(active_vars,5):.4f}")

        return Thresholds(upper=max(upper, lower + 1e-4), lower=lower)
