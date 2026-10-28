"""
Health Assistant core logic:
  - Calorie calculation using MET values
  - State aggregation (time-in-state, sitting alerts)
  - Activity intensity grading (WHO-based)
  - Goal management with adaptive target adjustment
  - Activity log for trend analysis
  - LLM health tips via Ollama API
  - Conversation history management
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests

# ──────────────────────────────────────────────────────────────────────────────
#  MET-based calorie calculator
# ──────────────────────────────────────────────────────────────────────────────

# Metabolic Equivalent of Task (MET) values
# Reference: Ainsworth et al. 2011 Compendium of Physical Activities
# 1 MET = energy cost of sitting quietly

MET_TABLE = {
    "LAYING":             0.95,
    "SITTING":            1.00,
    "STANDING":           1.30,
    "WALKING":            3.50,
    "WALKING_UPSTAIRS":   8.80,
    "WALKING_DOWNSTAIRS": 5.30,
    "IDLE":               1.00,
    "CHECK":              1.20,
}

# Activity intensity classification (WHO guidelines: 150 min/week moderate OR 75 min/week vigorous)
# Map activities to intensity levels: light=0, moderate=1, vigorous=2
INTENSITY_MAP = {
    "LAYING":             0,   # light
    "SITTING":            0,   # light
    "STANDING":           0,   # light
    "WALKING":            1,   # moderate
    "WALKING_UPSTAIRS":   2,   # vigorous
    "WALKING_DOWNSTAIRS": 1,   # moderate
    "IDLE":               0,
    "CHECK":              0,
}
INTENSITY_NAMES = {0: "久坐/低强度", 1: "中等强度", 2: "高强度"}

# Default body weight (kg)
DEFAULT_WEIGHT_KG = 65.0

# Sitting alert threshold (minutes)
SITTING_ALERT_MINUTES = 45
SITTING_MODERATE_MINUTES = 20


# ──────────────────────────────────────────────────────────────────────────────
#  Calorie Tracker
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CalorieTracker:
    """Track cumulative calories burned based on detected activity."""
    weight_kg: float = DEFAULT_WEIGHT_KG
    cumulative_cal: float = 0.0
    session_start_sim_s: float = 0.0
    last_activity_sim_s: float = 0.0
    _history: list = field(default_factory=list)

    def update(self, activity_name: str, duration_s: float) -> float:
        """Add calories burned during a time window (duration_s in seconds)."""
        met = MET_TABLE.get(activity_name, 1.0)
        cal = met * self.weight_kg * (duration_s / 3600.0)
        self.cumulative_cal += cal
        self.last_activity_sim_s = self.session_start_sim_s + duration_s
        self._history.append({
            "sim_s": round(self.last_activity_sim_s, 2),
            "activity": activity_name,
            "cal": round(cal, 3),
            "cumulative": round(self.cumulative_cal, 1),
        })
        return cal

    @property
    def session_duration_min(self) -> float:
        return self.last_activity_sim_s / 60.0

    def reset(self):
        self.cumulative_cal = 0.0
        self.session_start_sim_s = 0.0
        self.last_activity_sim_s = 0.0
        self._history.clear()


# ──────────────────────────────────────────────────────────────────────────────
#  Activity Log — daily record for trend analysis
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ActivityLog:
    """
    Persistent daily activity log for trend analysis.
    Stores per-activity cumulative time (in minutes) and calories per day.
    """
    log_path: Path = field(default_factory=Path("logs/health_assistant/activity_log.json"))

    _daily: dict = field(default_factory=dict)
    _today_key: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def _today_key_now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def push(self, activity: str, duration_s: float, calories: float):
        """Append a new record for the current day."""
        day = self._today_key_now()
        if day not in self._daily:
            self._daily[day] = {
                "activities": {},
                "total_cal": 0.0,
                "total_active_min": 0.0,
                "total_sitting_min": 0.0,
            }
        record = self._daily[day]
        record["activities"][activity] = record["activities"].get(activity, 0.0) + duration_s / 60.0
        record["total_cal"] += calories

    def finalize_window(self, sitting_min: float, active_min: float):
        """Update sitting/active minutes for the current rolling window."""
        day = self._today_key_now()
        if day not in self._daily:
            self._daily[day] = {
                "activities": {},
                "total_cal": 0.0,
                "total_active_min": 0.0,
                "total_sitting_min": 0.0,
            }
        self._daily[day]["total_sitting_min"] = sitting_min
        self._daily[day]["total_active_min"] = active_min

    def get_recent_days(self, days: int = 7) -> list:
        """Return a list of daily records for the last N days."""
        result = []
        today = datetime.now()
        for i in range(days):
            d = (today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            result.append(self._daily.get(d, {
                "date": d,
                "activities": {},
                "total_cal": 0.0,
                "total_active_min": 0.0,
                "total_sitting_min": 0.0,
            }))
        return result

    def save(self):
        """Persist log to disk."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self._daily, f, ensure_ascii=False, indent=2)

    def load(self):
        """Load log from disk."""
        if self.log_path.exists():
            with open(self.log_path, encoding="utf-8") as f:
                self._daily = json.load(f)

    def reset(self):
        self._daily.clear()

    def get_trend(self) -> dict:
        """
        Compute weekly trend: returns dict with 'trend_direction' (up/down/stable)
        and 'change_percent'.
        """
        recent = self.get_recent_days(7)
        if len(recent) < 2:
            return {"trend_direction": "stable", "change_percent": 0.0}

        # Use total active minutes as trend metric
        active_vals = [r.get("total_active_min", 0.0) for r in recent if r.get("date")]
        if len(active_vals) < 2:
            return {"trend_direction": "stable", "change_percent": 0.0}

        recent_avg = np.mean(active_vals[:len(active_vals)//2]) if len(active_vals) >= 2 else 0
        older_avg = np.mean(active_vals[len(active_vals)//2:]) if len(active_vals) >= 2 else 0

        if older_avg == 0:
            return {"trend_direction": "stable", "change_percent": 0.0}

        pct = (recent_avg - older_avg) / older_avg * 100
        if pct > 10:
            direction = "up"
        elif pct < -10:
            direction = "down"
        else:
            direction = "stable"
        return {"trend_direction": direction, "change_percent": round(pct, 1)}


# ──────────────────────────────────────────────────────────────────────────────
#  Goal Manager — adaptive daily targets
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GoalManager:
    """
    Manage personalized daily activity goals with adaptive adjustment.
    Goals are adjusted based on historical completion rate.
    """
    default_daily_active_min: float = 30.0
    default_daily_cal: float = 200.0

    _current_active_goal: float = 30.0
    _current_cal_goal: float = 200.0
    _completion_history: list = field(default_factory=list)  # last 7 days: 0 or 1

    # Threshold for adaptive adjustment
    _consecutive_achieved: int = 0
    _consecutive_missed: int = 0
    _achievement_threshold: int = 3   # 3 days of over-achievement -> raise goal
    _miss_threshold: int = 3         # 3 days of under-achievement -> lower goal
    _step_active: float = 5.0       # adjust active goal by 5 min per step
    _step_cal: float = 30.0         # adjust calorie goal by 30 kcal per step

    def update_progress(self, achieved_active: float, achieved_cal: float):
        """Called at end of day to record goal achievement."""
        active_pct = achieved_active / self._current_active_goal
        cal_pct = achieved_cal / self._current_cal_goal
        # Consider day achieved if both >= 80% of goals met
        achieved = 1 if (active_pct >= 0.8 and cal_pct >= 0.8) else 0
        self._completion_history.append(achieved)
        if len(self._completion_history) > 7:
            self._completion_history.pop(0)

        # Adaptive adjustment
        if len(self._completion_history) >= self._achievement_threshold:
            if all(self._completion_history[-self._achievement_threshold:]):
                self._consecutive_achieved += 1
                if self._consecutive_achieved >= self._achievement_threshold:
                    # Raise goals
                    self._current_active_goal += self._step_active
                    self._current_cal_goal += self._step_cal
                    self._consecutive_achieved = 0
                    self._consecutive_missed = 0
            elif not any(self._completion_history[-self._miss_threshold:]):
                self._consecutive_missed += 1
                if self._consecutive_missed >= self._miss_threshold:
                    # Lower goals
                    self._current_active_goal = max(10.0, self._current_active_goal - self._step_active)
                    self._current_cal_goal = max(50.0, self._current_cal_goal - self._step_cal)
                    self._consecutive_missed = 0
                    self._consecutive_achieved = 0

    def get_goal_progress(self, achieved_active: float, achieved_cal: float) -> dict:
        """Return goal progress info for display."""
        active_pct = min(achieved_active / self._current_active_goal * 100, 100)
        cal_pct = min(achieved_cal / self._current_cal_goal * 100, 100)
        active_remaining = max(0.0, self._current_active_goal - achieved_active)
        cal_remaining = max(0.0, self._current_cal_goal - achieved_cal)

        if active_pct >= 100 and cal_pct >= 100:
            level = "goal_met"
        elif active_pct >= 50 or cal_pct >= 50:
            level = "goal_progress"
        else:
            level = "goal_start"

        return {
            "active_goal": self._current_active_goal,
            "cal_goal": self._current_cal_goal,
            "achieved_active": achieved_active,
            "achieved_cal": achieved_cal,
            "active_pct": round(active_pct, 1),
            "cal_pct": round(cal_pct, 1),
            "active_remaining": round(active_remaining, 1),
            "cal_remaining": round(cal_remaining, 1),
            "level": level,
        }

    def reset(self):
        self._current_active_goal = self.default_daily_active_min
        self._current_cal_goal = self.default_daily_cal
        self._completion_history.clear()
        self._consecutive_achieved = 0
        self._consecutive_missed = 0


# ──────────────────────────────────────────────────────────────────────────────
#  Activity Intensity Grader (WHO-based)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IntensityGrader:
    """
    Grade daily activity into three levels (Active/Moderate/Insufficient)
    based on WHO weekly recommendations:
      - >= 150 min/week moderate  OR  >= 75 min/week vigorous  => Active
      - >= 75 min/week moderate  OR  >= 37.5 min/week vigorous => Moderate
      - otherwise => Insufficient

    Daily approximation:
      - Active: >= 21.4 min moderate or >= 10.7 min vigorous
      - Moderate: >= 10.7 min moderate or >= 5.4 min vigorous
      - Insufficient: otherwise
    """
    DAILY_ACTIVE_THRESHOLD_MIN = 21.4
    DAILY_MODERATE_THRESHOLD_MIN = 10.7
    DAILY_VIGOROUS_THRESHOLD_MIN = 10.7
    DAILY_VIGOROUS_MODERATE_EQUIV = 5.4  # 75/14 ≈ 5.4

    # Cumulative minutes
    _moderate_min_today: float = 0.0
    _vigorous_min_today: float = 0.0
    _day_start_sim_s: float = 0.0

    def update(self, activity: str, duration_s: float):
        intensity = INTENSITY_MAP.get(activity, 0)
        dur_min = duration_s / 60.0
        if intensity == 2:
            self._vigorous_min_today += dur_min
        elif intensity == 1:
            self._moderate_min_today += dur_min

    def get_today_minutes(self) -> dict:
        return {
            "moderate_min": round(self._moderate_min_today, 1),
            "vigorous_min": round(self._vigorous_min_today, 1),
        }

    @property
    def total_equivalent_min(self) -> float:
        """Convert vigorous minutes to moderate-equivalent (2:1 ratio)."""
        return self._moderate_min_today + self._vigorous_min_today * 2

    def grade(self) -> str:
        """Return 'active', 'moderate', or 'insufficient'."""
        equiv = self.total_equivalent_min
        if equiv >= self.DAILY_ACTIVE_THRESHOLD_MIN:
            return "active"
        elif equiv >= self.DAILY_MODERATE_THRESHOLD_MIN:
            return "moderate"
        return "insufficient"

    def grade_label_zh(self) -> str:
        mapping = {
            "active": "活跃 🌟",
            "moderate": "适度 👍",
            "insufficient": "不足 ⚠️",
        }
        return mapping.get(self.grade(), "未知")

    def reset(self):
        self._moderate_min_today = 0.0
        self._vigorous_min_today = 0.0
        self._day_start_sim_s = 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  State Aggregator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StateAggregator:
    """
    Maintains a rolling view of recent activity windows and exposes
    derived metrics (sitting time, active minutes, current state).
    """
    window_duration_s: float = 2.56
    max_history_min: int = 30

    _states: list = field(default_factory=list)
    _last_llm_emit_sim_s: float = -9999.0
    _llm_interval_sim_s: float = 60.0

    def push(self, state: str, activity: Optional[str] = None,
             acc_variance: float = 0.0, confidence: float = 0.0,
             sim_s: float = 0.0):
        self._states.append({
            "sim_s": sim_s,
            "state": state,
            "activity": activity,
            "acc_variance": float(acc_variance),
            "confidence": float(confidence),
        })
        cutoff = sim_s - self.max_history_min * 60
        self._states = [s for s in self._states if s["sim_s"] > cutoff]

    @property
    def current_state(self) -> str:
        return self._states[-1]["state"] if self._states else "IDLE"

    @property
    def current_activity(self) -> Optional[str]:
        return self._states[-1].get("activity") if self._states else None

    @property
    def current_sim_s(self) -> float:
        return self._states[-1]["sim_s"] if self._states else 0.0

    @property
    def sitting_duration_min(self) -> float:
        idle_entries = [s for s in self._states if s["state"] == "IDLE"]
        if len(idle_entries) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(idle_entries)):
            dt = idle_entries[i]["sim_s"] - idle_entries[i-1]["sim_s"]
            total += dt / 60.0
        return total

    @property
    def active_duration_min(self) -> float:
        return sum(
            (self._states[i]["sim_s"] - self._states[i-1]["sim_s"])
            for i in range(1, len(self._states))
            if self._states[i]["state"] == "ACTIVE"
        ) / 60.0

    def get_summary(self) -> dict:
        total_windows = len(self._states)
        if total_windows == 0:
            return {"status": "No data yet", "sitting_min": 0, "active_min": 0,
                    "cal": 0.0, "current": "IDLE", "suggestion": "请开始活动。"}

        state_counts = {}
        activity_counts = {}
        for s in self._states:
            state_counts[s["state"]] = state_counts.get(s["state"], 0) + 1
            if s["activity"]:
                activity_counts[s["activity"]] = activity_counts.get(s["activity"], 0) + 1

        dominant_state = max(state_counts, key=state_counts.get)
        dominant_activity = max(activity_counts, key=activity_counts.get) if activity_counts else None

        sitting_min = self.sitting_duration_min
        active_min = self.active_duration_min

        alerts = []
        if sitting_min > SITTING_ALERT_MINUTES:
            alerts.append(f"\u26a0\ufe0f 久坐警告：已连续久坐超过 {int(sitting_min)} 分钟，建议起身活动！")
        elif sitting_min > SITTING_MODERATE_MINUTES:
            alerts.append(f"\ud83d\udca1 提示：已坐 {int(sitting_min)} 分钟，记得偶尔站起来伸展一下。")
        if dominant_state == "ACTIVE" and active_min > 5:
            alerts.append(f"\ud83c\udfc3 运动状态：已持续活动 {active_min:.0f} 分钟，继续保持！")

        return {
            "status":           dominant_state,
            "current_activity": dominant_activity or self.current_activity,
            "sitting_min":     round(sitting_min, 1),
            "active_min":       round(active_min, 1),
            "total_windows":   total_windows,
            "alerts":          alerts,
            "state_counts":    state_counts,
            "activity_counts": activity_counts,
        }

    def should_emit_llm(self) -> bool:
        cur_sim_s = self.current_sim_s
        if cur_sim_s - self._last_llm_emit_sim_s >= self._llm_interval_sim_s:
            self._last_llm_emit_sim_s = cur_sim_s
            return True
        return False

    @property
    def recent_states(self) -> list:
        return self._states[-10:]

    def reset(self):
        self._states.clear()
        self._last_llm_emit_sim_s = -9999.0


# ──────────────────────────────────────────────────────────────────────────────
#  Ollama LLM Client
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMClient:
    """
    Thin wrapper around the Ollama API for health tips generation.
    Falls back gracefully when Ollama is not available.
    """
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    _available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            self._available = (r.status_code == 200)
        except Exception:
            self._available = False
        return self._available

    def generate_health_tip(self, summary: dict, calorie_tracker: CalorieTracker,
                           intensity_grader: IntensityGrader,
                           goal_progress: dict,
                           trend: dict,
                           language: str = "zh") -> str:
        if not self.is_available():
            return self._fallback_tip(summary, calorie_tracker, intensity_grader, goal_progress)

        sitting = summary.get("sitting_min", 0)
        active = summary.get("active_min", 0)
        cur_act = summary.get("current_activity") or summary.get("status", "未知")
        alerts = "\n".join(summary.get("alerts", [])) or "暂无"
        cal = calorie_tracker.cumulative_cal

        intensity_info = intensity_grader.get_today_minutes()
        intensity_label = intensity_grader.grade_label_zh()

        goal_active_pct = goal_progress.get("active_pct", 0)
        goal_cal_pct = goal_progress.get("cal_pct", 0)

        trend_info = ""
        td = trend.get("trend_direction", "stable")
        tp = trend.get("change_percent", 0)
        if td == "up":
            trend_info = f"近期活动趋势上升（+{tp}%）"
        elif td == "down":
            trend_info = f"近期活动趋势下降（{tp}%），请注意增加活动"
        else:
            trend_info = "近期活动趋势保持稳定"

        sys_prompt = (
            "你是一个专业的健康管理助手AI。请根据用户最近的运动监测数据，"
            "用简洁友好的语言（1-3句话）给出健康建议。不要重复用户已知的信息。"
        )
        user_prompt = (
            f"用户当前状态：{cur_act}\n"
            f"过去30分钟久坐时间：{sitting:.0f} 分钟\n"
            f"过去30分钟活动时间：{active:.0f} 分钟\n"
            f"本次会话累计消耗卡路里：{cal:.1f} kcal\n"
            f"今日活动强度等级：{intensity_label}（中等强度 {intensity_info['moderate_min']:.1f} min，"
            f"高强度 {intensity_info['vigorous_min']:.1f} min）\n"
            f"今日目标完成进度：活动时间 {goal_active_pct:.0f}%，卡路里 {goal_cal_pct:.0f}%\n"
            f"活动趋势：{trend_info}\n"
            f"提醒事项：\n{alerts}\n\n"
            f"请给出简洁、实用、鼓励性的健康小贴士（中文，1-3句话）。"
        )

        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 128,
                },
            }
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=30,
            )
            if r.status_code == 200:
                return r.json().get("message", {}).get("content", "").strip()
        except Exception:
            pass

        return self._fallback_tip(summary, calorie_tracker, intensity_grader, goal_progress)

    @staticmethod
    def _fallback_tip(summary: dict, calorie_tracker: CalorieTracker,
                      intensity_grader: IntensityGrader, goal_progress: dict) -> str:
        sitting = summary.get("sitting_min", 0)
        active = summary.get("active_min", 0)
        cur = summary.get("current_activity") or summary.get("status", "未知")
        cal = calorie_tracker.cumulative_cal
        intensity_label = intensity_grader.grade_label_zh()
        goal_pct = goal_progress.get("active_pct", 0)

        if sitting > SITTING_ALERT_MINUTES:
            return ("\u26a0\ufe0f 久坐提醒：已连续久坐超过 45 分钟！"
                    "请起身活动一下，喝杯水，伸展一下身体。")
        if goal_pct >= 100:
            return (f"\ud83c\udfc6 今日目标已达成！当前活动强度{intensity_label}，"
                    f"已消耗 {cal:.1f} kcal，继续保持！")
        if cur in ("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"):
            return (f"\ud83c\udfc3 继续保持！当前正在 {cur}，"
                    f"已累计消耗 {cal:.1f} kcal，目标完成{intensity_label}。")
        if active > 10:
            return (f"\ud83d\udcaa 活动状态良好！过去 {active:.0f} 分钟运动量充足，"
                    f"今日目标完成{goal_pct:.0f}%，已消耗 {cal:.1f} kcal。")
        if sitting > 10:
            return (f"\ud83d\udca1 已坐 {sitting:.0f} 分钟，建议站起来活动一下，"
                    f"当前目标完成{intensity_label}。")
        return "\ud83c\udf31 健康助手已就绪，实时监测您的活动状态。"

    def list_models(self) -> list:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []


# ──────────────────────────────────────────────────────────────────────────────
#  Health Assistant — main orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class HealthAssistant:
    """
    Top-level orchestrator that wires the state machine to the UI and LLM.
    Integrates: CalorieTracker, StateAggregator, ActivityLog,
    GoalManager, IntensityGrader, and LLMClient.
    """
    def __init__(self,
                 calorie_weight_kg: float = DEFAULT_WEIGHT_KG,
                 llm_model: str = "qwen2.5:7b"):
        self.cal_tracker    = CalorieTracker(weight_kg=calorie_weight_kg)
        self.state_agg      = StateAggregator()
        self.activity_log  = ActivityLog()
        self.goal_mgr       = GoalManager()
        self.intensity_grader = IntensityGrader()
        self.llm            = LLMClient(model=llm_model)
        self.weight_kg      = calorie_weight_kg
        self._running       = False
        self._last_activity: Optional[str] = None
        self._activity_start_sim_s: float = 0.0
        self._current_sim_s: float = 0.0

    @property
    def running(self) -> bool:
        return self._running

    def step(self, state: str, activity: Optional[str] = None,
             acc_variance: float = 0.0, confidence: float = 0.0,
             sim_s: float = 0.0):
        """
        Called each time step. Updates all trackers and returns alerts.
        """
        self._current_sim_s = sim_s
        self.state_agg.push(state, activity, acc_variance, confidence, sim_s=sim_s)

        if self._last_activity is not None:
            duration = sim_s - self._activity_start_sim_s
            if duration > 0:
                cal = self.cal_tracker.update(self._last_activity, duration)
                self.intensity_grader.update(self._last_activity, duration)
                self.activity_log.push(self._last_activity, duration, cal)

        self._last_activity = activity or state
        self._activity_start_sim_s = sim_s

        summary = self.state_agg.get_summary()
        alerts = summary.get("alerts", [])
        return alerts, summary

    def get_health_tip(self) -> str:
        if not self.state_agg.should_emit_llm():
            return ""
        summary = self.state_agg.get_summary()
        goal_progress = self.get_goal_progress()
        trend = self.activity_log.get_trend()
        return self.llm.generate_health_tip(
            summary, self.cal_tracker, self.intensity_grader,
            goal_progress, trend,
        )

    def get_goal_progress(self) -> dict:
        """Return current goal progress."""
        return self.goal_mgr.get_progress(
            self.state_agg.active_duration_min,
            self.cal_tracker.cumulative_cal,
        )

    def get_intensity_info(self) -> dict:
        """Return today's intensity grading."""
        return {
            "grade": self.intensity_grader.grade(),
            "grade_label": self.intensity_grader.grade_label_zh(),
            "moderate_min": self.intensity_grader.get_today_minutes()["moderate_min"],
            "vigorous_min": self.intensity_grader.get_today_minutes()["vigorous_min"],
            "equivalent_min": round(self.intensity_grader.total_equivalent_min, 1),
        }

    def get_trend(self) -> dict:
        """Return activity trend over recent days."""
        return self.activity_log.get_trend()

    def get_full_status(self) -> dict:
        summary = self.state_agg.get_summary()
        goal_progress = self.get_goal_progress()
        intensity = self.get_intensity_info()
        trend = self.get_trend()
        active_30min = self.state_agg.active_duration_min
        return {
            "current_state":      self.state_agg.current_state,
            "current_activity":   self.state_agg.current_activity,
            "sitting_min":        summary.get("sitting_min", 0),
            "active_min":         active_30min,
            "calories_kcal":      round(self.cal_tracker.cumulative_cal, 1),
            "session_min":        round(self.cal_tracker.session_duration_min, 1),
            "weight_kg":         self.weight_kg,
            "llm_available":     self.llm.is_available(),
            "summary":            summary,
            "goal_progress":      goal_progress,
            "intensity":          intensity,
            "trend":              trend,
        }

    def reset_session(self):
        self.cal_tracker.reset()
        self.state_agg.reset()
        self.intensity_grader.reset()
        self.activity_log.reset()
        self._last_activity = None
        self._activity_start_sim_s = 0.0
        self._current_sim_s = 0.0
