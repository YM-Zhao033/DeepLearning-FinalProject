"""
Flask web server for the HAR Health Assistant Dashboard.

Usage
─────
  python -m src.web.app
  # or from project root: python src/web/app.py
  # then open http://localhost:5001

Endpoints
─────────
  GET  /                  — dashboard HTML
  GET  /api/status        — full health status (JSON)
  GET  /api/history       — session activity history (JSON)
  POST /api/simulate      — inject a simulated activity event
  POST /api/reset         — reset the session
  GET  /api/llm_available — check Ollama connection
  POST /api/demo/start    — start auto-demo loop
  POST /api/demo/stop     — stop auto-demo loop
  GET  /api/demo/status   — demo running status
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Resolve project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.models.health_assistant import HealthAssistant

# ── Flask app factory ─────────────────────────────────────────────────────────

def create_app() -> Flask:
    """
    Application factory.  Creates and configures the Flask app but does NOT
    start the development server (let the caller decide).
    """
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
        static_url_path="/static",
    )
    app.config["JSON_AS_ASCII"] = False

    # ── singleton health assistant instance ───────────────────────────────────
    _assistant: HealthAssistant | None = None

    def get_assistant() -> HealthAssistant:
        nonlocal _assistant
        if _assistant is None:
            _assistant = HealthAssistant(calorie_weight_kg=65.0, llm_model="qwen2.5:7b")
            _assistant.activity_log.load()
        return _assistant

    # ── routes ──────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/status")
    def api_status():
        ha = get_assistant()
        return jsonify(ha.get_full_status())

    @app.route("/api/history")
    def api_history():
        ha = get_assistant()
        return jsonify({
            "calorie_history": ha.cal_tracker._history[-50:],
            "activity_log": ha.activity_log.get_recent_days(7),
            "recent_states": ha.state_agg.recent_states,
        })

    @app.route("/api/simulate", methods=["POST"])
    def api_simulate():
        ha = get_assistant()
        data = request.get_json() or {}
        activity = data.get("activity", "SITTING")
        sim_s = float(data.get("sim_s", 0.0))

        LABEL_TO_STATE = {
            "SITTING": "IDLE", "STANDING": "IDLE", "LAYING": "IDLE", "IDLE": "IDLE",
            "WALKING": "ACTIVE", "WALKING_UPSTAIRS": "ACTIVE",
            "WALKING_DOWNSTAIRS": "ACTIVE", "CHECK": "CHECK",
        }
        state = LABEL_TO_STATE.get(activity, "IDLE")

        alerts, summary = ha.step(state, activity=activity, sim_s=sim_s)
        tip = ha.get_health_tip()

        return jsonify({
            "alerts": alerts,
            "summary": summary,
            "tip": tip,
            "status": ha.get_full_status(),
        })

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        ha = get_assistant()
        ha.reset_session()
        return jsonify({"ok": True, "message": "Session reset."})

    @app.route("/api/llm_available")
    def api_llm_available():
        ha = get_assistant()
        return jsonify({
            "available": ha.llm.is_available(),
            "models": ha.llm.list_models(),
        })

    @app.route("/api/activity_log")
    def api_activity_log():
        ha = get_assistant()
        return jsonify(ha.activity_log.get_recent_days(7))

    @app.route("/api/trend")
    def api_trend():
        ha = get_assistant()
        return jsonify(ha.get_trend())

    @app.route("/api/llm_tip", methods=["POST"])
    def api_llm_tip():
        ha = get_assistant()
        data = request.get_json() or {}
        lang = data.get("language", "zh")
        summary = ha.state_agg.get_summary()
        goal_progress = ha.get_goal_progress()
        trend = ha.get_trend()
        tip = ha.llm.generate_health_tip(
            summary, ha.cal_tracker, ha.intensity_grader,
            goal_progress, trend, language=lang,
        )
        return jsonify({"tip": tip})

    # ── auto-demo loop ───────────────────────────────────────────────────────

    DEMO_ACTIVITIES = [
        ("SITTING",           30),
        ("STANDING",          15),
        ("WALKING",           20),
        ("SITTING",           45),
        ("WALKING_UPSTAIRS",  10),
        ("WALKING",           15),
        ("WALKING_DOWNSTAIRS", 8),
        ("SITTING",           60),
        ("LAYING",            20),
        ("WALKING",           25),
        ("SITTING",           30),
    ]

    _demo_running = False
    _demo_thread: threading.Thread | None = None

    def _demo_loop():
        nonlocal _demo_running
        ha = get_assistant()
        sim_s = 0.0
        idx = 0
        while _demo_running:
            activity, duration = DEMO_ACTIVITIES[idx % len(DEMO_ACTIVITIES)]
            for _ in range(max(1, int(duration))):
                state_map = {
                    "SITTING": "IDLE", "STANDING": "IDLE", "LAYING": "IDLE", "IDLE": "IDLE",
                    "WALKING": "ACTIVE", "WALKING_UPSTAIRS": "ACTIVE",
                    "WALKING_DOWNSTAIRS": "ACTIVE", "CHECK": "CHECK",
                }
                state = state_map.get(activity, "IDLE")
                ha.step(state, activity=activity, sim_s=sim_s)
                sim_s += 2.56
                time.sleep(0.01)
            idx += 1
            time.sleep(0.5)

    @app.route("/api/demo/start", methods=["POST"])
    def api_demo_start():
        nonlocal _demo_running, _demo_thread
        if not _demo_running:
            _demo_running = True
            _demo_thread = threading.Thread(target=_demo_loop, daemon=True)
            _demo_thread.start()
        return jsonify({"ok": True, "running": _demo_running})

    @app.route("/api/demo/stop", methods=["POST"])
    def api_demo_stop():
        nonlocal _demo_running
        _demo_running = False
        return jsonify({"ok": True, "running": _demo_running})

    @app.route("/api/demo/status")
    def api_demo_status():
        nonlocal _demo_running
        return jsonify({"running": _demo_running})

    return app


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    print("=" * 60)
    print("  HAR Health Assistant Dashboard")
    print("  → http://localhost:5001")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
