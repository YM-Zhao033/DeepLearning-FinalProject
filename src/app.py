"""
Thin launcher for the HAR Health Assistant web dashboard.

Delegates to src.web.app.  Run from the project root:
    python app.py
    # or
    python -m src.web.app
"""

from src.web.app import create_app

if __name__ == "__main__":
    app = create_app()
    print("=" * 60)
    print("  HAR Health Assistant Dashboard")
    print("  → http://localhost:5001")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
