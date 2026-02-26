"""
app.py — Flask backend exposing:
  POST /train    → train + optional Optuna-style tuning
  POST /predict  → generate predictions + Excel report
  GET  /explain  → permutation-based feature importance
  GET  /monitor  → drift, rolling error, model registry
  GET  /versions → list saved model versions
  GET  /download/<filename> → download generated Excel files
"""

import os
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

from core import (
    CSV_PATH, BASE_DIR, LOG_DIR,
    load_data, train_test_split, detect_drift, ALL_FEATS,
)
from model import (
    tune_model, train, evaluate, explain as explain_model,
    save_model, load_model, list_versions,
)
from predict_engine import predict_date, build_excel, rolling_vc_error, log_prediction

app = Flask(__name__)
CORS(app)

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _json_safe(obj):
    """Recursively convert numpy/pandas types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    return obj


# ── /train ─────────────────────────────────────────────────────────────────────
@app.route("/train", methods=["POST"])
def train_endpoint():
    """
    Body (JSON, all optional):
    {
      "tune":    true/false  (default false — fast train),
      "n_iter":  20          (tuning candidates),
      "n_splits": 5          (CV folds),
      "retrain_full": true   (default true — retrain on full data after eval)
    }
    """
    body        = request.get_json(silent=True) or {}
    do_tune     = bool(body.get("tune", False))
    n_iter      = int(body.get("n_iter", 20))
    n_splits    = int(body.get("n_splits", 5))
    retrain_full = bool(body.get("retrain_full", True))

    try:
        df, road_params, road_order = load_data(CSV_PATH)
        train_df, test_df = train_test_split(df)

        # --- Tune ---
        best_params = None
        tuning_log  = None
        if do_tune:
            best_params = tune_model(train_df, n_iter=n_iter,
                                     n_splits=n_splits, verbose=False)
            tuning_log  = {k: str(v) for k, v in best_params.items()}

        # --- Train on train split ---
        model = train(train_df, params=best_params, verbose=False)

        # --- Evaluate on test split ---
        metrics = evaluate(model, test_df)

        # --- Retrain on full data ---
        if retrain_full:
            model = train(df, params=best_params, verbose=False)

        # --- Drift check ---
        drift = detect_drift(train_df, test_df, feats=ALL_FEATS[:8])

        # --- Save ---
        version = save_model(model, best_params or {}, metrics,
                             road_params, road_order, CSV_PATH)

        return jsonify({
            "status":    "ok",
            "version":   version,
            "tuned":     do_tune,
            "tuning":    tuning_log,
            "metrics":   _json_safe(metrics),
            "drift":     _json_safe(drift),
            "train_rows": len(train_df),
            "test_rows":  len(test_df),
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e),
                        "trace": traceback.format_exc()}), 500


# ── /predict ───────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Body (JSON, all optional):
    {
      "date":    "2026-02-18",   (default PRED_DATE)
      "version": "latest"
    }
    Returns predictions JSON + download URL for Excel report.
    """
    body    = request.get_json(silent=True) or {}
    version = body.get("version", "latest")
    date_str = body.get("date", "2026-02-18")

    try:
        target_date = pd.Timestamp(date_str)
    except Exception:
        return jsonify({"status": "error", "message": f"Invalid date: {date_str}"}), 400

    try:
        model, road_params, road_order, meta = load_model(version)
    except FileNotFoundError:
        return jsonify({"status": "error",
                        "message": "No trained model found. Run /train first."}), 404

    try:
        predictions, summary = predict_date(
            model, road_params, road_order, target_date)

        date_tag = target_date.strftime("%Y%m%d")
        xlsx_name = f"Traffic_LOS_{date_tag}.xlsx"
        xlsx_path = os.path.join(OUTPUTS_DIR, xlsx_name)
        build_excel(predictions, road_params, target_date, xlsx_path)

        # Log predictions for drift monitoring
        log_path = os.path.join(LOG_DIR, "prediction_log.csv")
        for row in summary:
            log_prediction(
                pred_vc=row["peak_vc"],
                actual_vc=0.0,
                road=row["road"],
                hour=-1,
                log_path=log_path
            )


        # Serializable prediction payload
        pred_payload = {}
        for road, hours in predictions.items():
            pred_payload[road] = [
                {k: int(v) if isinstance(v, (int, float)) else v
                 for k, v in zip(["car","moto","van","med_lorry","hvy_lorry","bus"],
                                 row)}
                for row in hours
            ]

        return jsonify({
            "status":      "ok",
            "date":        date_str,
            "model_version": meta.get("version"),
            "summary":     _json_safe(summary),
            "excel_url":   f"/download/{xlsx_name}",
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e),
                        "trace": traceback.format_exc()}), 500


# ── /explain ───────────────────────────────────────────────────────────────────
@app.route("/explain", methods=["GET"])
def explain_endpoint():
    """
    Query params: version (default latest), n_repeats (default 10)
    Returns permutation-importance ranking (SHAP-equivalent, V/C MAE objective).
    """
    version   = request.args.get("version", "latest")
    n_repeats = int(request.args.get("n_repeats", 10))

    try:
        model, road_params, road_order, meta = load_model(version)
        df, _, _ = load_data(CSV_PATH)
        importance = explain_model(model, df, n_repeats=n_repeats)
        return jsonify({
            "status":  "ok",
            "version": meta.get("version"),
            "importance": importance,
        })
    except FileNotFoundError:
        return jsonify({"status": "error",
                        "message": "No model found. Run /train first."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e),
                        "trace": traceback.format_exc()}), 500


# ── /monitor ───────────────────────────────────────────────────────────────────
@app.route("/monitor", methods=["GET"])
def monitor_endpoint():
    """
    Returns:
      - Data drift (PSI/KS) between train and test split
      - Prediction drift (rolling V/C MAE)
      - Feature integrity summary (null rates, range violations)
      - Model registry (all versions)
    """
    try:
        df, _, _ = load_data(CSV_PATH)
        train_df, test_df = train_test_split(df)

        # Drift
        drift = detect_drift(train_df, test_df)

        # Prediction drift / rolling error
        log_path = os.path.join(LOG_DIR, "prediction_log.csv")
        rolling  = rolling_vc_error(log_path)

        # Feature integrity
        null_rates = df[ALL_FEATS].isnull().mean().round(4).to_dict()
        integrity  = {f: {"null_pct": round(float(v)*100, 2)}
                      for f, v in null_rates.items()}

        # Dataset summary
        dataset_summary = {
            "rows":        len(df),
            "roads":       df["road"].nunique(),
            "days":        df["date"].nunique(),
            "date_range":  [str(df["date"].min().date()),
                            str(df["date"].max().date())],
            "los_dist":    df["los_grade"].value_counts().to_dict(),
        }

        # Model registry
        versions = list_versions()

        return jsonify({
            "status":          "ok",
            "drift":           _json_safe(drift),
            "rolling_vc_error": _json_safe(rolling),
            "feature_integrity": _json_safe(integrity),
            "dataset":         _json_safe(dataset_summary),
            "model_versions":  versions,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e),
                        "trace": traceback.format_exc()}), 500


# ── /versions ─────────────────────────────────────────────────────────────────
@app.route("/versions", methods=["GET"])
def versions_endpoint():
    try:
        return jsonify({"status": "ok", "versions": list_versions()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── /download ─────────────────────────────────────────────────────────────────
@app.route("/download/<filename>", methods=["GET"])
def download_endpoint(filename):
    """Serve generated Excel files."""
    try:
        return send_from_directory(OUTPUTS_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"status": "error", "message": f"File not found: {filename}"}), 404


@app.route("/", methods=["GET"])
@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Serve the web dashboard."""
    return send_file(os.path.join(BASE_DIR, "dashboard.html"))


# ── Health ─────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
