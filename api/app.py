"""
Flask API serving the short-range tornado forecast (contract: api/README.md).

A read-only layer over the forecasts the scheduler writes to disk
(scripts/09_serving/run_forecast_job.py); no request handler loads a model or
pulls HRRR data. Everything lives in src/tornado_predictor/forecast_app.py.

Run locally:
    python scripts/09_serving/run_forecast_job.py     # fills data/live/ (needs network)
    python api/app.py                                 # serves on :5060 (override with PORT)

Test:
    curl http://localhost:5060/health
    curl http://localhost:5060/forecast/latest
"""
import os

from tornado_predictor.forecast_app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5060)), debug=False)
