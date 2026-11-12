"""Capture MLflow UI screenshots as evidence of the tracked runs and the registry.

Usage: ``python scripts/capture_mlflow_screenshots.py <output_dir>`` (needs Playwright
with Chromium; the CI ``docker-pipeline`` job runs it after the pipeline finishes).
"""

import json
import os
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")
EXPERIMENT = os.environ.get("CHALLENGE_EXPERIMENT_NAME", "11mlet-purchase-intent")
MODEL = os.environ.get("CHALLENGE_REGISTERED_MODEL_NAME", "11mlet-purchase-intent")
ALIAS = os.environ.get("CHALLENGE_CHAMPION_ALIAS", "champion")
VIEWPORT = {"width": 1600, "height": 1000}
RENDER_WAIT_MS = 4000
DARK_MODE_SCRIPT = "localStorage.setItem('_mlflow_dark_mode_toggle_enabled', 'true')"


def api(path: str, payload: dict | None = None) -> dict:
    """Call the MLflow REST API (GET without payload, POST with one)."""
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    request = Request(f"{BASE_URL}/api/2.0/mlflow/{path}", data=data, headers=headers)
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def page_urls() -> dict[str, str]:
    """Build the UI address of each evidence page from the tracked experiment."""
    experiment = api(f"experiments/get-by-name?experiment_name={quote(EXPERIMENT)}")
    experiment_id = experiment["experiment"]["experiment_id"]
    runs = api("runs/search", {"experiment_ids": [experiment_id]})["runs"]
    run_ids = quote(json.dumps([run["info"]["run_id"] for run in runs]))
    champion = api(f"registered-models/alias?name={quote(MODEL)}&alias={ALIAS}")["model_version"]
    ui = f"{BASE_URL}/#"
    experiments = quote(json.dumps([experiment_id]))
    return {
        "mlflow_experimentos": f"{ui}/experiments/{experiment_id}",
        "mlflow_comparacao": f"{ui}/compare-runs?runs={run_ids}&experiments={experiments}",
        "mlflow_run_metricas": (
            f"{ui}/experiments/{experiment_id}/runs/{champion['run_id']}/model-metrics"
        ),
        "mlflow_model_registry": f"{ui}/models/{quote(MODEL)}/versions/{champion['version']}",
    }


def capture(urls: dict[str, str], output_dir: Path) -> None:
    """Open each page in headless Chromium and save a full-page screenshot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport=VIEWPORT, color_scheme="dark")
        context.add_init_script(DARK_MODE_SCRIPT)
        for name, url in urls.items():
            page = context.new_page()
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(RENDER_WAIT_MS)
            page.screenshot(path=str(output_dir / f"{name}.png"), full_page=True)
            page.close()
            print(f"{name}.png <- {url}")
        browser.close()


if __name__ == "__main__":
    capture(page_urls(), Path(sys.argv[1]))
