"""Ingestion scheduler: fetch -> clean -> validate -> load into Postgres, on
startup and then periodically via APScheduler. No manual step required.
"""
import logging
import os
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingestion")

INTERVAL_MINUTES = int(os.environ.get("INGESTION_INTERVAL_MINUTES", "60"))

app = FastAPI(title="Churn Ingestion Service")
scheduler = BackgroundScheduler()

last_run: dict = {"status": "pending", "finished_at": None, "detail": None}


def run_ingestion_job():
    global last_run
    try:
        result = pipeline.run()
        last_run = {
            "status": "ok",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "detail": result,
        }
    except Exception as e:
        logger.exception("ingestion job failed")
        last_run = {
            "status": "error",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "detail": str(e),
        }


@app.on_event("startup")
def startup():
    scheduler.add_job(run_ingestion_job, "date")  # run once immediately
    scheduler.add_job(run_ingestion_job, "interval", minutes=INTERVAL_MINUTES)
    scheduler.start()


@app.on_event("shutdown")
def shutdown():
    scheduler.shutdown()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return last_run
