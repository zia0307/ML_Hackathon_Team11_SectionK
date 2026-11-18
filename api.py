"""
PathSnap API (Laptop 1).

Run from the `code/` folder:
    py -m uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000

Endpoints
    GET  /health            -> server status
    GET  /trips             -> the reserved evaluation trips (ids only, no labels)
    POST /predict           -> body {"trip_id": "00000015"}; runs candidates -> classifier -> HMM/Viterbi,
                               then scores the result against server-side reference data
    GET  /docs              -> interactive API documentation

Ground-truth (.route) data stays on this server. It is blanked before prediction and is only used
afterwards to compute the accuracy numbers that are returned. Reference arcs are never sent to the UI.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pathsnap_pipeline as P

try:
    import HMM_route as H
except ImportError:  # file saved in lower case
    import hmm_route as H

MODEL_PATH = Path(os.getenv("PATHSNAP_MODEL", "results/best_tuned_model.joblib"))
HMM_CONFIG = Path(os.getenv("PATHSNAP_HMM_CONFIG", "hmm_results/hmm_config.json"))
EVAL_DIR = Path(os.getenv("PATHSNAP_EVAL_DIR", "evaluator_test"))
TOP_K = int(os.getenv("PATHSNAP_TOP_K", "15"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model %s and decoder config %s", MODEL_PATH, HMM_CONFIG)
    STATE["model"] = joblib.load(MODEL_PATH)
    STATE["params"] = H.load_params(HMM_CONFIG)
    STATE["trips"] = {}
    for trip_dir in H.find_trip_dirs(EVAL_DIR):
        trip = P.load_trip(trip_dir)
        # Reference arcs are computed once, kept server-side, and only used to score predictions.
        frame, _ = P.build_candidates_for_trip(trip, "reference", top_k=TOP_K)
        if frame.empty:
            log.warning("Trip %s produced no candidates; skipped", trip["trip_id"])
            continue
        ref = frame.drop_duplicates("obs_idx").set_index("obs_idx")["reference_arc_id"]
        STATE["trips"][trip["trip_id"]] = {
            "trip": trip, "ref": ref, "graph": H.TripGraph(trip["arcs"], len(trip["nodes"])),
        }
        log.info("Loaded trip %s (%d GPS points)", trip["trip_id"], len(trip["track"]))
    log.info("Ready: %d evaluation trips", len(STATE["trips"]))
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)
# The UI runs on another laptop (a different origin), so browsers need CORS enabled.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info("REQUEST %s %s from %s -> %s (%.0f ms)", request.method, request.url.path, client,
             response.status_code, 1000 * (time.perf_counter() - t0))
    return response


class PredictRequest(BaseModel):
    trip_id: str


def route_coords(route: list, trip: dict) -> list:
    """Polyline [lat, lon] following the arcs of a route (consecutive arcs share nodes)."""
    nodes, arcs = trip["nodes"], trip["arcs"]
    coords, prev_end = [], None
    for a in route:
        s, e = int(arcs[a, 0]), int(arcs[a, 1])
        if prev_end != s:
            coords.append([round(float(nodes[s, 1]), 6), round(float(nodes[s, 0]), 6)])
        coords.append([round(float(nodes[e, 1]), 6), round(float(nodes[e, 0]), 6)])
        prev_end = e
    return coords


def score(pred, ref, graph) -> dict:
    r = ref.reindex(pred["obs_idx"]).fillna(-1).to_numpy().astype(int)
    valid = r >= 0
    hmm = pred["hmm_arc"].to_numpy()
    clf = pred["classifier_arc"].to_numpy()
    n_valid = max(1, int(valid.sum()))
    ref_route = H.collapse(r[valid])
    out = {"observations_scored": int(valid.sum())}
    for name, arcs in (("classifier_only", clf), ("classifier_plus_hmm", hmm)):
        route = H.collapse(arcs)
        pairs, adj = H.continuity(route, graph)
        denom = max(len(route), len(ref_route), 1)
        out[name] = {
            "pointwise_accuracy": float((arcs[valid] == r[valid]).sum() / n_valid),
            "sequence_similarity": 1 - H.levenshtein(route, ref_route) / denom,
            "route_continuity": adj / pairs if pairs else 1.0,
            "route_breaks": pairs - adj,
        }
    return out


@app.get("/")
def root():
    return {"service": "PathSnap API", "docs": "/docs", "trips": "/trips", "predict": "POST /predict"}


@app.get("/health")
def health():
    return {"status": "ok", "trips_loaded": len(STATE.get("trips", {})),
            "model_loaded": "model" in STATE}


@app.get("/trips")
def list_trips():
    return {"trips": [{"trip_id": tid, "gps_points": int(len(v["trip"]["track"]))}
                      for tid, v in sorted(STATE["trips"].items())]}


@app.post("/predict")
def predict(req: PredictRequest):
    entry = STATE["trips"].get(req.trip_id)
    if entry is None:
        log.warning("PREDICT unknown trip_id=%s", req.trip_id)
        raise HTTPException(status_code=404, detail=f"Unknown trip_id {req.trip_id!r}")
    trip = entry["trip"]
    t0 = time.perf_counter()
    log.info("PREDICT start trip_id=%s (%d GPS points)", req.trip_id, len(trip["track"]))
    # Full pipeline, no ground truth: candidates -> classifier probabilities -> HMM/Viterbi.
    pred, route = H.reconstruct_trip(trip, STATE["model"], STATE["params"], top_k=TOP_K)
    metrics = score(pred, entry["ref"], entry["graph"])
    elapsed = time.perf_counter() - t0
    log.info("PREDICT done trip_id=%s hmm_acc=%.4f clf_acc=%.4f continuity=%.3f in %.2fs", req.trip_id,
             metrics["classifier_plus_hmm"]["pointwise_accuracy"],
             metrics["classifier_only"]["pointwise_accuracy"],
             metrics["classifier_plus_hmm"]["route_continuity"], elapsed)
    track = trip["track"]
    ok = np.isfinite(track).all(axis=1)
    gps = [[round(float(la), 6), round(float(lo), 6)] for lo, la in track[ok, :2]]
    return {
        "trip_id": req.trip_id,
        "observations": int(len(pred)),
        "gps_points": gps,                                   # [lat, lon]
        "predicted_route_arc_ids": [int(a) for a in route],
        "predicted_route_coords": route_coords(route, trip),  # [lat, lon] polyline
        "metrics": metrics,
        "processing_seconds": round(elapsed, 3),
    }
