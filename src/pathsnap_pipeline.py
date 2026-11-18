from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FEATURES = [
    "distance_m", "distance_ratio", "segment_length_m",
    "projection_fraction", "heading_difference_rad",
    "heading_cos", "heading_sin", "gps_speed_mps",
    # within-observation (relative) features: these actually carry ranking signal
    "dist_rank", "dist_minus_min_m", "dist_share",
]
ALIGN_WINDOW = 20      # max route arcs the vehicle may advance between consecutive GPS points
ALIGN_STEP_PENALTY_M = 1.0
SAMPLE_SPACING_M = 50.0  # arcs are sampled every ~50 m so long arcs are not missed


def get_dataset_root(data_dir: Path) -> Path:
    """Use an already-extracted dataset folder (the one that contains 'train' and 'student_test')."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    if (data_dir / "train").is_dir():
        return data_dir
    # tolerate one wrapper folder, e.g. data/Student folder/train
    found = sorted(p for p in data_dir.rglob("train") if p.is_dir())
    if not found:
        raise FileNotFoundError(f"'train' folder not found inside: {data_dir}")
    return found[0].parent


def load_trip(trip_dir: Path) -> dict:
    """Read one trip's four whitespace-separated files."""
    stem = trip_dir.name
    track = np.loadtxt(trip_dir / f"{stem}.track", dtype=float, ndmin=2)
    nodes = np.loadtxt(trip_dir / f"{stem}.nodes", dtype=float, ndmin=2)
    arcs = np.loadtxt(trip_dir / f"{stem}.arcs", dtype=int, ndmin=2)
    route_path = trip_dir / f"{stem}.route"
    route = np.loadtxt(route_path, dtype=int, ndmin=1) if route_path.exists() else np.array([], dtype=int)
    return {"trip_id": stem, "track": track, "nodes": nodes, "arcs": arcs, "route": route}


def inspect_dataset(root: Path, output_dir: Path) -> pd.DataFrame:
    rows = []
    for partition in ("train", "student_test"):
        folder = root / partition
        if not folder.exists():
            continue
        for trip_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            d = load_trip(trip_dir)
            t, n, a, r = d["track"], d["nodes"], d["arcs"], d["route"]
            dt = np.diff(t[:, 2]) if len(t) > 1 else np.array([])
            valid_arc = (a[:, 0] >= 0) & (a[:, 0] < len(n)) & (a[:, 1] >= 0) & (a[:, 1] < len(n))
            rows.append({
                "partition": partition, "trip_id": d["trip_id"],
                "gps_observations": len(t), "nodes": len(n), "arcs": len(a),
                "route_arcs": len(r), "track_missing_values": int(np.isnan(t).sum()),
                "timestamp_non_increasing": bool(np.any(dt <= 0)),
                "timestamp_gap_count_gt_1s": int(np.sum(dt > 1.5)),
                "invalid_arc_endpoints": int((~valid_arc).sum()),
                "track_file_bytes": (trip_dir / f"{trip_dir.name}.track").stat().st_size
            })
    inventory = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "dataset_inventory.csv", index=False)
    if not inventory.empty:
        inventory.groupby("partition").agg(
            trips=("trip_id", "count"),
            gps_observations_mean=("gps_observations", "mean"),
            gps_observations_median=("gps_observations", "median"),
            nodes_mean=("nodes", "mean"),
            arcs_mean=("arcs", "mean"),
            route_arcs_mean=("route_arcs", "mean"),
            missing_values=("track_missing_values", "sum"),
            timestamp_gaps=("timestamp_gap_count_gt_1s", "sum")
        ).to_csv(output_dir / "dataset_summary.csv")
    return inventory


def local_xy(lonlat: np.ndarray) -> np.ndarray:
    """Local equirectangular projection in metres, suitable for trip-scale distances."""
    lon0 = np.nanmean(lonlat[:, 0])
    lat0 = np.nanmean(lonlat[:, 1])
    x = (lonlat[:, 0] - lon0) * 111320.0 * np.cos(np.deg2rad(lat0))
    y = (lonlat[:, 1] - lat0) * 110540.0
    return np.column_stack([x, y])


def bearing_xy(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    # Bearing clockwise from north.
    return np.arctan2(vx, vy)


def point_segment_features(points: np.ndarray, starts: np.ndarray, ends: np.ndarray):
    vec = ends - starts
    denom = np.sum(vec * vec, axis=1)
    denom = np.maximum(denom, 1e-9)
    rel = points - starts
    frac = np.clip(np.sum(rel * vec, axis=1) / denom, 0.0, 1.0)
    projected = starts + frac[:, None] * vec
    distance = np.linalg.norm(points - projected, axis=1)
    length = np.sqrt(denom)
    return distance, frac, length, vec


def align_to_route(gps_xy: np.ndarray, route_pos_arcs: np.ndarray, starts: np.ndarray, ends: np.ndarray):
    """
    Monotone alignment of GPS points to the reference route (replaces the old
    nearest-midpoint + np.maximum.accumulate step, which locked onto the wrong
    arc whenever the route starts and ends in the same area).

    Dynamic programme over route positions: observation i may sit at route
    position j, and from i-1 to i the position may stay or advance by at most
    ALIGN_WINDOW arcs. Cost = point-to-SEGMENT distance (m) + small step penalty.
    Returns (route position per observation, distance to the assigned arc).
    """
    n, m = len(gps_xy), len(route_pos_arcs)
    rs, re = starts[route_pos_arcs], ends[route_pos_arcs]
    D = np.empty((n, m))
    for j in range(m):  # m route arcs, vectorised over n points
        d, _, _, _ = point_segment_features(gps_xy, np.repeat(rs[j][None], n, 0), np.repeat(re[j][None], n, 0))
        D[:, j] = d
    cost = D[0].copy()
    back = np.zeros((n, m), dtype=np.int16)
    for i in range(1, n):
        best = np.full(m, np.inf)
        arg = np.zeros(m, dtype=np.int16)
        for s in range(0, min(ALIGN_WINDOW, m - 1) + 1):
            cand = np.full(m, np.inf)
            cand[s:] = cost[:m - s] + ALIGN_STEP_PENALTY_M * s
            better = cand < best
            best[better] = cand[better]
            arg[better] = s
        cost = best + D[i]
        back[i] = arg
    pos = np.zeros(n, dtype=int)
    pos[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = pos[i] - back[i, pos[i]]
    return pos, D[np.arange(n), pos]


def build_candidates_for_trip(trip: dict, partition: str, top_k: int = 15) -> tuple[pd.DataFrame, dict]:
    """
    Candidate arcs per GPS observation via a point-to-SEGMENT query (arcs sampled
    every ~50 m, so long arcs are found), then exact segment distance, top_k kept.

    Labels: candidate arc == arc assigned by the monotone DP alignment to the
    supplied .route. This is still a reference, not verified ground truth.
    Candidate generation never looks at the route (no label leakage).
    """
    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    route = np.asarray(trip["route"], dtype=int).reshape(-1)
    valid_track = np.isfinite(track).all(axis=1)
    original_idx = np.flatnonzero(valid_track)
    track = track[valid_track]
    if len(track) < 2 or len(nodes) == 0 or len(arcs) == 0:
        return pd.DataFrame(), {"observations": len(track), "candidate_misses": len(track), "candidate_miss_rate": 1.0}

    valid_arc = (arcs[:, 0] >= 0) & (arcs[:, 0] < len(nodes)) & (arcs[:, 1] >= 0) & (arcs[:, 1] < len(nodes))
    arc_ids = np.flatnonzero(valid_arc)
    arcs_ok = arcs[valid_arc]
    all_xy = local_xy(np.vstack([track[:, :2], nodes]))
    gps_xy, node_xy = all_xy[:len(track)], all_xy[len(track):]
    starts, ends = node_xy[arcs_ok[:, 0]], node_xy[arcs_ok[:, 1]]
    n = len(track)

    # --- sample points along every arc (spacing <= SAMPLE_SPACING_M) ---
    seg_len_all = np.linalg.norm(ends - starts, axis=1)
    cnt = np.maximum(2, np.ceil(seg_len_all / SAMPLE_SPACING_M).astype(int) + 1)
    sample_arc = np.repeat(np.arange(len(arcs_ok)), cnt)
    offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    frac_s = offs / np.repeat(cnt - 1, cnt)
    sample_xy = starts[sample_arc] + frac_s[:, None] * (ends - starts)[sample_arc]
    tree = cKDTree(sample_xy)
    k_pts = int(min(len(sample_xy), max(40, 3 * top_k)))
    _, nn = tree.query(gps_xy, k=k_pts)
    nn = nn.reshape(n, -1)
    cand = sample_arc[nn]                                   # (n, k_pts) arcs, with duplicates

    # --- exact point-to-segment distance, drop duplicate arcs per observation ---
    flat_pts = np.repeat(gps_xy, cand.shape[1], axis=0)
    d, fr, sl, vec = point_segment_features(flat_pts, starts[cand.ravel()], ends[cand.ravel()])
    d, fr, sl = d.reshape(cand.shape), fr.reshape(cand.shape), sl.reshape(cand.shape)
    vec = vec.reshape(cand.shape + (2,))
    order = np.argsort(cand, axis=1, kind="stable")
    cs = np.take_along_axis(cand, order, 1)
    dup = np.zeros_like(cs, dtype=bool)
    dup[:, 1:] = cs[:, 1:] == cs[:, :-1]
    dd = np.take_along_axis(d, order, 1)
    dd[dup] = np.inf
    keep = np.argsort(dd, axis=1, kind="stable")[:, :min(top_k, cand.shape[1])]
    sel = np.take_along_axis(order, keep, 1)                # indices into the k_pts axis
    cand_local = np.take_along_axis(cand, sel, 1)
    dist = np.take_along_axis(d, sel, 1)
    frac = np.take_along_axis(fr, sel, 1)
    seg_len = np.take_along_axis(sl, sel, 1)
    vec_sel = np.take_along_axis(vec, sel[:, :, None], 1)
    valid = np.isfinite(dist)

    # --- smoothed GPS motion (5-point window) ---
    lo, hi = np.maximum(np.arange(n) - 2, 0), np.minimum(np.arange(n) + 2, n - 1)
    mv = gps_xy[hi] - gps_xy[lo]
    dt = np.maximum(np.abs(track[hi, 2] - track[lo, 2]), 1e-3)
    gps_speed = np.linalg.norm(mv, axis=1) / dt
    gps_heading = bearing_xy(mv[:, 0], mv[:, 1])
    arc_bearing = bearing_xy(vec_sel[..., 0], vec_sel[..., 1])
    diff = np.arctan2(np.sin(gps_heading[:, None] - arc_bearing), np.cos(gps_heading[:, None] - arc_bearing))

    # --- within-observation relative features ---
    dist_f = np.where(valid, dist, np.nan)
    dmin = np.nanmin(dist_f, axis=1, keepdims=True)
    inv = np.where(valid, 1.0 / (dist + 1.0), 0.0)
    share = inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)
    rank = np.broadcast_to(np.arange(dist.shape[1]), dist.shape)

    # --- reference labels via monotone DP alignment ---
    route_valid = route[(route >= 0) & (route < len(arcs))]
    route_valid = route_valid[np.isin(route_valid, arc_ids)]
    aligned_arc = np.full(n, -1, dtype=int)
    ref_dist = np.full(n, np.nan)
    if len(route_valid):
        route_pos_arcs = np.searchsorted(arc_ids, route_valid)
        pos, ref_dist = align_to_route(gps_xy, route_pos_arcs, starts, ends)
        aligned_arc = route_valid[pos]

    actual_arc = arc_ids[cand_local]
    obs_rep = np.broadcast_to(np.arange(n)[:, None], dist.shape)
    label = (actual_arc == aligned_arc[:, None]) & (aligned_arc[:, None] >= 0)
    m = valid
    frame = pd.DataFrame({
        "partition": partition, "trip_id": trip["trip_id"],
        "obs_idx": original_idx[obs_rep[m]], "candidate_rank": rank[m],
        "arc_id": actual_arc[m],
        "distance_m": dist[m], "distance_ratio": dist[m] / np.maximum(seg_len[m], 1.0),
        "segment_length_m": seg_len[m], "projection_fraction": frac[m],
        "heading_difference_rad": np.abs(diff[m]), "heading_cos": np.cos(diff[m]), "heading_sin": np.sin(diff[m]),
        "gps_speed_mps": gps_speed[obs_rep[m]],
        "dist_rank": rank[m], "dist_minus_min_m": (dist - dmin)[m], "dist_share": share[m],
        "label": label[m].astype(int), "reference_arc_id": aligned_arc[obs_rep[m]],
    })
    miss_obs = int((label.sum(axis=1) == 0).sum())
    diagnostics = {
        "observations": n, "candidate_misses": miss_obs, "candidate_miss_rate": miss_obs / max(1, n),
        "route_arcs_valid": int(len(route_valid)), "candidate_rows": int(len(frame)),
        "median_dist_to_reference_arc_m": float(np.nanmedian(ref_dist)) if len(route_valid) else float("nan"),
        "p95_dist_to_reference_arc_m": float(np.nanpercentile(ref_dist, 95)) if len(route_valid) else float("nan"),
    }
    return frame, diagnostics


def build_partition(root: Path, partition: str, output_dir: Path, top_k: int = 15):
    frames, diagnostics = [], []
    for trip_dir in sorted(p for p in (root / partition).iterdir() if p.is_dir()):
        trip = load_trip(trip_dir)
        frame, diag = build_candidates_for_trip(trip, partition, top_k=top_k)
        if not frame.empty:
            frames.append(frame)
        diagnostics.append({"trip_id": trip["trip_id"], **diag})
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    diag_df = pd.DataFrame(diagnostics)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_dir / f"{partition}_candidates.parquet", index=False)
    diag_df.to_csv(output_dir / f"{partition}_candidate_diagnostics.csv", index=False)
    return data, diag_df


def make_model_pipelines():
    # No class_weight anywhere: re-weighting distorts probabilities, and the HMM needs calibrated emissions.
    def pipe(estimator, scale=False):
        return Pipeline([("scale", StandardScaler() if scale else "passthrough"), ("model", estimator)])
    return {
        "LogisticRegression": pipe(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        "RandomForest": pipe(RandomForestClassifier(n_estimators=150, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE)),
        "HistGradientBoosting": pipe(HistGradientBoostingClassifier(max_iter=180, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, random_state=RANDOM_STATE)),
        "KNN": pipe(KNeighborsClassifier(n_neighbors=50, weights="uniform", n_jobs=-1), True),
        "MLP": pipe(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=200,
                                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=RANDOM_STATE), True),
    }


def observation_top1(frame: pd.DataFrame, predicted_score: np.ndarray, hits_only: bool = False) -> float:
    temp = frame[["trip_id", "obs_idx", "label"]].copy()
    temp["score"] = np.asarray(predicted_score)
    if hits_only:  # only observations whose reference arc is in the candidate set
        temp = temp[temp.groupby(["trip_id", "obs_idx"])["label"].transform("max") == 1]
    winners = temp.loc[temp.groupby(["trip_id", "obs_idx"])["score"].idxmax()]
    return float(winners["label"].mean()) if len(winners) else float("nan")


def make_top1_scorer(ref_df: pd.DataFrame):
    """Group-aware scorer for BayesSearchCV: observation-level Top-1 on the validation fold."""
    def scorer(est, X, y):
        prob = est.predict_proba(X)[:, 1]
        return observation_top1(ref_df.loc[X.index, ["trip_id", "obs_idx", "label"]], prob)
    return scorer


def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return 1 / (1 + np.exp(-np.clip(raw, -30, 30)))


def evaluate_frame(model, frame: pd.DataFrame) -> dict:
    X = frame[FEATURES]
    y = frame["label"].astype(int).to_numpy()
    t0 = time.perf_counter()
    pred = model.predict(X)
    prob = np.clip(_proba(model, X), 1e-15, 1 - 1e-15)
    infer_s = time.perf_counter() - t0
    n_obs = max(1, frame[["trip_id", "obs_idx"]].drop_duplicates().shape[0])
    has_pos = frame.groupby(["trip_id", "obs_idx"])["label"].transform("max") == 1
    return {
        "candidate_accuracy": accuracy_score(y, pred),
        "candidate_precision": precision_score(y, pred, zero_division=0),
        "candidate_recall": recall_score(y, pred, zero_division=0),
        "candidate_macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "candidate_log_loss": log_loss(y, np.column_stack([1 - prob, prob]), labels=[0, 1]),
        "observation_top1_accuracy": observation_top1(frame, prob),                 # overall (misses count as wrong)
        "observation_top1_hits_only": observation_top1(frame, prob, hits_only=True),  # only observations with reference in candidates
        "candidate_miss_rate": float(1 - has_pos.groupby([frame["trip_id"], frame["obs_idx"]]).first().mean()),
        "inference_seconds": infer_s,
        "inference_ms_per_observation": 1000 * infer_s / n_obs,
        "_confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def train_and_compare(train_df: pd.DataFrame, student_df: pd.DataFrame, out: Path, thin: int = 4):
    # Train only on observations whose reference arc is in the candidate set; thin 1 Hz data
    # (adjacent points are near-duplicates) to keep CV/tuning tractable. Student set is NOT thinned.
    train_df = train_df.copy()
    has_pos = train_df.groupby(["trip_id", "obs_idx"])["label"].transform("max").astype(bool)
    fit_df = train_df.loc[has_pos & (train_df["obs_idx"] % max(1, thin) == 0)].reset_index(drop=True)
    X, y = fit_df[FEATURES], fit_df["label"].astype(int)
    groups = fit_df["trip_id"].to_numpy()
    if len(np.unique(groups)) < 5:
        raise ValueError("Need at least five distinct training trips for GroupKFold.")
    cv = GroupKFold(n_splits=5)
    results, log, confusions = [], {}, {}
    models = make_model_pipelines()

    for name, estimator in models.items():
        print(f"  training {name} ...", flush=True)
        fold_scores = []
        for fold, (tr, va) in enumerate(cv.split(X, y, groups), start=1):
            m = clone(estimator)
            start = time.perf_counter()
            m.fit(X.iloc[tr], y.iloc[tr])
            train_seconds = time.perf_counter() - start
            vp = np.clip(_proba(m, X.iloc[va]), 1e-15, 1 - 1e-15)
            vpred = m.predict(X.iloc[va])
            fold_scores.append({
                "fold": fold,
                "macro_f1": f1_score(y.iloc[va], vpred, average="macro", zero_division=0),
                "candidate_accuracy": accuracy_score(y.iloc[va], vpred),
                "log_loss": log_loss(y.iloc[va], np.column_stack([1 - vp, vp]), labels=[0, 1]),
                "observation_top1_accuracy": observation_top1(fit_df.iloc[va], vp),
                "train_seconds": train_seconds,
            })
        fs = pd.DataFrame(fold_scores)
        start = time.perf_counter()
        estimator.fit(X, y)
        fit_seconds = time.perf_counter() - start
        sm = evaluate_frame(estimator, student_df)
        confusions[name] = sm.pop("_confusion")
        results.append({
            "model": name,
            "cv_observation_top1": fs["observation_top1_accuracy"].mean(),
            "cv_observation_top1_std": fs["observation_top1_accuracy"].std(),
            "cv_log_loss": fs["log_loss"].mean(),
            "cv_macro_f1": fs["macro_f1"].mean(), "cv_macro_f1_std": fs["macro_f1"].std(),
            "cv_candidate_accuracy": fs["candidate_accuracy"].mean(),
            "fit_seconds": fit_seconds,
            **{f"student_{k}": v for k, v in sm.items()},
        })
        log[name] = {"params": {k: str(v) for k, v in estimator.get_params().items() if "__" in k}, "fit_seconds": fit_seconds,
                     "mean_cv_fold_train_seconds": fs["train_seconds"].mean()}
        joblib.dump(estimator, out / f"{name}.joblib")
        fs.to_csv(out / f"{name}_cv_folds.csv", index=False)
        if name == "MLP":
            mlp_loss_curves(fit_df, out)

    # Primary selection metric: CV observation-level Top-1 (matches the map-matching goal); log loss breaks ties.
    results_df = pd.DataFrame(results).sort_values(["cv_observation_top1", "cv_log_loss"], ascending=[False, True])
    results_df.to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame([{"model": k, "tn": v[0][0], "fp": v[0][1], "fn": v[1][0], "tp": v[1][1]} for k, v in confusions.items()]
                 ).to_csv(out / "student_confusion_matrices.csv", index=False)
    (out / "experiment_log.json").write_text(json.dumps(
        {"random_state": RANDOM_STATE, "features": FEATURES, "thin": thin, "train_rows_used": int(len(fit_df)),
         "primary_metric": "cv_observation_top1", "models": log}, indent=2))
    best_name = results_df.iloc[0]["model"]
    return best_name, results_df, models[best_name], fit_df


def mlp_loss_curves(fit_df: pd.DataFrame, out: Path, epochs: int = 40):
    """MLP training vs validation log loss per epoch (validation = held-out trips)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, va = next(gss.split(fit_df, groups=fit_df["trip_id"]))
    y = fit_df["label"].astype(int).to_numpy()
    scaler = StandardScaler().fit(fit_df.iloc[tr][FEATURES])  # fit on training trips only
    X_tr, X_va = scaler.transform(fit_df.iloc[tr][FEATURES]), scaler.transform(fit_df.iloc[va][FEATURES])
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, random_state=RANDOM_STATE)
    rows = []
    for ep in range(1, epochs + 1):
        mlp.partial_fit(X_tr, y[tr], classes=[0, 1])
        rows.append({"epoch": ep,
                     "train_loss": log_loss(y[tr], mlp.predict_proba(X_tr)[:, 1], labels=[0, 1]),
                     "val_loss": log_loss(y[va], mlp.predict_proba(X_va)[:, 1], labels=[0, 1])})
    pd.DataFrame(rows).to_csv(out / "MLP_loss_curve.csv", index=False)


def _bar(ax, names, values, title, ylabel, errors=None):
    bars = ax.bar(names, values, yerr=errors, capsize=4, color="#4C72B0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8)


def make_plots(out: Path):
    """Save the model-comparison figures (PNG) under results/plots/."""
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    df = pd.read_csv(out / "model_comparison.csv")
    names = df["model"].tolist()

    panels = [
        ("cv_macro_f1", "Cross-validation Macro F1 (5-fold GroupKFold)", "Macro F1", "cv_macro_f1_std", "cv_macro_f1.png"),
        ("cv_observation_top1", "Cross-validation Top-1 Accuracy (5-fold GroupKFold)", "Top-1 accuracy", "cv_observation_top1_std", "cv_top1.png"),
        ("student_observation_top1_accuracy", "Student-testing Top-1 Accuracy", "Top-1 accuracy", None, "student_top1.png"),
        ("student_candidate_macro_f1", "Student-testing Macro F1", "Macro F1", None, "student_macro_f1.png"),
    ]
    for col, title, ylabel, err, fname in panels:
        fig, ax = plt.subplots(figsize=(7, 4))
        _bar(ax, names, df[col].to_numpy(), title, ylabel, df[err].to_numpy() if err else None)
        fig.tight_layout(); fig.savefig(plots / fname, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, df["cv_log_loss"], 0.4, label="Cross-validation", color="#4C72B0")
    ax.bar(x + 0.2, df["student_candidate_log_loss"], 0.4, label="Student testing", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25)
    ax.set_ylabel("Log loss (lower is better)"); ax.set_title("Log Loss comparison"); ax.legend()
    fig.tight_layout(); fig.savefig(plots / "log_loss.png", dpi=150); plt.close(fig)

    curve = out / "MLP_loss_curve.csv"
    if curve.exists():
        c = pd.read_csv(curve)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(c["epoch"], c["train_loss"], label="Training loss")
        ax.plot(c["epoch"], c["val_loss"], label="Validation loss (held-out trips)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Log loss"); ax.set_title("MLP training and validation loss"); ax.legend()
        fig.tight_layout(); fig.savefig(plots / "mlp_loss_curve.png", dpi=150); plt.close(fig)

    cm_path = out / "student_confusion_matrices.csv"
    if cm_path.exists():
        cm = pd.read_csv(cm_path)
        fig, axes = plt.subplots(1, len(cm), figsize=(3.2 * len(cm), 3.4))
        for ax, (_, r) in zip(np.atleast_1d(axes), cm.iterrows()):
            m = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
            ax.imshow(m, cmap="Blues")
            ax.set_title(r["model"], fontsize=9)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
            for (i, j), v in np.ndenumerate(m):
                ax.text(j, i, f"{int(v):,}", ha="center", va="center", fontsize=8, color="white" if v > m.max() / 2 else "black")
        fig.suptitle("Student-testing confusion matrices (candidate level)")
        fig.tight_layout(); fig.savefig(plots / "confusion_matrices.png", dpi=150); plt.close(fig)

    tuned = out / "tuned_vs_untuned_student_test.csv"
    if tuned.exists():
        t = pd.read_csv(tuned)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        _bar(axes[0], t["version"].tolist(), t["observation_top1_accuracy"].to_numpy(), "Top-1 accuracy", "Student testing")
        _bar(axes[1], t["version"].tolist(), t["candidate_macro_f1"].to_numpy(), "Macro F1", "Student testing")
        fig.suptitle("Untuned vs Bayesian-tuned")
        fig.tight_layout(); fig.savefig(plots / "tuned_vs_untuned.png", dpi=150); plt.close(fig)
    print(f"Plots saved under: {plots.resolve()}")


def bayesian_tune(best_name: str, base_pipeline, fit_df: pd.DataFrame, student_df: pd.DataFrame, out: Path, n_iter: int = 10, tune_frac: float = 0.5, tune_folds: int = 3):
    try:
        from skopt import BayesSearchCV
        from skopt.space import Integer, Real, Categorical
    except ImportError as exc:
        raise ImportError("Install scikit-optimize: pip install scikit-optimize") from exc

    spaces = {
        "RandomForest": {
            "model__n_estimators": Integer(60, 200), "model__max_depth": Integer(6, 20),
            "model__min_samples_leaf": Integer(2, 20), "model__max_features": Categorical(["sqrt", "log2", 0.7]),
        },
        "LogisticRegression": {
            "model__C": Real(1e-3, 100, prior="log-uniform"), "model__solver": Categorical(["lbfgs", "liblinear"]),
        },
        "KNN": {
            "model__n_neighbors": Integer(5, 100), "model__weights": Categorical(["uniform", "distance"]),
            "model__p": Integer(1, 2),
        },
        "MLP": {
            "model__hidden_layer_sizes": Categorical([(32,), (64,), (64, 32), (128, 64)]),
            "model__alpha": Real(1e-6, 1e-1, prior="log-uniform"),
            "model__learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform"),
        },
        "HistGradientBoosting": {
            "model__learning_rate": Real(0.02, 0.2, prior="log-uniform"), "model__max_iter": Integer(80, 350),
            "model__max_leaf_nodes": Integer(7, 31), "model__l2_regularization": Real(1e-4, 10, prior="log-uniform"),
        },
    }
    full_df = fit_df.reset_index(drop=True)
    # The search runs on a subset of observations, sampled within each trip (all candidates of an observation kept together)
    # so every trial finishes quickly; the best parameters are then refit on the full training data below.
    if tune_frac < 1.0:
        # Sample WITHIN each trip so all training trips stay represented in the search data.
        obs = full_df[["trip_id", "obs_idx"]].drop_duplicates().sample(frac=1.0, random_state=RANDOM_STATE)
        rank = obs.groupby("trip_id").cumcount()
        size = obs.groupby("trip_id")["obs_idx"].transform("size")
        keep = obs[rank < np.maximum(1, np.ceil(size * tune_frac))]
        fit_df = full_df.merge(keep, on=["trip_id", "obs_idx"]).reset_index(drop=True)
        print(f"Search sample keeps {keep['trip_id'].nunique()} of {full_df['trip_id'].nunique()} training trips")
    else:
        fit_df = full_df
    print(f"Bayesian search: {len(fit_df)} rows, {n_iter} trials x {tune_folds} folds")
    X, y = fit_df[FEATURES], fit_df["label"].astype(int)
    groups = fit_df["trip_id"].to_numpy()
    search = BayesSearchCV(
        estimator=clone(base_pipeline), search_spaces=spaces[best_name], n_iter=n_iter,
        scoring=make_top1_scorer(fit_df),          # objective: observation-level Top-1, group-aware CV
        cv=GroupKFold(n_splits=tune_folds),
        n_jobs=1,                                   # parallelism is inside the models; avoids CPU oversubscription
        random_state=RANDOM_STATE, refit=False, return_train_score=False, verbose=2,
    )
    start = time.perf_counter()
    search.fit(X, y, groups=groups)
    elapsed = time.perf_counter() - start
    pd.DataFrame(search.cv_results_).to_csv(out / "bayesian_trials.csv", index=False)
    tuned = clone(base_pipeline).set_params(**search.best_params_)
    tuned.fit(full_df[FEATURES], full_df["label"].astype(int))
    tuned_m, untuned_m = evaluate_frame(tuned, student_df), evaluate_frame(joblib.load(out / f"{best_name}.joblib"), student_df)
    tuned_m.pop("_confusion"), untuned_m.pop("_confusion")
    comparison = pd.DataFrame([{"version": "untuned", **untuned_m}, {"version": "bayesian_tuned", **tuned_m}])
    comparison.to_csv(out / "tuned_vs_untuned_student_test.csv", index=False)
    joblib.dump(tuned, out / "best_tuned_model.joblib")
    summary = {
        "selected_model": best_name, "objective": "observation-level Top-1 (GroupKFold by trip)",
        "best_params": search.best_params_, "best_cv_observation_top1": float(search.best_score_),
        "bayesian_trials": int(n_iter), "optimisation_seconds": elapsed,
        "student_test_comparison": comparison.to_dict(orient="records"),
    }
    (out / "bayesian_best_config.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Path to the extracted dataset folder (contains 'train' and 'student_test')")
    parser.add_argument("--workdir", default="data_work")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--thin", type=int, default=4, help="keep every Nth GPS point for training/CV/tuning only")
    parser.add_argument("--bayes-iter", type=int, default=10)
    parser.add_argument("--tune-frac", type=float, default=0.5, help="fraction of training observations used inside the Bayesian search")
    args = parser.parse_args()

    work = Path(args.workdir)
    out = work / "results"
    out.mkdir(parents=True, exist_ok=True)
    root = get_dataset_root(Path(args.data_dir))
    print(f"Dataset root: {root}")
    inventory = inspect_dataset(root, out)
    print(inventory.groupby("partition").agg(trips=("trip_id", "nunique"), gps_points=("gps_observations", "sum")))

    print("\nBuilding training candidates...")
    train_df, train_diag = build_partition(root, "train", work / "processed", args.top_k)
    print("Train rows:", len(train_df), "| mean miss rate:", round(train_diag["candidate_miss_rate"].mean(), 4),
          "| median dist to reference arc (m):", round(train_diag["median_dist_to_reference_arc_m"].median(), 2),
          "| trips with miss>50%:", int((train_diag["candidate_miss_rate"] > 0.5).sum()))
    print("\nBuilding student-testing candidates...")
    student_df, student_diag = build_partition(root, "student_test", work / "processed", args.top_k)
    print("Student rows:", len(student_df), "| mean miss rate:", round(student_diag["candidate_miss_rate"].mean(), 4))

    (out / "feature_list.json").write_text(json.dumps(FEATURES, indent=2))
    if train_df.empty or student_df.empty:
        raise RuntimeError("Candidate generation returned no data. Inspect input paths and data files.")

    best_name, comparison, best_pipeline, fit_df = train_and_compare(train_df, student_df, out, args.thin)
    print("\nFive-model comparison (LogReg, RandomForest, HistGB, KNN, MLP):")
    print(comparison.to_string(index=False))
    print("\nSelected model by CV observation-level Top-1:", best_name)
    make_plots(out)

    summary = bayesian_tune(best_name, best_pipeline, fit_df, student_df, out, args.bayes_iter, args.tune_frac)
    (out / "final_model_config.json").write_text(json.dumps({
        "model_file": "best_tuned_model.joblib", "features": FEATURES, "top_k": args.top_k,
        "sample_spacing_m": SAMPLE_SPACING_M, "random_state": RANDOM_STATE, "best_params": summary["best_params"]}, indent=2, default=str))
    print("\nBayesian optimisation complete:")
    print(json.dumps(summary, indent=2, default=str))
    make_plots(out)  # redraw so the tuned-vs-untuned figure is included
    print(f"\nAll outputs saved under: {out.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()