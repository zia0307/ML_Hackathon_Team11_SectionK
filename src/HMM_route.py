"""
PathSnap Part 3: HMM + Viterbi route reconstruction.

DESIGN (use this text in your report)
-------------------------------------
Hidden state     : at GPS observation t, the state is one of the K candidate road arcs generated for that
                   observation (top-K nearest arcs). State j at time t keeps its arc_id, so the
                   candidate -> road-segment correspondence is preserved end to end.
Emission         : the trained classifier's probability p(arc is the reference arc | features).
                   log-emission = emission_power * log(p). emission_power < 1 trusts the classifier less,
                   > 1 trusts it more, relative to the transition model.
Transition       : based on road-network connectivity between the arcs chosen at t-1 and t.
                   hops = 0 (same arc) -> 0 cost
                   hops = h (arc b reachable from arc a in h directed steps, h <= limit) -> -hop_penalty * h
                   otherwise (not reachable) -> -jump_penalty
                   Costs are capped at jump_penalty, so an invalid transition is expensive but never
                   impossible: the decoder always finds a path, even through bad or missing candidates.
Viterbi          : delta_t(j) = emission_t(j) + max_i [ delta_{t-1}(i) + trans(i -> j) ],
                   with back-pointers, then backtracking from argmax delta_T.
GPS gaps         : the time gap between consecutive observations (from the .track timestamps) relaxes the
                   hop limit (+1 hop per 2 s of extra gap, at most +4). A gap longer than gap_reset_s
                   starts a new independent segment (no transition across the gap).
Missing / invalid: observations with no candidates (invalid GPS rows are dropped upstream) are skipped,
                   so the gap logic above bridges them. Unreachable pairs get jump_penalty.
Parameters       : hop_penalty, jump_penalty, emission_power, max_hops are tuned by grid search on
                   student_test and/or training trips ONLY. The evaluator set is never used for tuning.

USAGE (run from the `code/` folder)
-----------------------------------
1) Tune the decoder and compare classifier-only vs classifier+HMM on student_test:
    py src/hmm_route.py --model data_work/results/best_tuned_model.joblib ^
        --processed data_work/processed --data-dir data --out data_work/hmm_results

2) FINAL evaluator run (only once, after the decoder parameters are fixed):
    py src/hmm_route.py --model data_work/results/best_tuned_model.joblib ^
        --out data_work/hmm_results --final-dir evaluator_test

For the FastAPI backend, import reconstruct_trip() from this file.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import pathsnap_pipeline as P  # noqa: E402

HOP_CAP = 8        # BFS depth stored per arc (max_hops + max gap relaxation must stay <= this)
UNREACH = 127      # int8 sentinel for "not reachable within HOP_CAP"
MAX_GAP_EXTRA = 4  # most extra hops a GPS gap can add
MAX_HOPS_LIMIT = HOP_CAP - MAX_GAP_EXTRA


@dataclass
class DecoderParams:
    hop_penalty: float = 1.0
    jump_penalty: float = 8.0
    emission_power: float = 1.0
    max_hops: int = 3
    gap_reset_s: float = 30.0

    def __post_init__(self):
        if self.max_hops > MAX_HOPS_LIMIT:
            raise ValueError(f"max_hops must be <= {MAX_HOPS_LIMIT}")


# ----------------------------------------------------------------------------- road network
class TripGraph:
    """Directed arc graph: arc a -> arc b if a's end node is b's start node."""

    def __init__(self, arcs: np.ndarray, n_nodes: int):
        arcs = np.asarray(arcs, dtype=int).reshape(-1, 2)
        self.src, self.dst = arcs[:, 0], arcs[:, 1]
        self.ok = (self.src >= 0) & (self.src < n_nodes) & (self.dst >= 0) & (self.dst < n_nodes)
        out = {}
        for a in np.flatnonzero(self.ok):
            out.setdefault(int(self.src[a]), []).append(int(a))
        self._out = out
        self._reach = {}
        self._succ_set = {}

    def successors(self, a: int):
        if a < 0 or a >= len(self.dst) or not self.ok[a]:
            return []
        return self._out.get(int(self.dst[a]), [])

    def successor_set(self, a: int):
        s = self._succ_set.get(a)
        if s is None:
            s = set(self.successors(a))
            self._succ_set[a] = s
        return s

    def adjacent(self, a: int, b: int) -> bool:
        return a != b and b in self.successor_set(a)

    def reach(self, a: int) -> dict:
        """{arc: hops} for arcs reachable from a in 1..HOP_CAP directed steps (cached)."""
        r = self._reach.get(a)
        if r is None:
            r, seen, frontier = {}, {a}, [a]
            for h in range(1, HOP_CAP + 1):
                nxt = []
                for u in frontier:
                    for v in self.successors(u):
                        if v not in seen:
                            seen.add(v)
                            r[v] = h
                            nxt.append(v)
                frontier = nxt
                if not frontier:
                    break
            self._reach[a] = r
        return r


def hop_matrix(graph: TripGraph, prev_arcs: np.ndarray, cur_arcs: np.ndarray) -> np.ndarray:
    H = np.full((len(prev_arcs), len(cur_arcs)), UNREACH, dtype=np.int8)
    for i, a in enumerate(prev_arcs):
        reach = graph.reach(int(a))
        for j, b in enumerate(cur_arcs):
            if a == b:
                H[i, j] = 0
            else:
                h = reach.get(int(b))
                if h is not None:
                    H[i, j] = h
    return H


# ----------------------------------------------------------------------------- per-trip sequence
@dataclass
class TripSeq:
    trip_id: str
    obs_idx: np.ndarray        # (n,) original track row of each observation that has candidates
    arcs: list                 # n arrays of candidate arc ids
    logp: list                 # n arrays of log classifier probabilities (same order as arcs)
    hops: list                 # hops[i] = hop matrix (K_{i-1} x K_i); hops[0] is None
    gap_s: np.ndarray          # seconds since the previous observation (0 for the first)
    clf_arc: np.ndarray        # classifier-only prediction (argmax probability)
    ref_arc: np.ndarray        # reference arc per observation (-1 if unknown)
    hit: np.ndarray            # 1 if the reference arc is among the candidates


def build_trip_seq(trip_id: str, frame: pd.DataFrame, timestamps: np.ndarray,
                   graph: TripGraph, prob: np.ndarray) -> TripSeq:
    f = frame[["obs_idx", "candidate_rank", "arc_id", "reference_arc_id", "label"]].copy()
    f["prob"] = np.asarray(prob)
    f = f.sort_values(["obs_idx", "candidate_rank"], kind="stable")
    obs = f["obs_idx"].to_numpy()
    arcs = f["arc_id"].to_numpy()
    pr = np.clip(f["prob"].to_numpy(), 1e-9, 1.0)
    ref = f["reference_arc_id"].to_numpy()
    lab = f["label"].to_numpy()
    uniq, start = np.unique(obs, return_index=True)
    end = np.r_[start[1:], len(obs)]

    arcs_l, logp_l, clf, refarc, hit = [], [], [], [], []
    for s, e in zip(start, end):
        arcs_l.append(arcs[s:e])
        logp_l.append(np.log(pr[s:e]))
        clf.append(arcs[s + int(np.argmax(pr[s:e]))])
        refarc.append(ref[s])
        hit.append(int(lab[s:e].max()))
    ts = np.asarray(timestamps, dtype=float)[uniq]
    gap = np.r_[0.0, np.diff(ts)]
    hops = [None] + [hop_matrix(graph, arcs_l[i - 1], arcs_l[i]) for i in range(1, len(arcs_l))]
    return TripSeq(trip_id, uniq, arcs_l, logp_l, hops, gap,
                   np.array(clf), np.array(refarc), np.array(hit))


# ----------------------------------------------------------------------------- Viterbi
def viterbi(seq: TripSeq, p: DecoderParams):
    """Return (chosen arc id per observation, chosen candidate index per observation)."""
    n = len(seq.arcs)
    back = [None] * n
    seg_end_choice = {}
    score = p.emission_power * seq.logp[0]
    for i in range(1, n):
        emit = p.emission_power * seq.logp[i]
        if seq.gap_s[i] > p.gap_reset_s:           # long gap: start a new independent segment
            seg_end_choice[i - 1] = int(np.argmax(score))
            score = emit
            continue
        extra = min(MAX_GAP_EXTRA, int(max(0.0, seq.gap_s[i] - 1.0) // 2))
        limit = p.max_hops + extra
        h = seq.hops[i]
        trans = np.where(h <= limit,
                         -np.minimum(p.hop_penalty * h.astype(np.float64), p.jump_penalty),
                         -p.jump_penalty)
        cand = score[:, None] + trans              # (K_prev, K_cur)
        bp = np.argmax(cand, axis=0)
        score = cand[bp, np.arange(cand.shape[1])] + emit
        back[i] = bp
    path = np.zeros(n, dtype=int)
    idx = int(np.argmax(score))
    for i in range(n - 1, -1, -1):
        path[i] = idx
        if back[i] is None:
            if i > 0:
                idx = seg_end_choice[i - 1]
        else:
            idx = int(back[i][idx])
    return np.array([seq.arcs[i][path[i]] for i in range(n)]), path


# ----------------------------------------------------------------------------- metrics
def collapse(arcs) -> list:
    out = []
    for a in arcs:
        a = int(a)
        if not out or out[-1] != a:
            out.append(a)
    return out


def levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def continuity(route: list, graph: TripGraph):
    """(# transitions, # directly connected transitions) along a collapsed arc sequence."""
    pairs = max(0, len(route) - 1)
    adj = sum(1 for a, b in zip(route[:-1], route[1:]) if graph.adjacent(a, b))
    return pairs, adj


def evaluate_trip(seq: TripSeq, graph: TripGraph, hmm_arc: np.ndarray) -> list:
    ref = seq.ref_arc
    valid = ref >= 0
    ref_route = collapse(ref[valid])
    ref_pairs, ref_adj = continuity(ref_route, graph)
    rows = []
    for method, arcs in (("classifier_only", seq.clf_arc), ("classifier_plus_hmm", hmm_arc)):
        correct = int((arcs[valid] == ref[valid]).sum()) if valid.any() else 0
        route = collapse(arcs)
        pairs, adj = continuity(route, graph)
        denom = max(len(route), len(ref_route), 1)
        ps, rs = set(route), set(ref_route)
        tp = len(ps & rs)
        prec, rec = tp / max(1, len(ps)), tp / max(1, len(rs))
        rows.append({
            "trip_id": seq.trip_id, "method": method,
            "observations": int(len(arcs)), "valid_obs": int(valid.sum()), "correct": correct,
            "pointwise_top1": correct / max(1, int(valid.sum())),
            "candidate_miss_rate": float(1 - seq.hit.mean()),
            "route_arcs_pred": len(route), "route_arcs_ref": len(ref_route),
            "seq_similarity": 1 - levenshtein(route, ref_route) / denom,
            "arc_precision": prec, "arc_recall": rec,
            "arc_f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec),
            "transitions": pairs, "adjacent_transitions": adj,
            "route_breaks": pairs - adj,
            "continuity_adjacent": adj / pairs if pairs else 1.0,
            "reference_continuity": ref_adj / ref_pairs if ref_pairs else 1.0,
        })
    return rows


def summarise(per_trip: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in per_trip.groupby("method", sort=False):
        rows.append({
            "method": method, "trips": g["trip_id"].nunique(),
            "observations": int(g["observations"].sum()),
            "pointwise_top1": g["correct"].sum() / max(1, g["valid_obs"].sum()),
            "mean_trip_pointwise_top1": g["pointwise_top1"].mean(),
            "mean_seq_similarity": g["seq_similarity"].mean(),
            "mean_arc_f1": g["arc_f1"].mean(),
            "continuity_adjacent": g["adjacent_transitions"].sum() / max(1, g["transitions"].sum()),
            "route_breaks": int(g["route_breaks"].sum()),
            "route_breaks_per_1000_obs": 1000 * g["route_breaks"].sum() / max(1, g["observations"].sum()),
            "candidate_miss_rate": g["candidate_miss_rate"].mean(),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- loading
def seqs_from_candidate_frame(df: pd.DataFrame, partition_dir: Path, model, trips=None):
    """Build TripSeq objects from a saved candidates parquet (train or student_test)."""
    if trips is not None:
        df = df[df["trip_id"].isin(trips)]
    prob = model.predict_proba(df[P.FEATURES])[:, 1]
    df = df.assign(_prob=prob)
    seqs, graphs = [], {}
    for tid, g in df.groupby("trip_id", sort=True):
        trip = P.load_trip(partition_dir / str(tid))
        graph = TripGraph(trip["arcs"], len(trip["nodes"]))
        graphs[str(tid)] = graph
        seqs.append(build_trip_seq(str(tid), g, trip["track"][:, 2], graph, g["_prob"].to_numpy()))
    return seqs, graphs


def find_trip_dirs(root: Path):
    for base in (root, *[p for p in root.iterdir() if p.is_dir()]):
        trips = sorted(p for p in base.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())
        if trips:
            return trips
    raise FileNotFoundError(f"No trip folders (<id>/<id>.track) found under {root}")


def seqs_from_raw_dir(root: Path, model, top_k: int):
    seqs, graphs = [], {}
    for trip_dir in find_trip_dirs(root):
        trip = P.load_trip(trip_dir)
        frame, _ = P.build_candidates_for_trip(trip, "final", top_k=top_k)
        if frame.empty:
            continue
        prob = model.predict_proba(frame[P.FEATURES])[:, 1]
        graph = TripGraph(trip["arcs"], len(trip["nodes"]))
        graphs[trip["trip_id"]] = graph
        seqs.append(build_trip_seq(trip["trip_id"], frame, trip["track"][:, 2], graph, prob))
    return seqs, graphs


# ----------------------------------------------------------------------------- tuning / evaluation
def decode_all(seqs, params: DecoderParams) -> dict:
    return {s.trip_id: viterbi(s, params)[0] for s in seqs}


def quick_score(seqs, graphs, preds):
    correct = valid = breaks = obs = 0
    for s in seqs:
        arcs = preds[s.trip_id]
        v = s.ref_arc >= 0
        correct += int((arcs[v] == s.ref_arc[v]).sum())
        valid += int(v.sum())
        pairs, adj = continuity(collapse(arcs), graphs[s.trip_id])
        breaks += pairs - adj
        obs += len(arcs)
    return correct / max(1, valid), breaks / max(1, obs) * 1000


def tune_decoder(seqs, graphs, out: Path):
    grid = {
        "hop_penalty": [0.5, 1.0, 2.0],
        "jump_penalty": [4.0, 8.0, 16.0],
        "emission_power": [0.5, 1.0, 2.0],
        "max_hops": [2, 3],
    }
    keys = list(grid)
    trials = []
    t0 = time.perf_counter()
    for i, vals in enumerate(itertools.product(*grid.values()), 1):
        params = DecoderParams(**dict(zip(keys, vals)))
        acc, brk = quick_score(seqs, graphs, decode_all(seqs, params))
        trials.append({**dict(zip(keys, vals)), "pointwise_top1": acc, "breaks_per_1000_obs": brk})
        print(f"  trial {i:02d}: {dict(zip(keys, vals))} -> top1={acc:.4f}, breaks/1000={brk:.1f}", flush=True)
    elapsed = time.perf_counter() - t0
    tdf = pd.DataFrame(trials).sort_values(["pointwise_top1", "breaks_per_1000_obs"], ascending=[False, True])
    tdf.to_csv(out / "hmm_tuning_trials.csv", index=False)
    best = tdf.iloc[0]
    params = DecoderParams(hop_penalty=float(best["hop_penalty"]), jump_penalty=float(best["jump_penalty"]),
                           emission_power=float(best["emission_power"]), max_hops=int(best["max_hops"]))
    return params, tdf, elapsed, grid


def report(seqs, graphs, params, out: Path, prefix: str):
    t0 = time.perf_counter()
    preds = decode_all(seqs, params)
    decode_seconds = time.perf_counter() - t0
    rows = []
    for s in seqs:
        rows += evaluate_trip(s, graphs[s.trip_id], preds[s.trip_id])
    per_trip = pd.DataFrame(rows)
    summary = summarise(per_trip)
    summary["decode_seconds_total"] = decode_seconds
    summary["decode_ms_per_observation"] = 1000 * decode_seconds / max(1, summary["observations"].iloc[0])
    per_trip.to_csv(out / f"{prefix}_per_trip.csv", index=False)
    summary.to_csv(out / f"{prefix}_summary.csv", index=False)
    pd.concat([pd.DataFrame({"trip_id": s.trip_id, "obs_idx": s.obs_idx, "classifier_arc": s.clf_arc,
                             "hmm_arc": preds[s.trip_id], "reference_arc_id": s.ref_arc}) for s in seqs]
              ).to_csv(out / f"{prefix}_predictions.csv", index=False)
    plot_per_trip(per_trip, out / f"{prefix}_per_trip_top1.png")
    return summary, per_trip


def plot_per_trip(per_trip: pd.DataFrame, path: Path):
    pv = per_trip.pivot(index="trip_id", columns="method", values="pointwise_top1")
    ax = pv.plot(kind="bar", figsize=(9, 4), color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Pointwise Top-1"); ax.set_title("Classifier-only vs classifier + HMM (per trip)")
    ax.set_ylim(0, 1.02); ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ----------------------------------------------------------------------------- API entry point
def reconstruct_trip(trip: dict, model, params: DecoderParams, top_k: int = 15):
    """
    Reconstruct the route for one raw trip (dict from pathsnap_pipeline.load_trip).
    The reference route is blanked first, so no ground truth is used for prediction.
    Returns (DataFrame[obs_idx, classifier_arc, hmm_arc], collapsed HMM route as a list of arc ids).
    """
    blind = {**trip, "route": np.array([], dtype=int)}
    frame, _ = P.build_candidates_for_trip(blind, "inference", top_k=top_k)
    if frame.empty:
        return pd.DataFrame(columns=["obs_idx", "classifier_arc", "hmm_arc"]), []
    prob = model.predict_proba(frame[P.FEATURES])[:, 1]
    graph = TripGraph(trip["arcs"], len(trip["nodes"]))
    seq = build_trip_seq(trip["trip_id"], frame, trip["track"][:, 2], graph, prob)
    hmm_arc, _ = viterbi(seq, params)
    out = pd.DataFrame({"obs_idx": seq.obs_idx, "classifier_arc": seq.clf_arc, "hmm_arc": hmm_arc})
    return out, collapse(hmm_arc)


def load_params(path: Path) -> DecoderParams:
    cfg = json.loads(path.read_text())["params"]
    return DecoderParams(**cfg)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data_work/results/best_tuned_model.joblib")
    ap.add_argument("--processed", default="data_work/processed")
    ap.add_argument("--data-dir", default="data", help="extracted dataset folder containing train/ and student_test/")
    ap.add_argument("--out", default="data_work/hmm_results")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--tune-on", choices=["student", "train", "both"], default="student",
                    help="which permitted data tunes the decoder parameters (never the evaluator set)")
    ap.add_argument("--tune-train-trips", type=int, default=15, help="number of training trips sampled when tuning on train")
    ap.add_argument("--final-dir", default=None,
                    help="FINAL evaluation folder (e.g. evaluator_test). Uses saved hmm_config.json; no tuning.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.model)

    if args.final_dir:
        print("FINAL evaluation: decoder parameters come from hmm_config.json (not tuned here).")
        params = load_params(out / "hmm_config.json")
        seqs, graphs = seqs_from_raw_dir(Path(args.final_dir), model, args.top_k)
        summary, per_trip = report(seqs, graphs, params, out, "final")
        print(summary.to_string(index=False))
        print(per_trip.pivot(index="trip_id", columns="method", values="pointwise_top1").to_string())
        print(f"\nSaved final_* files under {out.resolve()}")
        return

    root = P.get_dataset_root(Path(args.data_dir))
    processed = Path(args.processed)
    student_df = pd.read_parquet(processed / "student_test_candidates.parquet")
    student_seqs, student_graphs = seqs_from_candidate_frame(student_df, root / "student_test", model)

    tune_seqs, tune_graphs = [], {}
    if args.tune_on in ("student", "both"):
        tune_seqs += student_seqs
        tune_graphs.update(student_graphs)
    if args.tune_on in ("train", "both"):
        train_df = pd.read_parquet(processed / "train_candidates.parquet")
        ids = np.array(sorted(train_df["trip_id"].unique()))
        rng = np.random.default_rng(P.RANDOM_STATE)
        pick = rng.choice(ids, size=min(args.tune_train_trips, len(ids)), replace=False)
        tr_seqs, tr_graphs = seqs_from_candidate_frame(train_df, root / "train", model, trips=set(pick))
        tune_seqs += tr_seqs
        tune_graphs.update(tr_graphs)

    print(f"Tuning decoder on {len(tune_seqs)} trips (tune-on={args.tune_on}) ...")
    params, trials, elapsed, grid = tune_decoder(tune_seqs, tune_graphs, out)
    (out / "hmm_config.json").write_text(json.dumps({
        "params": asdict(params), "tuned_on": args.tune_on, "tuning_trips": len(tune_seqs),
        "objective": "pointwise Top-1 (ties: fewer route breaks)", "grid": grid,
        "optimisation_seconds": elapsed, "random_state": P.RANDOM_STATE, "top_k": args.top_k,
    }, indent=2))
    print("\nBest decoder parameters:", asdict(params))

    summary, _ = report(student_seqs, student_graphs, params, out, "student")
    print("\nClassifier-only vs classifier + HMM on student_test:")
    print(summary.to_string(index=False))
    if args.tune_on != "train":
        print("\nNote: decoder parameters were tuned on student_test, so these numbers are slightly optimistic. "
              "The classifier itself never saw these trips.")
    print(f"\nAll outputs saved under {out.resolve()}")


if __name__ == "__main__":
    main()