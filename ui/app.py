"""
PathSnap evaluator UI (Laptop 2).

Shows the trips the API offers, sends the chosen trip_id to the API and shows
the returned route and metrics on a Folium map. The UI reads no data files and
computes no predictions itself - everything comes from the API.

Run on Laptop 2:
    streamlit run ui/app.py
Then set the API address in the sidebar, e.g. http://<laptop-1-ip>:8000
"""
from __future__ import annotations

import os
import time

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")

METRIC_LABELS = {
    "observation_top1_accuracy": "Top-1 accuracy",
    "route_continuity": "Route continuity",
    "lcs_precision": "LCS precision",
    "lcs_recall": "LCS recall",
    "normalized_edit_similarity": "Edit similarity",
}

# ---------------- sidebar: API connection ----------------
st.sidebar.header("API connection")
api_url = st.sidebar.text_input(
    "API address (Laptop 1)",
    value=os.environ.get("PATHSNAP_API_URL", "http://localhost:8000"),
    help="Use Laptop 1's IP on the same Wi-Fi, e.g. http://192.168.1.20:8000",
).rstrip("/")

try:
    t0 = time.perf_counter()
    health = requests.get(f"{api_url}/health", timeout=5).json()
    ping_ms = (time.perf_counter() - t0) * 1000
    n_trips = health.get("trips", health.get("trips_loaded", "?"))
    st.sidebar.success(f"Connected ({ping_ms:.0f} ms)\n\nStatus: {health.get('status')} · trips: {n_trips}")
except requests.RequestException as exc:
    st.sidebar.error(f"Can't reach the API at {api_url}\n\n{exc}")
    st.title("PathSnap – Route Reconstruction")
    st.info("Start the API on Laptop 1 and enter its address in the sidebar.")
    st.stop()

st.sidebar.divider()
st.sidebar.caption("The UI sends only a trip ID. Prediction and scoring run on Laptop 1; "
                   "reference routes never leave the API.")


@st.cache_data(ttl=60)
def get_trips(url: str) -> list[str]:
    """Trip IDs from the API. Accepts ["id", ...] or [{"trip_id": "id", ...}, ...]."""
    trips = requests.get(f"{url}/trips", timeout=10).json()["trips"]
    return [t["trip_id"] if isinstance(t, dict) else str(t) for t in trips]


def normalize(res: dict) -> dict:
    """Bring the API response into one shape, whichever backend version is running.

    Supports both response formats used by the team:
      A) metrics.observation_top1_accuracy / comparison / predicted_route (segments)
      B) metrics.classifier_only + metrics.classifier_plus_hmm / predicted_route_coords (polyline)
    """
    if "comparison" in res or "classifier_plus_hmm" not in res.get("metrics", {}):
        res.setdefault("road_network", [])
        return res

    def conv(m: dict) -> dict:
        return {"observation_top1_accuracy": m.get("pointwise_accuracy"),
                "route_continuity": m.get("route_continuity"),
                "normalized_edit_similarity": m.get("sequence_similarity"),
                "disconnected_transitions": m.get("route_breaks")}

    raw = res["metrics"]
    hmm, cls = conv(raw["classifier_plus_hmm"]), conv(raw["classifier_only"])
    return {
        "trip_id": res["trip_id"],
        "method": "classifier+hmm",
        "metrics": {**hmm, "gps_points": res.get("observations", len(res.get("gps_points", []))),
                    "inference_seconds": res.get("processing_seconds", 0.0)},
        "comparison": {"classifier_only": cls, "classifier+hmm": hmm},
        "gps_points": res["gps_points"],
        "predicted_route": res.get("predicted_route_coords", []),   # one polyline of [lat, lon]
        "road_network": res.get("road_network", []),
    }


def pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.1f}%"


# ---------------- trip selection ----------------
st.title("PathSnap – Route Reconstruction")
st.caption("Choose one of the evaluation trips. The API runs the full prediction pipeline "
           "and scores the result against its own reference data.")

trips = get_trips(api_url)
history: dict = st.session_state.setdefault("history", {})

trip_id = st.pills(
    f"Evaluation trips ({len(trips)})",
    trips,
    selection_mode="single",
    default=trips[0] if trips else None,
    key="trip_choice",
)
done = [t for t in trips if t in history]
st.caption(f"Reconstructed this session: {', '.join(done)}" if done else "No trips reconstructed yet.")
run = st.button("Reconstruct route", type="primary", disabled=trip_id is None)

if run and trip_id:
    with st.spinner(f"Laptop 1 is processing trip {trip_id} ..."):
        try:
            t0 = time.perf_counter()
            resp = requests.post(f"{api_url}/predict", json={"trip_id": trip_id}, timeout=300)
            resp.raise_for_status()
            result = normalize(resp.json())
            result["round_trip_seconds"] = time.perf_counter() - t0
            st.session_state["result"] = result
            history[trip_id] = {
                "method": result["method"],
                **{k: result["metrics"].get(k) for k in ("observation_top1_accuracy", "route_continuity",
                                                         "gps_points", "inference_seconds")},
            }
            st.rerun()  # refresh the "reconstructed" list above
        except requests.RequestException as exc:
            st.error(f"Prediction failed: {exc}")

result = st.session_state.get("result")
if not result:
    st.info("Select a trip and press **Reconstruct route**.")
    st.stop()

# ---------------- metrics ----------------
m = result["metrics"]
method = result["method"]
st.subheader(f"Trip {result['trip_id']}")
st.caption(f"Method: **{method}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Top-1 accuracy", pct(m["observation_top1_accuracy"]),
          help="Share of GPS points matched to the reference road segment (scored on the API)")
c2.metric("Route continuity", pct(m["route_continuity"]),
          help="Share of road changes where the next predicted segment touches the previous one")
c3.metric("GPS points", f"{m['gps_points']:,}")
c4.metric("Server time", f"{m['inference_seconds']:.2f} s",
          help=f"Model + decoding on Laptop 1. Full request incl. network: "
               f"{result.get('round_trip_seconds', 0):.2f} s")

# Classifier-only vs classifier+HMM, shown when the API sends both
comparison = result.get("comparison")
if comparison:
    st.markdown("**Classifier only vs classifier + HMM/Viterbi**")
    rows = []
    for name, vals in comparison.items():
        rows.append({"Method": name, **{METRIC_LABELS.get(k, k): pct(vals.get(k)) for k in METRIC_LABELS},
                     "Disconnected jumps": vals.get("disconnected_transitions", "–")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ---------------- map ----------------
def build_map(res: dict) -> folium.Map:
    gps = res["gps_points"]
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=15, tiles="OpenStreetMap")

    roads = folium.FeatureGroup(name="Road network (grey)")
    if res["road_network"]:
        folium.PolyLine(res["road_network"], color="#777777", weight=1.5, opacity=0.6).add_to(roads)
    roads.add_to(fmap)

    if res.get("classifier_route"):  # optional: classifier-only route when HMM output is the main route
        cls = folium.FeatureGroup(name="Classifier-only route (orange)", show=False)
        folium.PolyLine(res["classifier_route"], color="orange", weight=4, opacity=0.8).add_to(cls)
        cls.add_to(fmap)

    route = folium.FeatureGroup(name="Predicted route (red)")
    if res.get("predicted_route"):
        folium.PolyLine(res["predicted_route"], color="red", weight=5, opacity=0.85).add_to(route)
    route.add_to(fmap)

    pts = folium.FeatureGroup(name="Raw GPS observations (blue)")
    step = max(1, len(gps) // 1500)  # keep the map responsive on long trips
    for lat, lon in gps[::step]:
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True, fill_opacity=0.8).add_to(pts)
    pts.add_to(fmap)

    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)

    extra = '<span style="color:orange">&#9644;</span> Classifier-only route<br>' if res.get("classifier_route") else ""
    legend = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
                padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px; color: #222;">
      <b>Trip {res['trip_id']}</b><br>
      <span style="color:blue">&#9679;</span> Raw GPS observations<br>
      <span style="color:red">&#9644;</span> Predicted route<br>
      {extra}
      <span style="color:#777">&#9644;</span> Road network
    </div>"""
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(fmap)
    lats = [p[0] for p in gps]
    lons = [p[1] for p in gps]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    return fmap


fmap = build_map(result)
st_folium(fmap, height=620, use_container_width=True, returned_objects=[], key=f"map_{result['trip_id']}")

st.download_button(
    "Download this map (HTML)",
    data=fmap.get_root().render(),
    file_name=f"pathsnap_trip_{result['trip_id']}.html",
    mime="text/html",
)

# ---------------- session summary ----------------
if history:
    st.subheader(f"Trips reconstructed this session ({len(history)}/{len(trips)})")
    df = pd.DataFrame.from_dict(history, orient="index").rename_axis("Trip").reset_index()
    avg = df[["observation_top1_accuracy", "route_continuity"]].mean()
    a1, a2 = st.columns(2)
    a1.metric("Mean Top-1 accuracy", pct(avg["observation_top1_accuracy"]))
    a2.metric("Mean route continuity", pct(avg["route_continuity"]))
    show = df.assign(
        **{"Top-1 accuracy": df["observation_top1_accuracy"].map(pct),
           "Route continuity": df["route_continuity"].map(pct),
           "GPS points": df["gps_points"],
           "Server time (s)": df["inference_seconds"].round(2)}
    )[["Trip", "method", "Top-1 accuracy", "Route continuity", "GPS points", "Server time (s)"]]
    st.dataframe(show.rename(columns={"method": "Method"}), hide_index=True, use_container_width=True)

with st.expander("Raw API response (metrics only)"):
    st.json({k: v for k, v in result.items()
             if k not in ("gps_points", "predicted_route", "road_network", "classifier_route")})
