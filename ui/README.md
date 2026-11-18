# PathSnap – Evaluator UI (Laptop 2)

Streamlit + Folium interface for the two-laptop demo. The UI lists the evaluation trips offered by the
API on Laptop 1, sends **only the selected trip ID** to the API, and shows the returned route and scores.
It contains no data, model or reference routes – all prediction and scoring happens on Laptop 1.

## Tech

| Tool | Purpose |
|---|---|
| Streamlit | Web interface (trip buttons, metrics, tables) |
| Folium + streamlit-folium | Interactive map (Leaflet / OpenStreetMap) |
| requests | HTTP calls to the FastAPI backend |
| pandas | Comparison and session tables |

## Run

1. Start the API on Laptop 1 and note its address, e.g. `http://10.1.25.41:8000`.
2. Start the UI:
   - Windows: double-click `START_UI.bat`
   - macOS / Linux: `bash start_ui.sh`
   - or manually:
     ```bash
     pip install -r requirements.txt
     streamlit run app.py
     ```
3. Open http://localhost:8501, enter the API address in the sidebar and press Enter
   (it should show "Connected · trips: 10").
4. Pick a trip and click **Reconstruct route**.

Both laptops must be on the same network. Quick check from Laptop 2: open `http://<laptop-1-ip>:8000/health`.

## API calls

| Call | Used for |
|---|---|
| `GET /health` | Connection status in the sidebar |
| `GET /trips` | The trip buttons (accepts `["id", ...]` or `[{"trip_id": "id", ...}, ...]`) |
| `POST /predict` `{"trip_id": "..."}` | Route, GPS points and scores for one trip |

`normalize()` in `app.py` accepts both response formats used in the project
(`metrics.classifier_plus_hmm` / `predicted_route_coords`, or `comparison` / `predicted_route`).

## What is shown

- Top-1 accuracy, route continuity, GPS points and server time for the selected trip
- Classifier-only vs classifier + HMM/Viterbi comparison table
- Folium map: raw GPS points (blue), predicted route (red), road network (grey, when provided),
  start/end markers, legend and layer control
- Table of all trips reconstructed in the session with mean scores
- "Download this map (HTML)" for the report
