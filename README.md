# PathSnap: Distributed Map-Matching and Route Reconstruction

PathSnap is a GPS-based road map-matching system that reconstructs a likely vehicle route from noisy GPS observations and a directed road network.

A GPS point may be displaced from the road because of measurement noise, nearby parallel roads, intersections, or irregular sampling. PathSnap addresses this by generating nearby road-arc candidates, scoring them with machine learning, and using an HMM with Viterbi decoding to produce a route that is more consistent with road-network connectivity.

The project is organised as a distributed application:

- **Laptop 1:** Python processing pipeline, trained model, HMM/Viterbi decoder, and FastAPI backend.
- **Laptop 2:** Streamlit user interface, which sends trip requests to the backend and displays the reconstructed route and evaluation metrics.

---

## Table of contents

- [System overview](#system-overview)
- [Team](#team)
- [Technology stack](#technology-stack)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [How the system works](#how-the-system-works)
- [Setup](#setup)
- [Run the machine-learning pipeline](#run-the-machine-learning-pipeline)
- [Run HMM route reconstruction](#run-hmm-route-reconstruction)
- [Run the FastAPI backend](#run-the-fastapi-backend-laptop-1)
- [Run the Streamlit UI](#run-the-streamlit-ui-laptop-2)
- [API endpoints](#api-endpoints)
- [Model evaluation results](#model-evaluation-results)
- [Evaluation metrics](#evaluation-metrics)
- [Generated outputs](#generated-outputs)
- [Limitations and important notes](#limitations-and-important-notes)
- [Reproducibility](#reproducibility)

---

## System overview

PathSnap uses three main stages:

1. **Candidate generation and feature extraction:** For every GPS observation, the pipeline finds nearby road arcs and calculates features describing the relationship between the observation and each candidate.
2. **Machine-learning classification and sequence decoding:** A classifier estimates the probability of each candidate being the correct road arc. The HMM/Viterbi decoder combines these probabilities with directed road-network connectivity to select a coherent sequence.
3. **API and visualisation:** FastAPI exposes the prediction workflow. The Streamlit UI on another laptop lets a user select an evaluation trip and view the reconstructed route and its metrics.

### Architecture

```text
                    GPS Tracks + Road Network
                               |
                               v
                    Dataset Loading and Checks
                               |
                               v
                       Candidate Generation
                      (Top-K nearby road arcs)
                               |
                               v
                        Feature Extraction
                               |
                               v
                    Machine-Learning Classifiers
                               |
                               v
                       Model Comparison
                   (5-fold GroupKFold by trip)
                               |
                               v
                       Bayesian Optimisation
                               |
                               v
                       Tuned Classifier
                               |
                               v
                     Candidate Probabilities
                               |
                               v
                         HMM + Viterbi
                               |
                               v
                       Reconstructed Route
                               |
                               v
                         FastAPI Backend
                               |
                      Network / HTTP Requests
                               |
                               v
                        Streamlit Frontend
                               |
                               v
                     Route Map and Metrics
```

---

## Team

| Member | Name               | SRN           | Workstream                                                                                         |
| ------ | ------------------ | ------------- | -------------------------------------------------------------------------------------------------- |
| 1      | Zia Kadijah        | PES2UG24CS620 | Data and features: dataset exploration, preprocessing, candidate generation and feature extraction |
| 2      | Vaibhav R          | PES2UG24CS664 | Machine learning and optimisation: model training, comparison and Bayesian optimisation            |
| 3      | Jyeshta J          | PES2UG24CS637 | Route reconstruction: HMM, Viterbi and inference backend                                           |
| 4      | Parnika V Hallalli | PES2UG24CS645 | UI and deployment: Streamlit frontend, map visualisation, integration and testing                  |

---

## Technology stack

| Area                                 | Technologies                                                                 |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| Programming language                 | Python                                                                       |
| Data handling                        | NumPy, Pandas                                                                |
| Machine learning                     | Scikit-learn                                                                 |
| Hyperparameter optimisation          | scikit-optimize                                                              |
| Spatial indexing                     | SciPy `cKDTree`                                                              |
| Model persistence                    | Joblib                                                                       |
| Data storage for intermediate tables | Parquet / PyArrow                                                            |
| Plotting                             | Matplotlib                                                                   |
| Route reconstruction                 | Hidden Markov Model and Viterbi algorithm                                    |
| Backend                              | FastAPI                                                                      |
| ASGI server                          | Uvicorn                                                                      |
| Request validation                   | Pydantic                                                                     |
| Frontend                             | Streamlit                                                                    |
| Map visualisation                    | Folium (used by the project UI, if enabled in the installed UI dependencies) |

---

## Repository structure

The following reflects the project layout shown for the backend and experiment workspace. The `ui/` directory is maintained by the frontend workstream and should be present in the shared repository after the UI changes are merged.

```text
code/
├── configs/                  # Project configuration files
├── data/                     # Extracted development dataset
│   ├── train/
│   └── student_test/
├── evaluator_test/           # Reserved final-evaluation dataset
├── processed/                # Cached candidate tables and diagnostics
├── results/                  # Model comparison, trained models and plots
├── hmm_results/              # HMM configuration and decoder evaluation
├── eval_results/             # Final evaluation outputs
├── src/
│   ├── HMM_route.py          # HMM/Viterbi decoder and decoder evaluation
│   └── pathsnap_pipeline.py  # Data processing, features, model training and tuning
├── api.py                    # FastAPI backend
├── requirements.txt          # Backend and ML dependencies
├── README.md
└── ui/                       # Streamlit frontend (when merged into this checkout)
    ├── .streamlit/
    ├── app.py
    ├── requirements.txt
    ├── START_UI.bat
    ├── start_ui.sh
    └── README.md
```

Generated folders such as `processed/`, `results/`, `hmm_results/`, and `eval_results/` are created or populated by running the corresponding stages. The datasets are expected to be available locally; do not assume that the full dataset is committed to the repository.

---

## Dataset

### Expected dataset layout

The processing pipeline expects an extracted dataset containing `train/` and `student_test/` directories. The final evaluation data is kept separately.

```text
data/
├── train/
│   └── <trip_id>/
│       ├── <trip_id>.track
│       ├── <trip_id>.nodes
│       ├── <trip_id>.arcs
│       └── <trip_id>.route
└── student_test/
    └── <trip_id>/
        ├── <trip_id>.track
        ├── <trip_id>.nodes
        ├── <trip_id>.arcs
        └── <trip_id>.route

evaluator_test/
└── <trip_id>/
    ├── <trip_id>.track
    ├── <trip_id>.nodes
    ├── <trip_id>.arcs
    └── <trip_id>.route
```

Each trip is stored in its own directory. The four whitespace-separated files are:

| File     | Contents                                                                                            |
| -------- | --------------------------------------------------------------------------------------------------- |
| `.track` | GPS longitude, latitude and timestamp                                                               |
| `.nodes` | Road-network node longitude and latitude; row position is the node index                            |
| `.arcs`  | Directed arc endpoints: source node index and destination node index; row position is the arc index |
| `.route` | Reference route represented as an ordered sequence of arc IDs                                       |

Node and arc indices belong to an individual trip's road network. They are not globally meaningful across trips and are not used as classifier features.

The `.route` sequence is not assumed to be row-aligned with the GPS observations. The training pipeline aligns GPS observations to the supplied route using a monotone dynamic-programming procedure.

### Dataset partitions and data integrity

| Partition         | Intended use                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------- |
| `train/`          | Classifier training and cross-validation; may also be used for permitted decoder tuning  |
| `student_test/`   | Development evaluation and decoder tuning, depending on the selected `--tune-on` setting |
| `evaluator_test/` | Final evaluation only, after the model and decoder configuration have been fixed         |

**Important:** Do not use `evaluator_test/` to train models, select features, select a model, or tune classifier/HMM parameters. The final evaluator run loads the already-saved model and HMM configuration.

---

## How the system works

### 1. Candidate generation

For each GPS observation, the pipeline searches for nearby road arcs and retains up to `K` candidates. The default is `K = 15`.

Road arcs are sampled at approximately 50-metre intervals to avoid overlooking long arcs. A spatial KDTree is used to find nearby samples efficiently, followed by point-to-segment distance calculations.

Candidate generation is performed independently of the reference route. The reference route is used for creating labels and evaluating predictions, not for deciding which candidates are generated.

### 2. Feature extraction

The classifier receives 11 features for each GPS-observation/candidate-arc pair:

| Feature                  | Meaning                                                                          |
| ------------------------ | -------------------------------------------------------------------------------- |
| `distance_m`             | Distance from the GPS point to the candidate road segment                        |
| `distance_ratio`         | Normalised distance feature                                                      |
| `segment_length_m`       | Length of the candidate segment                                                  |
| `projection_fraction`    | Position of the projected GPS point along the segment                            |
| `heading_difference_rad` | Difference between GPS movement heading and road-segment heading                 |
| `heading_cos`            | Cosine of the heading difference                                                 |
| `heading_sin`            | Sine of the heading difference                                                   |
| `gps_speed_mps`          | Smoothed GPS speed in metres per second                                          |
| `dist_rank`              | Candidate's distance rank among candidates for the same observation              |
| `dist_minus_min_m`       | Candidate distance minus the nearest candidate distance                          |
| `dist_share`             | Candidate distance relative to the total candidate distance for that observation |

The relative features help the classifier compare candidates belonging to the same GPS observation.

### 3. Reference-label alignment

The `.route` file provides an ordered reference sequence, but it is not directly aligned one-to-one with GPS points. The pipeline uses monotone dynamic programming to associate GPS observations with positions along the reference route.

The alignment cost uses point-to-segment distance and a small step penalty. This provides approximate labels for supervised training.

### 4. Classifier training and model selection

The current pipeline compares five classifiers:

1. Logistic Regression
2. Random Forest
3. HistGradientBoosting
4. K-Nearest Neighbours (KNN)
5. Multi-Layer Perceptron (MLP)

Five-fold `GroupKFold` cross-validation is used. The group is the trip ID, so observations from a trip are kept together rather than being randomly distributed across training and validation folds.

The primary model-selection metric is **observation-level CV Top-1 accuracy**. CV log loss is used as a tie-breaker.

Training observations are thinned using the `--thin` setting because adjacent GPS samples can be near-duplicates. The student-test observations are not thinned.

### 5. Bayesian hyperparameter optimisation

After selecting the classifier, the pipeline uses `BayesSearchCV` from scikit-optimize to search for better hyperparameters.

- The objective is observation-level Top-1 accuracy.
- Cross-validation during the search is group-aware by trip.
- A fraction of observations is sampled within each training trip to make the search more manageable.
- The selected configuration is refitted on the full eligible training data.
- The tuned and untuned versions are compared on `student_test/`.

The number of trials is controlled by `--bayes-iter`. The source-code default is 10; use the value from the actual final run when reporting the experiment.

### 6. HMM and Viterbi route reconstruction

The classifier scores candidates independently. A high-probability candidate at one observation may not connect to the candidate selected at the previous observation. The HMM decoder uses road-network connectivity to find a more coherent sequence.

- **Hidden state:** one of the candidate road arcs at each GPS observation.
- **Emission score:** the classifier probability transformed using `emission_power * log(p)`.
- **Transition score:** based on directed hops between consecutive candidate arcs.
- **Viterbi decoding:** finds the sequence with the highest combined emission and transition score, using back-pointers and backtracking.

The decoder parameters include:

- `hop_penalty`
- `jump_penalty`
- `emission_power`
- `max_hops`
- `gap_reset_s`

Staying on the same arc has no hop cost. Reachable transitions are penalised according to the number of directed hops. Unreachable transitions receive a larger penalty, but are not made impossible, so the decoder can still return a path.

GPS time gaps are also considered. Larger gaps can relax the hop limit, while a gap greater than `gap_reset_s` starts a new independent decoding segment.

The decoder parameters are tuned using permitted training/student data only. The evaluator partition is not used for tuning.

---

## Setup

### Requirements

Python 3.10–3.12 is recommended. Create a virtual environment from the `code/` directory.

**Windows PowerShell:**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the API dependencies are not already included in the root `requirements.txt`, install them:

```bash
pip install fastapi "uvicorn[standard]" pydantic
```

The pipeline also requires the dependencies listed in `requirements.txt`, including NumPy, Pandas, SciPy, scikit-learn, scikit-optimize, Matplotlib, Joblib and PyArrow.

### Check the dataset

Before running the pipeline, confirm that the extracted `data/` directory contains the expected `train/` and `student_test/` folders. The final evaluation data should be in `evaluator_test/`.

---

## Run the machine-learning pipeline

Run commands from the `code/` project directory.

```bash
python src/pathsnap_pipeline.py --data-dir data --workdir . --top-k 15 --thin 4 --bayes-iter 2 --tune-frac 0.5
```

The example above uses two Bayesian optimisation trials. Change `--bayes-iter` to match the number of trials you intend to run.

### Pipeline arguments

| Argument       |     Default | Description                                                         |
| -------------- | ----------: | ------------------------------------------------------------------- |
| `--data-dir`   |      `data` | Extracted dataset directory containing `train/` and `student_test/` |
| `--workdir`    | `data_work` | Working directory for generated `results/` and `processed/` folders |
| `--top-k`      |        `15` | Maximum number of candidate road arcs retained per GPS observation  |
| `--thin`       |         `4` | Keep every Nth GPS observation for training, CV and tuning          |
| `--bayes-iter` |        `10` | Number of Bayesian optimisation trials                              |
| `--tune-frac`  |       `0.5` | Fraction of training observations used within the Bayesian search   |

Using `--workdir .` writes outputs into the `processed/` and `results/` directories at the project root, matching the paths used in the example HMM and API commands below.

The pipeline:

1. Inspects the dataset and writes inventory/summary files.
2. Generates candidate arcs and extracts features for training and student-test trips.
3. Trains and compares the five classifiers using trip-grouped cross-validation.
4. Selects the model using CV observation-level Top-1 accuracy.
5. Performs Bayesian hyperparameter optimisation on the selected model.
6. Saves the tuned model, configuration, comparison files and plots.

---

## Run HMM route reconstruction

Run decoder tuning after the machine-learning pipeline has completed successfully and generated the candidate Parquet files and tuned model.

```bash
python src/HMM_route.py --model results/best_tuned_model.joblib --processed processed --data-dir data --out hmm_results --tune-on student
```

The `--tune-on` argument accepts:

- `student` — tune decoder parameters using student-test trips (the source-code default).
- `train` — tune using a sampled set of training trips.
- `both` — tune using both permitted sets.

For `train` or `both`, `--tune-train-trips` controls the number of training trips sampled for decoder tuning.

Example:

```bash
python src/HMM_route.py --model results/best_tuned_model.joblib --processed processed --data-dir data --out hmm_results --tune-on train --tune-train-trips 15
```

The tuning run saves `hmm_results/hmm_config.json` and reports classifier-only versus classifier-plus-HMM results on the student trips.

### Final evaluator run

Only after the classifier and decoder parameters are fixed, run the final evaluation:

```bash
python src/HMM_route.py --model results/best_tuned_model.joblib --out hmm_results --final-dir evaluator_test
```

This mode loads the saved `hmm_results/hmm_config.json`. It does not tune the decoder on evaluator data. Final evaluation reports are written under `hmm_results/` with the `final_` prefix; the project may also collect or copy them into `eval_results/`.

---

## Run the FastAPI backend (Laptop 1)

The API file is at the project root (`api.py`). It imports the processing and HMM modules from `src/`, so `--app-dir src` is included to make those modules importable.

Run from the `code/` directory after the model and HMM configuration have been generated:

```bash
python -m uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
```

The backend loads:

- `results/best_tuned_model.joblib`
- `hmm_results/hmm_config.json`
- The evaluation trips from `evaluator_test/`

These paths can be overridden with environment variables:

| Environment variable  | Default                           | Purpose                                  |
| --------------------- | --------------------------------- | ---------------------------------------- |
| `PATHSNAP_MODEL`      | `results/best_tuned_model.joblib` | Trained classifier file                  |
| `PATHSNAP_HMM_CONFIG` | `hmm_results/hmm_config.json`     | Saved HMM decoder configuration          |
| `PATHSNAP_EVAL_DIR`   | `evaluator_test`                  | Evaluation dataset directory             |
| `PATHSNAP_TOP_K`      | `15`                              | Number of candidate arcs per observation |

The API loads the evaluation trips and their reference information at startup. Reference routes remain on the backend and are used only to calculate evaluation metrics after prediction. They are not returned to the frontend.

The backend listens on all network interfaces using `0.0.0.0`, allowing the other laptop to reach it over the local network.

---

## Run the Streamlit UI (Laptop 2)

The frontend is maintained in the `ui/` directory. It contains:

```text
ui/
├── .streamlit/
├── README.md
├── START_UI.bat
├── app.py
├── requirements.txt
└── start_ui.sh
```

On Laptop 2:

1. Ensure both laptops are connected to the same network.
2. Start the FastAPI backend on Laptop 1.
3. Find Laptop 1's local IPv4 address.
4. Configure the frontend to use Laptop 1's address and port `8000` wherever the API base URL is defined in `ui/app.py` or its Streamlit configuration.
5. Open a terminal in the `ui/` directory.
6. Install the UI dependencies:

   ```bash
   pip install -r requirements.txt
   ```

7. Start the UI.

   **Windows:**

   ```powershell
   .\START_UI.bat
   ```

   **Linux/macOS:**

   ```bash
   bash start_ui.sh
   ```

The UI startup scripts are provided by the frontend workstream. Refer to `ui/README.md` for any additional frontend-specific configuration.

### Network checks

- Laptop 1 must allow inbound connections on port `8000` through its firewall.
- The frontend must use Laptop 1's reachable LAN IPv4 address, not `localhost` or `127.0.0.1`. Those addresses would refer to Laptop 2 itself when used from the frontend laptop.
- Check that the backend is reachable from Laptop 2 by opening `http://<LAPTOP-1-IP>:8000/health` in a browser.
- FastAPI's interactive documentation is available at `http://<LAPTOP-1-IP>:8000/docs`.

---

## API endpoints

| Method | Endpoint   | Description                                                                         |
| ------ | ---------- | ----------------------------------------------------------------------------------- |
| `GET`  | `/`        | Returns basic service information and endpoint paths                                |
| `GET`  | `/health`  | Returns server status, number of loaded trips and whether the model is loaded       |
| `GET`  | `/trips`   | Lists available evaluation trip IDs and GPS point counts                            |
| `POST` | `/predict` | Reconstructs a selected trip and returns the predicted route and evaluation metrics |
| `GET`  | `/docs`    | Interactive Swagger API documentation                                               |

### Example prediction request

```http
POST /predict
Content-Type: application/json
```

```json
{
  "trip_id": "00000015"
}
```

### Response structure

The `/predict` endpoint returns the following fields:

```json
{
  "trip_id": "00000015",
  "observations": 120,
  "gps_points": [[12.9716, 77.5946]],
  "predicted_route_arc_ids": [14, 15, 18],
  "predicted_route_coords": [
    [12.9716, 77.5946],
    [12.972, 77.595]
  ],
  "metrics": {
    "observations_scored": 120,
    "classifier_only": {
      "pointwise_accuracy": 0.0,
      "sequence_similarity": 0.0,
      "route_continuity": 0.0,
      "route_breaks": 0
    },
    "classifier_plus_hmm": {
      "pointwise_accuracy": 0.0,
      "sequence_similarity": 0.0,
      "route_continuity": 0.0,
      "route_breaks": 0
    }
  },
  "processing_seconds": 0.0
}
```

The values above illustrate the response shape only; they are not measured project results. Coordinates are returned as `[latitude, longitude]` pairs.

---

## Model evaluation results

The following values are from the supplied `results/model_comparison.csv` experiment output. They describe the classifier comparison before the separate tuned-versus-untuned and HMM route comparisons.

| Model                | CV Observation Top-1 | CV Macro-F1 | CV Log Loss | Student Observation Top-1 | Student Macro-F1 | Student Log Loss | Fit Time (s) |
| -------------------- | -------------------: | ----------: | ----------: | ------------------------: | ---------------: | ---------------: | -----------: |
| Random Forest        |           **0.8397** |  **0.9243** |  **0.0796** |                    0.8721 |       **0.9438** |           0.0589 |        32.58 |
| HistGradientBoosting |               0.8395 |      0.9197 |      0.0840 |                **0.8728** |           0.9437 |       **0.0561** |         3.49 |
| MLP                  |               0.8391 |      0.9216 |      0.1165 |                    0.8656 |           0.9410 |           0.0599 |        35.31 |
| KNN                  |               0.8278 |      0.9083 |      0.2365 |                    0.8592 |           0.9311 |           0.1088 |         2.02 |
| Logistic Regression  |               0.7943 |      0.8708 |      0.2055 |                    0.8226 |           0.9114 |           0.0996 |         1.05 |

### Interpretation

- Random Forest achieved the highest CV observation-level Top-1 accuracy and CV Macro-F1 in this experiment.
- HistGradientBoosting achieved the highest student observation-level Top-1 accuracy and the lowest student log loss.
- Random Forest and HistGradientBoosting had very similar student-test results.
- KNN had the highest student inference time among these models in the supplied comparison, while Logistic Regression had the lowest.
- The pipeline selects the model using CV observation-level Top-1 accuracy, not student-test accuracy. Therefore, Random Forest is the selected baseline model for this particular comparison.

### Bayesian optimisation results

The pipeline writes the detailed tuned-versus-untuned comparison to:

```text
results/tuned_vs_untuned_student_test.csv
```

The selected model, best parameters, best search score and trial count are saved in:

```text
results/bayesian_best_config.json
```

Add the final tuned model's student-test Top-1, Macro-F1 and log loss here after checking those files. Do not treat the baseline table above as the tuned-model result.

### HMM results

The HMM stage writes the classifier-only versus classifier-plus-HMM comparison under `hmm_results/`. The final evaluator run writes final evaluation files using the `final_` prefix.

Add the actual student and final evaluator results here after checking the generated HMM summary CSV files. In particular, report pointwise accuracy, sequence similarity, route continuity and route breaks for both classifier-only and classifier-plus-HMM predictions.

---

## Evaluation metrics

| Metric                           | What it measures                                                                                                  |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Candidate accuracy               | Whether the classifier correctly labels individual candidate rows                                                 |
| Observation-level Top-1 accuracy | Whether the highest-scored candidate for each GPS observation is the reference arc                                |
| Top-1 accuracy on candidate hits | Top-1 accuracy for observations where the reference arc was present in the generated candidates                   |
| Candidate miss rate              | Fraction of observations for which the reference arc was not present in the candidate set                         |
| Macro-F1                         | Average F1 score across classes, treating classes equally                                                         |
| Log loss                         | Quality of the predicted probabilities; lower is better                                                           |
| Sequence similarity              | Similarity between the predicted collapsed arc sequence and reference sequence, based on normalised edit distance |
| Route continuity                 | Fraction of consecutive predicted route-arc pairs that are connected in the directed road graph                   |
| Route breaks                     | Number of disconnected consecutive arc pairs in the predicted route                                               |

A low candidate miss rate is important: if the correct reference arc is absent from the candidate set, the classifier cannot select it.

---

## Generated outputs

### `processed/`

Contains cached candidate tables and candidate-generation diagnostics, including:

- `train_candidates.parquet`
- `student_test_candidates.parquet`
- Per-trip candidate diagnostic CSV files

### `results/`

Contains model and experiment outputs, including:

- `dataset_inventory.csv` and `dataset_summary.csv` — dataset inspection
- `model_comparison.csv` — five-model comparison
- `*_cv_folds.csv` — fold-level validation scores for each classifier
- `student_confusion_matrices.csv` — student-test confusion matrices
- `experiment_log.json` — experiment configuration and timing information
- `MLP_loss_curve.csv` — MLP training/validation loss history
- `bayesian_trials.csv` — Bayesian optimisation trial results
- `bayesian_best_config.json` — selected model and best search configuration
- `tuned_vs_untuned_student_test.csv` — student-test comparison before and after tuning
- `best_tuned_model.joblib` — tuned classifier used for inference
- `feature_list.json` and `final_model_config.json` — inference feature and model configuration
- `plots/` — generated comparison plots

### `hmm_results/`

Contains:

- `hmm_config.json` — selected HMM decoder parameters and tuning metadata
- Student classifier-only versus classifier-plus-HMM comparison outputs
- Decoder tuning outputs and plots, where generated
- Final evaluator output files with the `final_` prefix

### `eval_results/`

Used in the project workspace for final evaluation outputs or collected evaluation artefacts. Check the generated files in this directory when preparing the final submission.

---

## Limitations and important notes

- **Reference alignment is approximate:** The supplied `.route` sequence is aligned to GPS observations using dynamic programming. It is treated as reference information, not independently verified ground truth.
- **Candidate generation limits the classifier:** If the correct road arc is missing from the top-K candidates, the classifier cannot predict it. Candidate miss rate should therefore be reported alongside classifier accuracy.
- **Adjacent GPS samples are correlated:** Training observations are thinned to reduce the effect of near-duplicate 1 Hz observations.
- **Student-test decoder tuning can make student metrics optimistic:** If `--tune-on student` is used, the HMM parameters are tuned on student trips. State this clearly when reporting student HMM results.
- **Evaluator data must remain isolated:** The evaluator partition must not influence model selection or parameter tuning.
- **Arc IDs are trip-specific:** An arc ID only identifies an arc within its own trip's road network.
- **API reference data stays server-side:** The backend uses reference routes to calculate evaluation metrics after prediction; it does not send those labels to the frontend.

---

## Reproducibility

- The processing pipeline uses random seed `42`.
- The experiment log records the feature list, thinning setting, model configurations and timings.
- The Bayesian optimisation trial count is controlled by `--bayes-iter`.
- The HMM configuration records its tuning mode, number of tuning trips, objective, grid and random seed.
- Keep the model file, feature configuration and HMM configuration together when reproducing API inference.
- Record the exact commands and settings used for the final run, especially `--bayes-iter`, `--thin`, `--top-k` and `--tune-on`.

---

## Acknowledgements

Developed as a team project for the PES University Machine Learning course (UE24CS352A).
