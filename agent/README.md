# Breast-classification-MCP

An end-to-end example that trains a multi-task Keras model for breast ultrasound image classification and exposes it through an HTTP + MCP (Model Context Protocol) server with a lightweight React frontend for interaction.

The project demonstrates:
- Multi-input, multi-output Keras model combining an image branch (DenseNet121) and a tabular branch (sklearn breast_cancer dataset).
- Simple HTTP routes using FastMCP’s custom routes for training, evaluation, confusion matrices, and inference.
- A small React UI to kick off training, visualize confusion matrices, and classify uploaded images.

## Repository layout

```
agent/
  server.py                # FastMCP server: HTTP routes, training, prediction
  api.py                   # (Optional) FastAPI client for MCP tools (not required to run)
  Dataset_BUSI_with_GT/    # Ultrasound dataset with benign/malignant/normal folders
  frontend/                # Vite + React frontend
    index.html
    package.json
    src/App.jsx
  test_images/             # Sample images for quick tests
  saved_models/            # Will be created at runtime (saved model + labels)
  README.md                # This file
  requirements.txt         # Python backend dependencies
```

## Architecture

### Model (Keras / TensorFlow)

The model is multi-input, multi-output:

- Image branch
  - Input: `image_input` with shape (256, 256, 3)
  - Backbone: DenseNet121 pretrained on ImageNet (`include_top=False`), frozen
  - Head: GlobalAveragePooling2D → Dense(512, relu) → Dropout(0.5) → Dense(num_img_classes, softmax) as `img_output`

- Tabular branch
  - Input: `tabular_input` with shape (num_tab_features,)
  - Head: Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(1, sigmoid) as `txt_output`

Training strategy (two-stage):
- Text branch: Only the tabular head trains; the image head loss weight is 0 during this phase.
- Image branch: Only the image head trains; the tabular head loss weight is 0 and dummy zeros are fed to `tabular_input`.

Saving:
- After the full pipeline training, the server saves the model to `saved_models/`:
  - Timestamped: `multitask_model_YYYYMMDD_HHMMSS.keras`
  - Stable latest symlink/file: `multitask_model_latest.keras`
- Label sidecar JSON files are also saved:
  - `labels_YYYYMMDD_HHMMSS.json`
  - `labels_latest.json`
  - Includes `{ "labels": [..], "class_indices": {name: index} }` for consistent inference mapping.

### MCP server (FastMCP)

`server.py` uses [FastMCP] to:

- Register tool implementations (callable both via MCP and via thin HTTP wrappers).
- Expose HTTP endpoints using `@mcp.custom_route` with permissive CORS for `http://localhost:5173`.
- Default HTTP base assumed by the frontend: `http://localhost:8000`.

Key routes:
- GET `/health` — health check
- POST `/load_image_data` — returns dataset stats (not the generators)
- POST `/load_tabular_data` — returns train/val splits for the sklearn dataset (as JSON lists)
- POST `/build_multitask_model` — returns model metadata (no weights)
- POST `/train_text_branch` — trains only the tabular branch (fresh in-process model)
- POST `/train_image_branch` — trains only the image branch (fresh in-process model)
- POST `/train_and_evaluate_full_pipeline` — runs the full pipeline and saves model + labels
- POST `/confusion_matrix` — computes a confusion matrix (tabular or image branch)
- POST `/confusion_matrix_image` — returns PNG of the confusion matrix
- POST `/predict_image` — classify an uploaded image using a previously saved model

Inference specifics:
- `/predict_image` accepts multipart/form-data with `image` and `model_path`, or JSON with `image_base64`.
- It loads the Keras model once and caches it, and looks for labels sidecars automatically.
- Supports multi-input models; non-image inputs are filled with zeros for inference.

### Frontend (Vite + React)

UI features (`frontend/src/App.jsx`):
- Health, load data, build model, train branch buttons for quick actions.
- Confusion matrix: request JSON or PNG preview for chosen modality.
- Image classification panel:
  - Choose a saved model path (defaults to `saved_models/multitask_model_latest.keras`).
  - Upload an image for classification.
  - Displays predicted label, index, and per-class probabilities.
  - If backend returns generic labels like `class_0/1/2`, the UI maps them to explicit names in the order `benign`, `malignant`, `normal`.

## Setup

### Prerequisites
- Python 3.9+ recommended
- Node 18+ recommended
- Disk space sufficient for the dataset in `Dataset_BUSI_with_GT/`

### Python backend

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the server:

```bash
python server.py
```

The HTTP server will listen on `http://localhost:8000` by default. CORS is configured for `http://localhost:5173` (Vite).

Optional environment variables:
- `MODEL_PATH` — default model path used by `/predict_image` if `model_path` isn’t provided in the request.

### Frontend

Install and start the Vite dev server:

```bash
cd frontend
npm install
npm run dev
```

Open the printed dev URL (typically `http://localhost:5173`). The app assumes the API base `http://localhost:8000` unless you inject `window.__MCP_API_BASE__` in `index.html`.

## Data expectations

- Image dataset directory: `Dataset_BUSI_with_GT/` containing subfolders `benign/`, `malignant/`, `normal/` with images (PNG).
- The loader scans recursively and uses the folder name as the class label.
- Tabular data comes from `sklearn.datasets.load_breast_cancer` and is standardized.

## API overview

- GET `/health`: returns `OK`.
- POST `/load_image_data`: returns dataset stats (class_indices, counts).
- POST `/load_tabular_data`: returns splits and num_features.
- POST `/build_multitask_model`: returns model metadata (no weights created).
- POST `/train_text_branch`: trains tabular head only (in-process, ephemeral model).
- POST `/train_image_branch`: trains image head only (in-process, ephemeral model).
- POST `/train_and_evaluate_full_pipeline`: trains both branches, returns reports and saves model + labels.
- POST `/confusion_matrix`: computes CM for requested modality.
- POST `/confusion_matrix_image`: returns PNG CM for requested modality.
- POST `/predict_image`: classify uploaded image using saved model.

All responses include CORS headers for `http://localhost:5173`.

## Notes and tips

- CPU mode is enforced via `CUDA_VISIBLE_DEVICES=-1` in `server.py` for portability. Remove/change as needed for GPU.
- The image loader currently includes any files under class folders; ensure only the intended images are present or adapt the glob/filtering as needed.
- Sidecar labels: place a `labels.json` in the same folder or `model_path.labels.json` to control the class names shown by the frontend.

## Troubleshooting

- If multipart form parsing fails: ensure `python-multipart` is installed (included in requirements.txt).
- If the frontend can’t reach the backend: check that the server is listening on `http://localhost:8000` and CORS allows `http://localhost:5173`.
- TensorFlow issues: verify the Python version and that `tensorflow` is installed for your platform (consider `tensorflow-cpu` if GPU is not desired).

## References

- Breast Cancer Classification (PyTorch) by vishrut-b: https://github.com/vishrut-b/ML-Project-with-PyTorch-Breast-Cancer-Classification/tree/main
- Breast Cancer Image Classification with DenseNet121 by m3mentomor1: https://github.com/m3mentomor1/Breast-Cancer-Image-Classification-with-DenseNet121
