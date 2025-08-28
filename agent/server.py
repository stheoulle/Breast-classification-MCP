from fastmcp import FastMCP
from starlette.responses import PlainTextResponse, JSONResponse, Response
import logging
import os, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for servers
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import shutil
import json


mcp = FastMCP("agent-llm-server")


# Provide a lightweight CORS solution: set CORS headers on health and handle OPTIONS preflight for all paths.
# This avoids touching FastMCP internals (some FastMCP versions don't expose the underlying ASGI app).

# ==============================
# Health Check
# ==============================
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:5173",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
    }
    return PlainTextResponse("OK", headers=headers)

# Handle preflight (OPTIONS) for any path so browser preflight requests succeed.
@mcp.custom_route("/{path:path}", methods=["OPTIONS"])
async def cors_preflight(request, path: str = ""):
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:5173",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
    }
    return PlainTextResponse("", headers=headers)


# ==============================
# Data Preparation
# ==============================

def _load_image_data_impl(train_data="Dataset_BUSI_with_GT", image_size=(256, 256), batch_size=16):
    train_files = [i for i in glob.glob(train_data + "/*/*")]
    labels = [os.path.dirname(i).split("/")[-1] for i in train_files]
    training_data = pd.DataFrame({"Path": train_files, "Label": labels})
    train_df, val_df = train_test_split(training_data, train_size=0.8, shuffle=True, random_state=123)
    datagen = ImageDataGenerator(rescale=1.0/255)
    train_img_gen = datagen.flow_from_dataframe(train_df, x_col="Path", y_col="Label",
                                                target_size=image_size, class_mode="categorical",
                                                shuffle=True, batch_size=batch_size)
    val_img_gen = datagen.flow_from_dataframe(val_df, x_col="Path", y_col="Label",
                                              target_size=image_size, class_mode="categorical",
                                              shuffle=False, batch_size=batch_size)
    return {
        "num_classes": len(train_img_gen.class_indices),
        "num_train_samples": train_img_gen.n,
        "num_val_samples": val_img_gen.n,
        "class_indices": train_img_gen.class_indices,
        "train_img_gen": train_img_gen,
        "val_img_gen": val_img_gen
    }

@mcp.tool
def load_image_data(train_data="Dataset_BUSI_with_GT", image_size=(256, 256), batch_size=16):
    # tool wrapper that delegates to the plain implementation
    return _load_image_data_impl(train_data=train_data, image_size=image_size, batch_size=batch_size)


def _load_tabular_data_impl():
    # Use return_X_y=True to avoid Bunch typing issues with some type checkers
    X_tab, Y_tab = load_breast_cancer(return_X_y=True)
    X_tab = np.asarray(X_tab)
    Y_tab = np.asarray(Y_tab)
    scaler = StandardScaler()
    X_tab = scaler.fit_transform(X_tab)
    X_tab_train, X_tab_val, Y_tab_train, Y_tab_val = train_test_split(X_tab, Y_tab, test_size=0.2, random_state=42)
    return {
        "X_train": X_tab_train.tolist(),
        "X_val": X_tab_val.tolist(),
        "Y_train": Y_tab_train.tolist(),
        "Y_val": Y_tab_val.tolist(),
        "num_features": X_tab.shape[1]
    }

@mcp.tool
def load_tabular_data():
    return _load_tabular_data_impl()


# ==============================
# Model Construction
# ==============================

def _build_multitask_model_impl(num_img_classes: int, num_tab_features: int):
    image_input = Input(shape=(256, 256, 3), name='image_input')
    base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=image_input)
    for layer in base_model.layers:
        layer.trainable = False
    x_img = GlobalAveragePooling2D()(base_model.output)
    x_img = Dense(512, activation='relu', name="img_dense1")(x_img)
    x_img = Dropout(0.5)(x_img)
    img_output = Dense(num_img_classes, activation='softmax', name='img_output')(x_img)

    tab_input = Input(shape=(num_tab_features,), name='tabular_input')
    x_tab = Dense(64, activation='relu', name="txt_dense1")(tab_input)
    x_tab = Dropout(0.3)(x_tab)
    x_tab = Dense(32, activation='relu', name="txt_dense2")(x_tab)
    txt_output = Dense(1, activation='sigmoid', name='txt_output')(x_tab)

    model = Model(inputs=[image_input, tab_input], outputs=[img_output, txt_output])

    return {
        "model_name": model.name,
        "inputs": [inp.name for inp in model.inputs],
        "outputs": [out.name for out in model.outputs],
        "num_layers": len(model.layers),
        "trainable_layers": sum([1 for layer in model.layers if layer.trainable]),
    }

@mcp.tool
def build_multitask_model(num_img_classes: int, num_tab_features: int):
    return _build_multitask_model_impl(num_img_classes=num_img_classes, num_tab_features=num_tab_features)


# ==============================
# Training Functions (plain implementations + tool wrappers)
# ==============================

def _train_text_branch_impl(model, X_train, Y_train, X_val, Y_val, num_img_classes):
    # Ensure inputs are numpy arrays with correct shapes/dtypes
    X_train = np.array(X_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32).reshape(-1, 1)
    Y_val = np.array(Y_val, dtype=np.float32).reshape(-1, 1)

    for layer in model.layers:
        layer.trainable = "txt_" in layer.name or layer.name == "txt_output"
    model.compile(optimizer="adam",
                  loss={"img_output": "binary_crossentropy", "txt_output": "binary_crossentropy"},
                  loss_weights={"img_output": 0.0, "txt_output": 1.0},
                  metrics={"txt_output": "accuracy"})
    history = model.fit(
        {"tabular_input": X_train, "image_input": np.zeros((len(X_train), 256, 256, 3))},
        {"txt_output": Y_train, "img_output": np.zeros((len(X_train), num_img_classes))},
        validation_data=(
            {"tabular_input": X_val, "image_input": np.zeros((len(X_val), 256, 256, 3))},
            {"txt_output": Y_val, "img_output": np.zeros((len(X_val), num_img_classes))}
        ),
        epochs=5, batch_size=8
    )
    y_pred_txt = (model.predict({"tabular_input": X_val, "image_input": np.zeros((len(X_val), 256, 256, 3))})[1] > 0.5).astype(int)
    # flatten labels for sklearn
    return {"report": classification_report(Y_val.flatten(), y_pred_txt.flatten())}

@mcp.tool
def train_text_branch(model, X_train, Y_train, X_val, Y_val, num_img_classes):
    return _train_text_branch_impl(model, X_train, Y_train, X_val, Y_val, num_img_classes)


def _train_image_branch_impl(model, train_img_gen, val_img_gen, num_tab_features):
    for layer in model.layers:
        layer.trainable = "img_" in layer.name or layer.name == "img_output"
    model.compile(optimizer="adam",
                  loss={"img_output": "categorical_crossentropy", "txt_output": "binary_crossentropy"},
                  loss_weights={"img_output": 1.0, "txt_output": 0.0},
                  metrics={"img_output": "accuracy"})

    def generator_with_dummy_tab(img_gen):
        while True:
            X_img, Y_img = next(img_gen)
            yield {"image_input": X_img, "tabular_input": np.zeros((X_img.shape[0], num_tab_features))}, {"img_output": Y_img, "txt_output": np.zeros((X_img.shape[0], 1))}

    model.fit(generator_with_dummy_tab(train_img_gen),
              validation_data=generator_with_dummy_tab(val_img_gen),
              steps_per_epoch=len(train_img_gen),
              validation_steps=len(val_img_gen),
              epochs=10)

    # Evaluate
    X_imgs, Y_imgs = [], []
    for batch in val_img_gen:
        X_imgs.append(batch[0])
        Y_imgs.append(batch[1])
        if len(X_imgs) * val_img_gen.batch_size >= val_img_gen.n:
            break
    X_imgs = np.concatenate(X_imgs)
    Y_imgs = np.concatenate(Y_imgs)

    y_pred_img = np.argmax(model.predict({"image_input": X_imgs, "tabular_input": np.zeros((len(X_imgs), num_tab_features))})[0], axis=1)
    y_true_img = np.argmax(Y_imgs, axis=1)
    return {"report": classification_report(y_true_img, y_pred_img, target_names=list(train_img_gen.class_indices.keys()))}

@mcp.tool
def train_image_branch(model, train_img_gen, val_img_gen, num_tab_features):
    return _train_image_branch_impl(model, train_img_gen, val_img_gen, num_tab_features)

# ==============================
# Full Pipeline
# ==============================

def _train_and_evaluate_full_pipeline_impl():
    # Load data using the local implementations
    img_data = _load_image_data_impl()
    tab_data = _load_tabular_data_impl()

    # Build a real Keras model object for training (the _build_* impl returns metadata only)
    model = _build_multitask_model_impl()
    _build_multitask_model_impl(num_img_classes=img_data["num_classes"], num_tab_features=tab_data["num_features"])
    # model = _create_multitask_model_object(
    #     num_img_classes=img_data["num_classes"],
    #     num_tab_features=tab_data["num_features"],
    # )

    # Train text and image branches using the in-process implementations
    txt_result = _train_text_branch_impl(
        model,
        tab_data["X_train"],
        tab_data["Y_train"],
        tab_data["X_val"],
        tab_data["Y_val"],
        img_data["num_classes"],
    )

    img_result = _train_image_branch_impl(
        model,
        img_data["train_img_gen"],
        img_data["val_img_gen"],
        tab_data["num_features"],
    )

    # After training both branches, save the model and labels for later inference
    try:
        save_dir = os.path.join(os.getcwd(), "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path_ts = os.path.join(save_dir, f"multitask_model_{timestamp}.keras")
        model_path_latest = os.path.join(save_dir, "multitask_model_latest.keras")

        # Save once to timestamped path, then copy to latest for a stable reference
        model.save(model_path_ts)
        try:
            shutil.copyfile(model_path_ts, model_path_latest)
        except Exception:
            # If copy fails (e.g., different FS), save directly to latest as fallback
            model.save(model_path_latest)

        # Save labels sidecar
        class_indices = img_data["class_indices"]  # name -> index
        labels = [name for name, _idx in sorted(class_indices.items(), key=lambda kv: kv[1])]
        labels_ts = os.path.join(save_dir, f"labels_{timestamp}.json")
        labels_latest = os.path.join(save_dir, "labels_latest.json")
        labels_payload = {"labels": labels, "class_indices": class_indices}
        with open(labels_ts, "w") as f:
            json.dump(labels_payload, f)
        try:
            shutil.copyfile(labels_ts, labels_latest)
        except Exception:
            with open(labels_latest, "w") as f:
                json.dump(labels_payload, f)
    except Exception as e:
        # Non-fatal: training succeeded; surface save error in response
        model_path_ts = model_path_latest = labels_ts = labels_latest = None
        save_error = str(e)
    else:
        save_error = None

    return {
        "textual_report": txt_result["report"],
        "image_report": img_result["report"],
        "model_path_latest": model_path_latest,
        "model_path_timestamp": model_path_ts,
        "labels_path_latest": labels_latest,
        "labels_path_timestamp": labels_ts,
        "labels": labels if 'labels' in locals() else None,
        "class_indices": class_indices if 'class_indices' in locals() else None,
        "save_error": save_error,
    }

@mcp.tool
def train_and_evaluate_full_pipeline():
    return _train_and_evaluate_full_pipeline_impl()


# Provide simple HTTP wrappers for key tools so the frontend can POST to them directly.
# These wrappers add CORS headers and return only JSON-serializable data.
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "http://localhost:5173",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Credentials": "true",
}

@mcp.custom_route("/load_image_data", methods=["POST"])
async def load_image_data_route(request):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    train_data = body.get("train_data", "Dataset_BUSI_with_GT")
    image_size = tuple(body.get("image_size", (256, 256)))
    batch_size = int(body.get("batch_size", 16))
    result = _load_image_data_impl(train_data=train_data, image_size=image_size, batch_size=batch_size)
    # Remove non-serializable generators before returning
    safe = {k: v for k, v in result.items() if k not in ("train_img_gen", "val_img_gen")}
    return JSONResponse(safe, headers=CORS_HEADERS)


@mcp.custom_route("/load_tabular_data", methods=["POST"])
async def load_tabular_data_route(request):
    result = _load_tabular_data_impl()
    return JSONResponse(result, headers=CORS_HEADERS)


@mcp.custom_route("/build_multitask_model", methods=["POST"])
async def build_multitask_model_route(request):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    num_img_classes = int(body.get("num_img_classes", 2))
    num_tab_features = int(body.get("num_tab_features", 30))
    result = _build_multitask_model_impl(num_img_classes=num_img_classes, num_tab_features=num_tab_features)
    return JSONResponse(result, headers=CORS_HEADERS)


@mcp.custom_route("/train_and_evaluate_full_pipeline", methods=["POST"])
async def train_and_evaluate_full_pipeline_route(request):
    # This runs the full pipeline and can take a while; no arguments expected.
    result = _train_and_evaluate_full_pipeline_impl()
    return JSONResponse(result, headers=CORS_HEADERS)


# # Helper to create an actual Keras Model object (used by training wrappers)
def _create_multitask_model_object(num_img_classes: int, num_tab_features: int):
    image_input = Input(shape=(256, 256, 3), name='image_input')
    base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=image_input)
    for layer in base_model.layers:
        layer.trainable = False
    x_img = GlobalAveragePooling2D()(base_model.output)
    x_img = Dense(512, activation='relu', name="img_dense1")(x_img)
    x_img = Dropout(0.5)(x_img)
    img_output = Dense(num_img_classes, activation='softmax', name='img_output')(x_img)

    tab_input = Input(shape=(num_tab_features,), name='tabular_input')
    x_tab = Dense(64, activation='relu', name="txt_dense1")(tab_input)
    x_tab = Dropout(0.3)(x_tab)
    x_tab = Dense(32, activation='relu', name="txt_dense2")(x_tab)
    txt_output = Dense(1, activation='sigmoid', name='txt_output')(x_tab)

    model = Model(inputs=[image_input, tab_input], outputs=[img_output, txt_output])
    return model


# HTTP wrapper to train only the text branch using in-process tabular data and a fresh model
@mcp.custom_route("/train_text_branch", methods=["POST"])
async def train_text_branch_route(request):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    num_img_classes = int(body.get("num_img_classes", 2))
    tab = _load_tabular_data_impl()
    #model_obj = _build_multitask_model_impl(num_img_classes=num_img_classes, num_tab_features=tab["num_features"])
    model_obj = _create_multitask_model_object(num_img_classes=num_img_classes, num_tab_features=tab["num_features"])    
    try:
        result = _train_text_branch_impl(model_obj, tab["X_train"], tab["Y_train"], tab["X_val"], tab["Y_val"], num_img_classes)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500, headers=CORS_HEADERS)
    return JSONResponse(result, headers=CORS_HEADERS)


# HTTP wrapper to train only the image branch using in-process image generators and a fresh model
@mcp.custom_route("/train_image_branch", methods=["POST"])
async def train_image_branch_route(request):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    num_tab_features = int(body.get("num_tab_features", 30))
    img = _load_image_data_impl()
    #model_obj = _build_multitask_model_impl(num_img_classes=img["num_classes"], num_tab_features=num_tab_features)
    model_obj = _create_multitask_model_object(num_img_classes=img["num_classes"], num_tab_features=num_tab_features)
    try:
        result = _train_image_branch_impl(model_obj, img["train_img_gen"], img["val_img_gen"], num_tab_features)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500, headers=CORS_HEADERS)
    return JSONResponse(result, headers=CORS_HEADERS)

# ==============================
# Confusion Matrix Plot (PNG)
# ==============================
@mcp.custom_route("/confusion_matrix_image", methods=["POST"])

async def confusion_matrix_image_route(request):
    """Return a PNG image rendering of the confusion matrix for the chosen modality, using a saved model if available."""
    try:
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    except Exception:
        body = {}

    modality = (body.get("modality") or body.get("branch") or "tabular").lower()
    model_path = body.get("model_path") or os.getenv("MODEL_PATH")
    if not model_path:
        # Try default location
        model_path = os.path.join(os.getcwd(), "saved_models", "multitask_model_latest.keras")
    if not os.path.exists(model_path):
        return JSONResponse({"error": f"No saved model found at '{model_path}'"}, status_code=404, headers=CORS_HEADERS)

    try:
        model = _get_or_load_model(model_path)
    except Exception as e:
        return JSONResponse({"error": f"Failed to load model: {e}"}, status_code=500, headers=CORS_HEADERS)

    if modality == "tabular":
        tab = _load_tabular_data_impl()
        X_val = np.array(tab["X_val"], dtype=np.float32)
        Y_val = np.array(tab["Y_val"], dtype=np.float32).reshape(-1, 1)
        # Find tabular input name
        input_names = [getattr(t, "name", "").split(":")[0] for t in model.inputs]
        feed = {}
        for name in input_names:
            if "tabular" in name:
                feed[name] = X_val
            elif "image" in name:
                feed[name] = np.zeros((len(X_val), 256, 256, 3))
        y_pred_prob = model.predict(feed)[1]
        y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
        y_true = Y_val.reshape(-1).astype(int)
        labels = ["class_0", "class_1"]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    elif modality == "image":
        img = _load_image_data_impl()
        val_img_gen = img["val_img_gen"]
        num_tab_features = img["train_img_gen"].image_shape[0] if hasattr(img["train_img_gen"], "image_shape") else 30
        class_indices = img["class_indices"]
        labels = [name for name, _idx in sorted(class_indices.items(), key=lambda kv: kv[1])]
        # Collect validation set
        X_imgs, Y_imgs = [], []
        for batch in val_img_gen:
            X_imgs.append(batch[0])
            Y_imgs.append(batch[1])
            if len(X_imgs) * val_img_gen.batch_size >= val_img_gen.n:
                break
        X_imgs = np.concatenate(X_imgs)
        Y_imgs = np.concatenate(Y_imgs)
        # Find input names
        input_names = [getattr(t, "name", "").split(":")[0] for t in model.inputs]
        feed = {}
        for name in input_names:
            if "image" in name:
                feed[name] = X_imgs
            elif "tabular" in name:
                feed[name] = np.zeros((len(X_imgs), 30))
        y_pred_logits = model.predict(feed)[0]
        y_pred = np.argmax(y_pred_logits, axis=1)
        y_true = np.argmax(Y_imgs, axis=1)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    else:
        return JSONResponse({"error": f"Unsupported modality: {modality}"}, status_code=400, headers=CORS_HEADERS)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label', title=f'Confusion Matrix ({modality})')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    headers = {**CORS_HEADERS, "Content-Type": "image/png"}
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


# Note: training-specific endpoints like train_text_branch and train_image_branch expect complex
# Python objects (models/generators). They are not wrapped here. If you need them over HTTP we
# should design a serialized protocol or a job system. For now the frontend can use the
# above wrapper endpoints for basic interactions.


# ==============================
# Image Prediction Endpoint (no training)
# ==============================

# Simple caches so we don't reload models/labels each request
_MODEL_CACHE = {}
_LABELS_CACHE = {}

def _load_labels_for_model(model_path: str, num_classes: int):
    """Try to load labels from sidecar JSON; fallback to class_0..N-1."""
    if model_path in _LABELS_CACHE:
        return _LABELS_CACHE[model_path]
    sidecars = [f"{model_path}.labels.json", os.path.join(os.path.dirname(model_path), "labels.json")]
    labels = None
    for p in sidecars:
        try:
            if os.path.exists(p):
                with open(p, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "labels" in data and isinstance(data["labels"], list):
                        labels = data["labels"]
                        break
                    if isinstance(data, list):
                        labels = data
                        break
        except Exception:
            # ignore sidecar parse errors, fallback below
            pass
    if not labels:
        labels = [f"class_{i}" for i in range(num_classes)]
    _LABELS_CACHE[model_path] = labels
    return labels

def _preprocess_image_bytes(img_bytes: bytes, target_size=(256, 256)) -> np.ndarray:
    """Decode bytes to a 4D NHWC float32 array scaled to [0,1]."""
    img_tensor = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img_tensor = tf.image.resize(img_tensor, target_size, method="bilinear")
    img_tensor = tf.cast(img_tensor, tf.float32) / 255.0
    img_tensor = tf.expand_dims(img_tensor, axis=0)  # add batch dim
    return img_tensor.numpy()

def _get_or_load_model(model_path: str):
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{model_path}': {e}")
    _MODEL_CACHE[model_path] = model
    return model

@mcp.custom_route("/predict_image", methods=["POST"])
async def predict_image_route(request):
    """Predict the class of an uploaded image using a pre-trained model on disk.

    Accepts multipart/form-data with fields:
      - image: file upload (required)
      - model_path: string path to a saved model (optional if MODEL_PATH env is set)
    Optional fields:
      - image_size: JSON or comma-separated "W,H" to override default 256x256

    Returns JSON with predicted label, index, probabilities, and labels list.
    """
    # Parse multipart form or JSON
    content_type = request.headers.get("content-type", "")
    form = None
    body_json = None
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
    elif content_type.startswith("application/json"):
        try:
            body_json = await request.json()
        except Exception:
            body_json = {}
    else:
        # try best-effort form parse
        try:
            form = await request.form()
        except Exception:
            form = None

    # Resolve model path
    model_path = None
    if form is not None:
        model_path = (form.get("model_path") or "").strip() or None
    if not model_path and body_json:
        model_path = (body_json.get("model_path") or "").strip() or None
    if not model_path:
        # allow query param or env var
        model_path = (request.query_params.get("model_path") or os.getenv("MODEL_PATH") or "").strip() or None
    if not model_path:
        return JSONResponse({"error": "model_path is required (or set MODEL_PATH env var)"}, status_code=400, headers=CORS_HEADERS)
    if not os.path.exists(model_path):
        return JSONResponse({"error": f"Model not found at '{model_path}'"}, status_code=400, headers=CORS_HEADERS)

    # Get image bytes
    img_bytes = None
    if form is not None:
        file_obj = form.get("image") or form.get("file")
        if file_obj is not None:
            img_bytes = await file_obj.read() if hasattr(file_obj, "read") else file_obj
    if img_bytes is None and body_json:
        # Support base64 in JSON as a fallback
        import base64
        b64 = body_json.get("image_base64")
        if isinstance(b64, str) and b64:
            try:
                # strip data URL prefix if present
                if "," in b64 and b64.strip().startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                img_bytes = base64.b64decode(b64)
            except Exception as e:
                return JSONResponse({"error": f"Failed to decode base64 image: {e}"}, status_code=400, headers=CORS_HEADERS)
    if img_bytes is None:
        return JSONResponse({"error": "No image provided. Send as multipart 'image' or JSON 'image_base64'."}, status_code=400, headers=CORS_HEADERS)

    # Optional custom image size
    target_size = (256, 256)
    size_raw = None
    if form is not None:
        size_raw = form.get("image_size")
    if not size_raw and body_json:
        size_raw = body_json.get("image_size")
    if size_raw:
        try:
            if isinstance(size_raw, (list, tuple)) and len(size_raw) == 2:
                target_size = (int(size_raw[0]), int(size_raw[1]))
            elif isinstance(size_raw, str) and "," in size_raw:
                w, h = size_raw.split(",", 1)
                target_size = (int(w), int(h))
        except Exception:
            pass  # keep default on parse error

    # Load model and preprocess image
    try:
        model = _get_or_load_model(model_path)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=500, headers=CORS_HEADERS)
    try:
        img_arr = _preprocess_image_bytes(img_bytes, target_size)
    except Exception as e:
        return JSONResponse({"error": f"Failed to preprocess image: {e}"}, status_code=400, headers=CORS_HEADERS)

    # Build inputs mapping (support single or multi-input models)
    feed = None
    try:
        model_inputs = model.inputs
        input_names = [getattr(t, "name", "").split(":")[0] for t in model_inputs]
        if len(model_inputs) == 1:
            feed = img_arr
        else:
            feed = {}
            placed = False
            for name in input_names:
                if "image" in name:
                    feed[name] = img_arr
                    placed = True
            # find tabular input shape (None, F)
            for i, t in enumerate(model_inputs):
                name = input_names[i]
                if name not in feed:
                    shape = t.shape
                    # shape like (None, F)
                    feat = int(shape[-1]) if shape[-1] is not None else 30
                    feed[name] = np.zeros((1, feat), dtype=np.float32)
            if not placed:
                # if we didn't find by name, assume first is image
                first_name = input_names[0]
                feed[first_name] = img_arr
    except Exception:
        # Fallback: try positional 2-input assumption
        try:
            feat = int(model.inputs[1].shape[-1]) if len(model.inputs) > 1 and model.inputs[1].shape[-1] is not None else 30
        except Exception:
            feat = 30
        feed = [img_arr, np.zeros((1, feat), dtype=np.float32)] if len(model.inputs) > 1 else img_arr

    # Predict
    try:
        pred = model.predict(feed)
    except Exception as e:
        return JSONResponse({"error": f"Model prediction failed: {e}"}, status_code=500, headers=CORS_HEADERS)

    # Resolve image head output and probabilities
    probs = None
    try:
        if isinstance(pred, (list, tuple)):
            # pick img head by name when possible
            try:
                out_names = [getattr(t, "name", "").split(":")[0] for t in model.outputs]
                if "img_output" in out_names:
                    probs = pred[out_names.index("img_output")]
                else:
                    probs = pred[0]
            except Exception:
                probs = pred[0]
        else:
            probs = pred
        probs = np.asarray(probs)[0]
        num_classes = int(probs.shape[-1]) if probs.ndim == 1 else int(probs.shape[1])
        if probs.ndim > 1:
            probs = probs.reshape(-1)
    except Exception as e:
        return JSONResponse({"error": f"Failed to interpret model outputs: {e}"}, status_code=500, headers=CORS_HEADERS)

    # Labels and argmax
    labels = _load_labels_for_model(model_path, num_classes=len(probs))
    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)

    return JSONResponse(
        {
            "model_path": model_path,
            "predicted_index": pred_idx,
            "predicted_label": pred_label,
            "probabilities": probs.astype(float).tolist(),
            "labels": labels,
        },
        headers=CORS_HEADERS,
    )

if __name__ == "__main__":
    mcp.run(transport="http")  # exposes /health and all @mcp.tool automatically
