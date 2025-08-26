from fastmcp import FastMCP
from starlette.responses import PlainTextResponse, JSONResponse
import os, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

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
# Move implementation into plain functions and keep @mcp.tool wrappers thin so other code
# (HTTP wrappers) can call the implementations directly.

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
    dataset = load_breast_cancer()
    X_tab = dataset.data
    Y_tab = dataset.target
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
# Training Functions
# ==============================
@mcp.tool
def train_text_branch(model, X_train, Y_train, X_val, Y_val, num_img_classes):
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
    return {"report": classification_report(Y_val, y_pred_txt)}


@mcp.tool
def train_image_branch(model, train_img_gen, val_img_gen, num_tab_features):
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


# ==============================
# Full Pipeline
# ==============================

def _train_and_evaluate_full_pipeline_impl():
    img_data = _load_image_data_impl()
    tab_data = _load_tabular_data_impl()
    model = _build_multitask_model_impl(img_data["num_classes"], tab_data["num_features"])
    txt_result = train_text_branch(model, tab_data["X_train"], tab_data["Y_train"], tab_data["X_val"], tab_data["Y_val"], img_data["num_classes"])
    img_result = train_image_branch(model, img_data["train_img_gen"], img_data["val_img_gen"], tab_data["num_features"])
    return {"textual_report": txt_result["report"], "image_report": img_result["report"]}

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


# Note: training-specific endpoints like train_text_branch and train_image_branch expect complex
# Python objects (models/generators). They are not wrapped here. If you need them over HTTP we
# should design a serialized protocol or a job system. For now the frontend can use the
# above wrapper endpoints for basic interactions.

if __name__ == "__main__":
    mcp.run(transport="http")  # exposes /health and all @mcp.tool automatically
