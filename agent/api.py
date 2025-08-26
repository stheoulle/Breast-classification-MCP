from fastapi import FastAPI
from fastmcp import Client
import asyncio

app = FastAPI()
MCP_URL = "http://127.0.0.1:808$"
client = Client(MCP_URL)

@app.get("/health")
async def health_check():
    async with client:
        return await client.call_tool("health_check")


@app.get("/load_images")
async def load_images():
    async with client:
        return await client.call_tool("load_image_data")


@app.get("/load_tabular")
async def load_tabular():
    async with client:
        return await client.call_tool("load_tabular_data")


@app.post("/build_model")
async def build_model(num_img_classes: int, num_tab_features: int):
    async with client:
        return await client.call_tool("build_multitask_model", {
            "num_img_classes": num_img_classes,
            "num_tab_features": num_tab_features
        })


@app.post("/train_text")
async def train_text(model: dict, X_train: list, Y_train: list, X_val: list, Y_val: list, num_img_classes: int):
    async with client:
        return await client.call_tool("train_text_branch", {
            "model": model,
            "X_train": X_train,
            "Y_train": Y_train,
            "X_val": X_val,
            "Y_val": Y_val,
            "num_img_classes": num_img_classes
        })


@app.post("/train_image")
async def train_image(model: dict, train_img_gen, val_img_gen, num_tab_features: int):
    async with client:
        return await client.call_tool("train_image_branch", {
            "model": model,
            "train_img_gen": train_img_gen,
            "val_img_gen": val_img_gen,
            "num_tab_features": num_tab_features
        })


@app.post("/full_pipeline")
async def full_pipeline():
    async with client:
        return await client.call_tool("train_and_evaluate_full_pipeline")
