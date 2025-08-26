import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_multitask_model(num_img_classes, num_tab_features):
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

def compile_for_image(model):
    for layer in model.layers:
        if "txt_" in layer.name or layer.name=="txt_output":
            layer.trainable = False
        if "img_" in layer.name or layer.name=="img_output":
            layer.trainable = True
    model.compile(
        optimizer="adam",
        loss={
            "img_output": "categorical_crossentropy",
            "txt_output": "binary_crossentropy"
        },
        loss_weights={
            "img_output": 1.0,
            "txt_output": 0.0
        },
        metrics={"img_output": "accuracy"}
    )
    return model

def compile_for_text(model):
    for layer in model.layers:
        if "img_" in layer.name or layer.name=="img_output":
            layer.trainable = False
        if "txt_" in layer.name or layer.name=="txt_output":
            layer.trainable = True
    model.compile(
        optimizer="adam",
        loss={
            "img_output": "categorical_crossentropy",
            "txt_output": "binary_crossentropy"
        },
        loss_weights={
            "img_output": 0.0,
            "txt_output": 1.0
        },
        metrics={"txt_output": "accuracy"}
    )
    return model
