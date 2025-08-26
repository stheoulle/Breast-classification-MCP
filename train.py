import numpy as np
from model import compile_for_text, compile_for_image

def train_textual_branch(model, X_tab_train, Y_tab_train, X_tab_val, Y_tab_val, num_img_classes):
    model = compile_for_text(model)
    history_txt = model.fit(
        {"tabular_input": X_tab_train, "image_input": np.zeros((len(X_tab_train),256,256,3))},
        {"txt_output": Y_tab_train, "img_output": np.zeros((len(X_tab_train), num_img_classes))},
        validation_data=(
            {"tabular_input": X_tab_val, "image_input": np.zeros((len(X_tab_val),256,256,3))},
            {"txt_output": Y_tab_val, "img_output": np.zeros((len(X_tab_val), num_img_classes))}
        ),
        epochs=5,
        batch_size=8
    )
    return model, history_txt

def generator_with_dummy_tab(img_gen, num_tab_features):
    while True:
        X_img, Y_img = next(img_gen)
        X_tab_dummy = np.zeros((X_img.shape[0], num_tab_features))
        Y_txt_dummy = np.zeros((X_img.shape[0], 1))
        yield {"image_input": X_img, "tabular_input": X_tab_dummy}, {"img_output": Y_img, "txt_output": Y_txt_dummy}

def train_image_branch(model, train_img_gen, val_img_gen, num_tab_features):
    model = compile_for_image(model)
    train_gen = generator_with_dummy_tab(train_img_gen, num_tab_features)
    val_gen = generator_with_dummy_tab(val_img_gen, num_tab_features)
    history_img = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_img_gen),
        validation_steps=len(val_img_gen),
        epochs=10
    )
    return model, history_img
