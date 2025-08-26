import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Remove for GPU
from data import load_image_data, load_tabular_data
from model import build_multitask_model
from train import train_textual_branch, train_image_branch
from evaluate import evaluate_textual, evaluate_image

if __name__ == "__main__":
    # Load data
    train_img_gen, val_img_gen, num_img_classes, class_indices = load_image_data()
    X_tab_train, X_tab_val, Y_tab_train, Y_tab_val, num_tab_features = load_tabular_data()
    batch_size = 16
    # Build model
    model = build_multitask_model(num_img_classes, num_tab_features)
    # Train textual branch
    model, history_txt = train_textual_branch(model, X_tab_train, Y_tab_train, X_tab_val, Y_tab_val, num_img_classes)
    # Evaluate textual branch
    evaluate_textual(model, X_tab_val, Y_tab_val, num_img_classes)
    # Train image branch
    model, history_img = train_image_branch(model, train_img_gen, val_img_gen, num_tab_features)
    # Evaluate image branch
    evaluate_image(model, val_img_gen, batch_size, num_tab_features, class_indices)
