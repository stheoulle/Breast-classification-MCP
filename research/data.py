import os, glob
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_image_data(train_data="Dataset_BUSI_with_GT", image_size=(256, 256), batch_size=16):
    train_files = [i for i in glob.glob(train_data + "/*/*")]
    labels = [os.path.dirname(i).split("/")[-1] for i in train_files]
    training_data = pd.DataFrame({"Path": train_files, "Label": labels})
    train_df, val_df = train_test_split(training_data, train_size=0.8, shuffle=True, random_state=123)
    datagen = ImageDataGenerator(rescale=1.0/255)
    train_img_gen = datagen.flow_from_dataframe(
        train_df, x_col="Path", y_col="Label",
        target_size=image_size, class_mode="categorical",
        shuffle=True, batch_size=batch_size
    )
    val_img_gen = datagen.flow_from_dataframe(
        val_df, x_col="Path", y_col="Label",
        target_size=image_size, class_mode="categorical",
        shuffle=False, batch_size=batch_size
    )
    num_img_classes = len(train_img_gen.class_indices)
    return train_img_gen, val_img_gen, num_img_classes, train_img_gen.class_indices

def load_tabular_data():
    dataset = load_breast_cancer()
    X_tab, Y_tab = dataset.data, dataset.target
    scaler = StandardScaler()
    X_tab = scaler.fit_transform(X_tab)
    X_tab_train, X_tab_val, Y_tab_train, Y_tab_val = train_test_split(
        X_tab, Y_tab, test_size=0.2, random_state=42
    )
    num_tab_features = X_tab.shape[1]
    return X_tab_train, X_tab_val, Y_tab_train, Y_tab_val, num_tab_features
