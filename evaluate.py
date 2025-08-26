import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_textual(model, X_tab_val, Y_tab_val, num_img_classes):
    y_pred_txt = (model.predict({"tabular_input": X_tab_val, "image_input": np.zeros((len(X_tab_val),256,256,3))})[1] > 0.5).astype(int)
    print(classification_report(Y_tab_val, y_pred_txt))
    cm_txt = confusion_matrix(Y_tab_val, y_pred_txt)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_txt, annot=True, fmt="d", cmap="Blues", xticklabels=["No Cancer", "Cancer"], yticklabels=["No Cancer", "Cancer"])
    plt.title("Confusion Matrix - Textual Branch")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def evaluate_image(model, val_img_gen, batch_size, num_tab_features, class_indices):
    val_img_gen.reset()
    X_imgs, Y_imgs = [], []
    for batch in val_img_gen:
        X_imgs.append(batch[0])
        Y_imgs.append(batch[1])
        if len(X_imgs) * batch_size >= val_img_gen.n:
            break
    X_imgs = np.concatenate(X_imgs, axis=0)
    Y_imgs = np.concatenate(Y_imgs, axis=0)
    y_pred_img = np.argmax(model.predict({"image_input": X_imgs, "tabular_input": np.zeros((len(X_imgs), num_tab_features))})[0], axis=1)
    y_true_img = np.argmax(Y_imgs, axis=1)
    print(classification_report(y_true_img, y_pred_img, target_names=list(class_indices.keys())))
    cm = confusion_matrix(y_true_img, y_pred_img)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(class_indices.keys()), yticklabels=list(class_indices.keys()))
    plt.title("Confusion Matrix - Image Branch")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
