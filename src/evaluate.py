import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json, os

def evaluate_model(model_path, data_loader):
    model = tf.keras.models.load_model(model_path)
    x_test, y_test = data_loader.load_test_data()
    results = model.evaluate(x_test, y_test, verbose=1)
    print("Loss & Accuracy:", results)

    # Dự đoán
    y_pred = model.predict(x_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred_cls = np.argmax(y_pred, axis=1)

    print(classification_report(y_true, y_pred_cls))
    return y_true, y_pred_cls

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def plot_history(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    plt.figure()
    plt.plot(history['accuracy'], label='train_acc')
    plt.plot(history['val_accuracy'], label='val_acc')
    plt.legend(); plt.title("Training Accuracy")
    plt.savefig("reports/loss_acc_plot.png")

if __name__ == "__main__":
    from data_load import DataLoader
    data_loader = DataLoader()
    y_true, y_pred = evaluate_model("model/fruit_cnn.h5", data_loader)
    labels = data_loader.class_names
    plot_confusion_matrix(y_true, y_pred, labels, "reports/confusion_type.png")
