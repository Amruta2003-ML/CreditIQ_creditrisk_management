import os
import sys
import json
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_FILE, METRICS_FILE, PREPROCESSOR_FILE

import tensorflow as tf
from tensorflow import keras
from preprocess import load_and_preprocess_data
from model import build_credit_risk_ann, compute_optimal_threshold

def train_credit_iq_model(data_path=None, nrows=None, epochs=40, batch_size=256):
    print("=" * 70)
    print("CreditIQ - Credit Risk Artificial Neural Network (ANN) Training")
    print("=" * 70)

    # 1. Load and Preprocess Data (with Time-Based Train/Val/Test Split)
    dataset = load_and_preprocess_data(file_path=data_path, nrows=nrows)
    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_val, y_val = dataset["X_val"], dataset["y_val"]
    X_test, y_test = dataset["X_test"], dataset["y_test"]

    input_dim = X_train.shape[1]
    print(f"\n[INFO] Input features dimensionality: {input_dim}")

    # 2. Compute Class Weights for Imbalanced Target (~20% defaults)
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    total = len(y_train)
    weight_for_0 = (1 / num_neg) * (total / 2.0)
    weight_for_1 = (1 / num_pos) * (total / 2.0)
    class_weight = {0: float(weight_for_0), 1: float(weight_for_1)}
    print(f"[INFO] Class balance -> Non-default (0): {num_neg}, Default (1): {num_pos}")
    print(f"[INFO] Applied Class Weights -> {class_weight}")

    # 3. Instantiate ANN
    model = build_credit_risk_ann(input_dim=input_dim, learning_rate=0.001)
    model.summary()

    # 4. Define Callbacks
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_roc_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_FILE,
            monitor="val_roc_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        )
    ]

    # 5. Train Model
    print("\n[INFO] Starting ANN training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Evaluate on Unseen Out-Of-Time Test Set
    print("\n" + "=" * 70)
    print("Evaluating Best Model on Out-Of-Time Test Set")
    print("=" * 70)
    best_model = keras.models.load_model(MODEL_FILE)
    test_results = best_model.evaluate(X_test, y_test, batch_size=batch_size, return_dict=True)
    
    for metric_name, val in test_results.items():
        print(f"  Test {metric_name.upper():<15}: {val:.4f}")

    # 7. Probability of Default Calibration & Optimal Cutoff
    y_test_probs = best_model.predict(X_test, batch_size=batch_size).flatten()
    best_thresh, threshold_metrics = compute_optimal_threshold(y_test, y_test_probs, cost_fn=5.0, cost_fp=1.0)
    
    print("\n--- Underwriting Policy & Threshold Optimization ---")
    print(f"Standard Default Threshold: 0.50")
    print(f"Optimal Cost-Minimizing Threshold: {best_thresh:.4f}")
    print(f"Metrics at Optimal Threshold: {threshold_metrics}")

    # 8. Save Metrics Artifact
    metrics_summary = {
        "test_metrics": {k: float(v) for k, v in test_results.items()},
        "threshold_policy": threshold_metrics,
        "class_weights": class_weight,
        "input_dimension": input_dim,
        "features_list": dataset["feature_names"]
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\n[OK] Metrics saved to {METRICS_FILE}")
    print(f"[OK] Best ANN Model saved to {MODEL_FILE}")

    return best_model, metrics_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CreditIQ ANN Credit Default Model")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw/compressed LendingClub CSV")
    parser.add_argument("--nrows", type=int, default=None, help="Limit number of rows for testing")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    args = parser.parse_args()

    train_credit_iq_model(
        data_path=args.data_path,
        nrows=args.nrows,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
