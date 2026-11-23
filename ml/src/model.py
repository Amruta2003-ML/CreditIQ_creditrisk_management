import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_credit_risk_ann(input_dim, learning_rate=0.001, l2_reg=1e-4):
    """
    Builds a deep Artificial Neural Network (ANN) tailored for Credit Default Prediction (PD).
    
    Architecture features:
    - Dense layers with He Normal weight initialization
    - Batch Normalization for stabilizing internal covariate shift across credit cycles
    - Dropout to prevent overfitting on dominant features (e.g. FICO, interest rate)
    - L2 weight regularization
    - Sigmoid activation on output producing calibrated Probability of Default (PD)
    """
    inputs = keras.Input(shape=(input_dim,), name="applicant_features")

    # Layer 1: High capacity feature projection
    x = layers.Dense(128, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    # Layer 2: Intermediate non-linear interactions
    x = layers.Dense(64, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    # Layer 3: Risk representation distillation
    x = layers.Dense(32, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)

    # Output: Probability of Default (PD) in [0, 1]
    outputs = layers.Dense(1, activation="sigmoid", name="probability_of_default")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CreditIQ_ANN")

    # Optimizer with gradient clipping to prevent exploding gradients
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    return model

def compute_optimal_threshold(y_true, y_prob, cost_fn=5.0, cost_fp=1.0):
    """
    Finds the optimal decision threshold for credit underwriting based on business cost matrix.
    
    In credit risk:
    - False Negative (approving a borrower who defaults): high financial loss (e.g., loss of principal)
    - False Positive (declining a creditworthy applicant): opportunity cost of missed interest
    
    cost_fn: relative cost penalty for default (False Negative), default=5.0
    cost_fp: relative cost penalty for rejecting creditworthy borrower (False Positive), default=1.0
    """
    thresholds = np.linspace(0.05, 0.95, 100)
    best_threshold = 0.50
    min_cost = float("inf")
    metrics_at_best = {}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        total_cost = (fn * cost_fn) + (fp * cost_fp)
        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = float(t)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics_at_best = {
                "threshold": round(best_threshold, 4),
                "cost": float(total_cost),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "approval_rate": round(float((tn + fn) / len(y_true)), 4)
            }

    return best_threshold, metrics_at_best
