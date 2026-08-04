"""
Pipeline completa: carica i dati, riduce le variabili a 30 (rimozione delle
variabili correlate + selezione tramite loadings della PCA, come descritto
nella tesi), allena i quattro modelli confrontati (Random Forest, XGBoost,
LightGBM, TabNet) e stampa un riepilogo delle metriche di valutazione per
ciascuno.

Uso:
    python main.py

I parametri dei modelli sono gli stessi usati nel notebook originale.
"""

import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from src.data_loading import load_class_data
from src.eda_utils import remove_correlated_features, seleziona_variabili_pca
from src.preprocessing import build_preprocessing_pipeline, preprocess_arrays
from src.evaluation import evaluate_model


# Percorsi dei dati (vedi data/README.md per il formato atteso)
TRAIN_FEATURES_PATH = "data/train_features_bone_marrow.h5"
TEST_FEATURES_PATH = "data/test_features_bone_marrow.h5"

# Soglia di correlazione oltre la quale una delle due variabili viene eliminata
CORRELATION_THRESHOLD = 0.5

# Numero di variabili finali selezionate tramite i loadings della PCA
N_TOP_FEATURES = 30


def load_raw_data():
    """Carica i dati grezzi (64 feature + target) e li converte in DataFrame."""
    X_train, y_train = load_class_data(TRAIN_FEATURES_PATH)
    X_test, y_test = load_class_data(TEST_FEATURES_PATH)

    df_train = pd.DataFrame(X_train)
    df_test = pd.DataFrame(X_test)
    df_train[64] = pd.Series(y_train).astype("category").cat.codes
    df_test[64] = pd.Series(y_test).astype("category").cat.codes

    return df_train, df_test


def reduce_features(df_train, df_test):
    """
    Riproduce la fase di riduzione delle variabili descritta nella tesi:

    1. Rimuove le variabili con correlazione assoluta > 0.5 (calcolata solo
       sul training set, per evitare data leakage).
    2. Seleziona le 30 variabili più rilevanti secondo i loadings della PCA
       (calcolati anch'essi solo sul training set).
    3. Applica la stessa selezione di variabili al test set, così da avere
       un preprocessing coerente tra i due dataset.

    Returns:
        tuple: (df_train_reduced, df_test_reduced), entrambi con le 30
            variabili selezionate più la colonna target.
    """
    target_col = df_train.columns[-1]

    X_train = df_train.iloc[:, :-1]
    X_test = df_test.iloc[:, :-1]

    # 1. Rimozione delle variabili correlate (fit sul solo training set)
    X_train_filtered, to_drop = remove_correlated_features(X_train, threshold=CORRELATION_THRESHOLD)
    X_test_filtered = X_test.drop(columns=to_drop)

    df_train_filtered = X_train_filtered.copy()
    df_train_filtered[target_col] = df_train[target_col].values
    df_test_filtered = X_test_filtered.copy()
    df_test_filtered[target_col] = df_test[target_col].values

    # 2. Selezione delle 30 variabili più rilevanti tramite i loadings della PCA
    df_train_reduced = seleziona_variabili_pca(
        df_train_filtered, n_componenti=N_TOP_FEATURES, n_variabili=N_TOP_FEATURES
    )
    top_features = df_train_reduced.columns[:-1].tolist()

    # 3. Stessa selezione di variabili applicata al test set
    df_test_reduced = df_test_filtered[top_features + [target_col]]

    return df_train_reduced, df_test_reduced


def run_random_forest(X_train, y_train, X_test, y_test, numeric_columns, sample_weights):
    column_transform = build_preprocessing_pipeline(numeric_columns)

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=0,
        max_depth=6,
    )

    pipeline = Pipeline([
        ("processing", column_transform),
        ("modeling", model),
    ])

    start = time.time()
    pipeline.fit(X_train, y_train, modeling__sample_weight=sample_weights)
    train_time = time.time() - start

    y_pred = pipeline.predict(X_test)

    print(f"Tempo di training (Random Forest): {train_time:.2f} secondi")
    return evaluate_model(y_test, y_pred, model_name="Random Forest", verbose=False)


def run_xgboost(X_train, y_train, X_test, y_test, numeric_columns, sample_weights):
    column_transform = build_preprocessing_pipeline(numeric_columns)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        min_samples_leaf=3,
        use_label_encoder=False,
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=0,
    )

    pipeline = Pipeline([
        ("processing", column_transform),
        ("modeling", model),
    ])

    start = time.time()
    pipeline.fit(X_train, y_train, modeling__sample_weight=sample_weights)
    train_time = time.time() - start

    y_pred = pipeline.predict(X_test)

    print(f"Tempo di training (XGBoost): {train_time:.2f} secondi")
    return evaluate_model(y_test, y_pred, model_name="XGBoost", verbose=False)


def run_lightgbm(X_train, y_train, X_test, y_test, numeric_columns, sample_weights):
    column_transform = build_preprocessing_pipeline(numeric_columns)

    model = LGBMClassifier(
        objective="multiclass",
        num_class=5,
        eval_metric="multi_logloss",
        min_samples_leaf=3,
        use_label_encoder=False,
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=0,
    )

    pipeline = Pipeline([
        ("processing", column_transform),
        ("modeling", model),
    ])

    start = time.time()
    pipeline.fit(X_train, y_train, modeling__sample_weight=sample_weights)
    train_time = time.time() - start

    y_pred = pipeline.predict(X_test)

    print(f"Tempo di training (LightGBM): {train_time:.2f} secondi")
    return evaluate_model(y_test, y_pred, model_name="LightGBM", verbose=False)


def run_tabnet(X_train, y_train, X_test, y_test, sample_weights):
    X_train_arr, X_test_arr = preprocess_arrays(X_train.values, X_test.values)
    y_train_arr = y_train.values
    y_test_arr = y_test.values

    model = TabNetClassifier(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 50, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",
        seed=0,
        verbose=10,
        # 'cuda' se disponibile una GPU, altrimenti passare device_name='cpu'
        device_name="cuda" if torch.cuda.is_available() else "cpu",
    )

    start = time.time()
    model.fit(
        X_train=X_train_arr,
        y_train=y_train_arr,
        eval_set=[(X_test_arr, y_test_arr)],
        eval_name=["val"],
        eval_metric=["balanced_accuracy"],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        weights=sample_weights,
    )
    train_time = time.time() - start

    y_pred = model.predict(X_test_arr)

    print(f"Tempo di training (TabNet): {train_time:.2f} secondi")
    return evaluate_model(y_test_arr, y_pred, model_name="TabNet", verbose=False)


def main():
    # 1. Caricamento dati grezzi (64 variabili)
    df_train, df_test = load_raw_data()

    # 2. Riduzione a 30 variabili (correlazione + PCA loadings), come nella tesi
    df_train_reduced, df_test_reduced = reduce_features(df_train, df_test)

    X_train = df_train_reduced.iloc[:, :-1]
    y_train = df_train_reduced.iloc[:, -1]
    X_test = df_test_reduced.iloc[:, :-1]
    y_test = df_test_reduced.iloc[:, -1]

    numeric_columns = X_train.columns.tolist()
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # 3. Training e valutazione dei quattro modelli
    results = {}
    results["Random Forest"] = run_random_forest(X_train, y_train, X_test, y_test, numeric_columns, sample_weights)
    results["XGBoost"] = run_xgboost(X_train, y_train, X_test, y_test, numeric_columns, sample_weights)
    results["LightGBM"] = run_lightgbm(X_train, y_train, X_test, y_test, numeric_columns, sample_weights)
    results["TabNet"] = run_tabnet(X_train, y_train, X_test, y_test, sample_weights)

    print("\n=== Riepilogo finale ===")
    summary = pd.DataFrame(results).T
    print(summary.round(3))


if __name__ == "__main__":
    main()
