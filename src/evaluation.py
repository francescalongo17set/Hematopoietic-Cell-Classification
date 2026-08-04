"""
Calcolo delle metriche di valutazione dei modelli.

Nel notebook originale, questo identico blocco (accuracy, balanced accuracy,
F1 macro, recall macro, classification report) veniva ricalcolato "a mano"
dopo ogni modello (Random Forest, XGBoost, LightGBM, TabNet). Qui diventa
un'unica funzione richiamata ovunque serva.
"""

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    classification_report,
)


def evaluate_model(y_test, y_pred, model_name=None, verbose=True):
    """
    Calcola e stampa le metriche di valutazione usate nel confronto tra modelli.

    Le metriche (Balanced Accuracy, Recall Macro, F1 Macro) sono state scelte
    per la loro robustezza in presenza di classi minoritarie sottorappresentate.

    Args:
        y_test: etichette vere.
        y_pred: etichette predette dal modello.
        model_name (str): nome del modello, solo per etichettare l'output stampato.
        verbose (bool): se True, stampa anche il classification report completo.

    Returns:
        dict: dizionario con le metriche calcolate (accuracy, balanced_accuracy,
            f1_macro, recall_macro).
    """
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")

    if model_name:
        print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.3f}")
    print(f"F1 Macro: {f1_macro:.3f}")
    print(f"Recall Macro: {recall_macro:.3f}")

    if verbose:
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "f1_macro": f1_macro,
        "recall_macro": recall_macro,
    }
