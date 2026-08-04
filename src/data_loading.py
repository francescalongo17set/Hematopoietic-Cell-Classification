"""
Caricamento dei dati e mapping delle classi cellulari.

Codice estratto dalla sezione "Uploading dei dati" del notebook originale
(le stesse righe comparivano identiche più volte nel file).
"""

import numpy as np
import h5py


# Mappa le classi numeriche [0,1,2,3,4] alle sigle delle cellule corrispondenti
int_to_label = {
    0: "BAS",  # Basophil
    1: "EBO",  # Erythroblast
    2: "EOS",  # Eosinophil
    3: "LYT",  # Lymphocyte
    4: "MON",  # Monocyte
}


def load_class_data(features_path):
    """
    Carica i dati da un file HDF5.

    Args:
        features_path (str): Percorso del file HDF5 contenente le feature.

    Returns:
        X_data (np.ndarray): Feature delle classi selezionate.
        y_data (np.ndarray): Label delle classi selezionate.
    """
    with h5py.File(features_path, "r") as f:
        data = np.array(f["features"])

    # Separazione feature e labels
    X_data = data[:, :-1]  # Tutte le colonne tranne l'ultima
    y_data = data[:, -1]   # Ultima colonna come label

    return X_data, y_data
