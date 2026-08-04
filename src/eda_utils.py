"""
Funzioni di visualizzazione riusate più volte nella sezione EDA del notebook.

Nel notebook originale, il codice per il grafico di skewness era ripetuto
identico (a parte colore e titolo) una volta per ciascuna delle 5 classi
cellulari; lo stesso valeva per i grafici KDE/istogramma ripetuti per ogni
variabile. Qui diventano funzioni parametriche riutilizzabili.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew  # noqa: F401  (import mantenuto per compatibilità con l'uso originale)


def plot_skewness(df_subset, title, color="tomato"):
    """
    Disegna un grafico a barre dell'asimmetria (skewness) delle variabili
    per un sottoinsieme di dati (es. le osservazioni di una singola classe).

    Args:
        df_subset (pd.DataFrame): sottoinsieme di dati (es. solo una classe),
            con le sole colonne numeriche delle feature.
        title (str): titolo del grafico (es. "Asimmetria delle osservazioni
            assegnate alle cellule basofile").
        color (str): colore delle barre.
    """
    skewness = df_subset.skew()

    plt.figure(figsize=(10, 5))
    skewness.plot(kind="bar", color=color)
    plt.axhline(0, color="black", linestyle="dashed")
    plt.title(title)
    plt.ylabel("Valore di skewness")
    plt.show()


def plot_kde_by_class(df_std, variable, class_col, palette, title=None):
    """
    Disegna la kernel density estimation (KDE) di una variabile, colorata
    per classe.

    Args:
        df_std (pd.DataFrame): dataframe con le variabili standardizzate.
        variable: nome/indice della colonna da visualizzare.
        class_col: nome/indice della colonna che contiene l'etichetta di classe.
        palette: palette colori da usare (una voce per classe).
        title (str): titolo del grafico. Se None, viene generato automaticamente.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=df_std,
        x=df_std[variable],
        hue=df_std[class_col],
        fill=False,
        common_norm=False,
        palette=palette,
        linewidth=2,
    )
    plt.title(title or f"Kernel Density Estimation per Variabile {variable}")
    plt.xlabel(f"Variabile {variable}")
    plt.ylabel("Densità")
    plt.tight_layout()
    plt.show()


def plot_histogram_by_class(df_std, variable, class_col, palette, bins=40, title=None):
    """
    Disegna l'istogramma di una variabile, colorato per classe.

    Args:
        df_std (pd.DataFrame): dataframe con le variabili standardizzate.
        variable: nome/indice della colonna da visualizzare.
        class_col: nome/indice della colonna che contiene l'etichetta di classe.
        palette: palette colori da usare (una voce per classe).
        bins (int): numero di bin dell'istogramma.
        title (str): titolo del grafico. Se None, viene generato automaticamente.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(
        data=df_std,
        x=df_std[variable],
        hue=df_std[class_col],
        palette=palette,
        kde=False,
        bins=bins,
    )
    plt.title(title or f"Istogramma per Variabile {variable}")
    plt.xlabel(f"Valore di Variabile {variable}")
    plt.ylabel("Frequenza")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_boxplot_by_class(df, variable, class_col, palette, title=None):
    """
    Disegna il boxplot di una variabile condizionato alla classe.

    Nel notebook originale questo codice era ripetuto identico (a parte la
    variabile) per confrontare la variabilità di più feature tra le classi
    (es. variabili 0, 1, 4, 16, 51, 37, 39, 60, 61).

    Args:
        df (pd.DataFrame): dataset (non standardizzato) con la colonna target.
        variable: nome/indice della colonna da visualizzare.
        class_col: nome/indice della colonna che contiene l'etichetta di classe.
        palette: palette colori da usare (una voce per classe).
        title (str): titolo del grafico. Se None, viene generato automaticamente.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=df[class_col], y=df[variable], palette=palette)
    plt.title(title or f"Boxplot variabile {variable}")
    plt.xlabel("Classe")
    plt.ylabel(f"Variabile {variable}")
    plt.tight_layout()
    plt.show()


def plot_describe_table(df, col_start, col_end, filename=None):
    """
    Disegna una tabella con le statistiche descrittive (describe()) di un
    intervallo di colonne del dataset.

    Nel notebook originale questo codice era ripetuto identico (a parte
    l'intervallo di colonne e il dataset) per mostrare le 64 variabili in
    gruppi da ~20 sia per il training set che per il test set.

    Args:
        df (pd.DataFrame): dataset da cui estrarre le colonne.
        col_start (int): indice della prima colonna (inclusivo).
        col_end (int): indice dell'ultima colonna (esclusivo).
        filename (str): se fornito, salva la tabella come immagine PNG.
    """
    desc = df.iloc[:, col_start:col_end].describe()

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axis("off")
    ax.axis("tight")

    table = ax.table(
        cellText=desc.round(2).values,
        rowLabels=desc.index,
        colLabels=desc.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def plot_dataset_summary_table(df, filename=None):
    """
    Disegna una tabella riassuntiva con le informazioni generali del dataset
    (numero di righe/colonne, valori nulli, tipi di dato).

    Nel notebook originale questo codice era ripetuto identico per il
    training set e per il test set.

    Args:
        df (pd.DataFrame): dataset da riassumere.
        filename (str): se fornito, salva la tabella come immagine PNG.
    """
    import pandas as pd

    dtypes_count = df.dtypes.value_counts()
    dtype_detail = ", ".join([f"{dtype}: {count}" for dtype, count in dtypes_count.items()])

    info_dict = {
        "Numero di righe": [df.shape[0]],
        "Numero di colonne": [df.shape[1]],
        "Valori nulli totali": [df.isnull().sum().sum()],
        "Colonne con valori nulli": [df.isnull().any().sum()],
        "Tipo colonne (unici)": [df.dtypes.nunique()],
        "Colonne numeriche": [(df.dtypes == "float64").sum() + (df.dtypes == "int64").sum()],
        "Colonne oggetto": [(df.dtypes == "object").sum()],
        "Dettaglio tipi di dato": [dtype_detail],
    }

    summary_df = pd.DataFrame(info_dict).T
    summary_df.columns = ["Valore"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    ax.axis("tight")

    table = ax.table(
        cellText=summary_df.values,
        rowLabels=summary_df.index,
        colLabels=summary_df.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.2)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def remove_correlated_features(X, threshold=0.5):
    """
    Rimuove le variabili altamente correlate tra loro (in valore assoluto),
    tenendo solo il triangolo superiore della matrice di correlazione per
    non contare due volte la stessa coppia.

    Args:
        X (pd.DataFrame): dataset con le sole variabili numeriche (senza target).
        threshold (float): soglia di correlazione assoluta oltre la quale una
            delle due variabili correlate viene eliminata.

    Returns:
        tuple: (X_filtered, to_drop) dove X_filtered è il dataset senza le
            variabili eliminate, e to_drop è la lista delle variabili rimosse.
    """
    import numpy as np
    import pandas as pd

    correlation_matrix = X.corr()
    corr_matrix = correlation_matrix.abs()

    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_matrix = pd.DataFrame(corr_matrix.values, index=corr_matrix.index, columns=corr_matrix.columns)
    upper_triangle = upper_matrix.where(upper)

    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]

    X_filtered = X.drop(columns=to_drop)

    print(f"Variabili eliminate perché correlate > ±{threshold}:\n{to_drop}")
    print(f"Nuova forma del dataset: {X_filtered.shape}")

    return X_filtered, to_drop


def seleziona_variabili_pca(df, n_componenti=30, n_variabili=30):
    """
    Riduce il dataset alle variabili originali più influenti usando i loadings
    della PCA.

    Args:
        df (pd.DataFrame): dataset completo (64 feature + 1 target).
        n_componenti (int): numero di componenti principali da considerare.
        n_variabili (int): numero di variabili originali da selezionare.

    Returns:
        pd.DataFrame: dataset ridotto con le top variabili e la colonna target.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    # Separazione feature/target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]  # noqa: F841  (mantenuta per chiarezza, non usata direttamente qui)

    # Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA()
    pca.fit(X_scaled)

    # Loadings matrix
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    )

    # Calcolo dell'importanza media assoluta sulle prime n_componenti
    importance = loadings.iloc[:, :n_componenti].abs().mean(axis=1)

    # Selezione delle top n_variabili
    top_features = importance.sort_values(ascending=False).head(n_variabili).index

    # Creazione dataset ridotto
    df_ridotto = df[top_features.tolist() + [df.columns[-1]]]

    return df_ridotto
