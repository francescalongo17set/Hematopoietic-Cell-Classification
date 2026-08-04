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
