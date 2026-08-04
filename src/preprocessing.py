"""
Preprocessing condiviso tra i modelli di classificazione.

Nel notebook originale la stessa logica (SimpleImputer strategy='mean' +
StandardScaler) veniva applicata due modi diversi a seconda del modello:
- come ColumnTransformer dentro una sklearn Pipeline, per XGBoost e LightGBM
- "a mano" con fit_transform/transform, per TabNet (che richiede array numpy
  già pronti e non accetta una Pipeline sklearn come step di preprocessing)

Qui la logica è unificata in un solo posto.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessing_pipeline(numeric_columns):
    """
    Costruisce il preprocessing per modelli compatibili con sklearn Pipeline
    (es. XGBoost, LightGBM, Random Forest).

    Args:
        numeric_columns: elenco delle colonne numeriche su cui applicare
            imputazione e scaling.

    Returns:
        ColumnTransformer da inserire come step 'processing' in una Pipeline.
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    return ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_columns)
    ], remainder='drop', verbose_feature_names_out=False)


def preprocess_arrays(X_train, X_test):
    """
    Applica imputazione e scaling direttamente su array numpy, per modelli
    che non accettano una sklearn Pipeline come preprocessing (es. TabNet).

    Args:
        X_train (np.ndarray): feature di training.
        X_test (np.ndarray): feature di test.

    Returns:
        tuple: (X_train, X_test) trasformati.
    """
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)

    X_test = imputer.transform(X_test)
    X_test = scaler.transform(X_test)

    return X_train, X_test
