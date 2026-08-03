# Classificazione automatica di cellule del midollo osseo tramite tecniche di ML

Tesi di laurea — analisi esplorativa e classificazione multiclasse di caratteristiche
cellulari ematologiche estratte da immagini di midollo osseo.

## Descrizione del progetto

Il progetto affronta il problema della classificazione automatica di cellule del sangue
a partire da caratteristiche numeriche estratte da immagini microscopiche, con l'obiettivo
di supportare l'analisi clinica in ambito ematologico. Lo studio si concentra su cinque
tipologie cellulari: **Basofili, Eosinofili, Eritroblasti, Monociti e Linfociti**, alcune
delle quali associate a condizioni patologiche come i tumori del sangue.

Il lavoro si articola in due fasi principali:

1. **Analisi esplorativa dei dati (EDA)** — studio della distribuzione delle classi,
   sbilanciamenti, variabilità delle feature e relazioni tra di esse, con tecniche di
   riduzione dimensionale (PCA, UMAP, t-SNE) per visualizzare cluster naturali e
   sovrapposizioni tra classi.
2. **Classificazione multiclasse** — confronto tra diversi algoritmi di apprendimento
   supervisionato (Random Forest, XGBoost, LightGBM, TabNet), integrati in pipeline con
   preprocessing (imputazione, normalizzazione, bilanciamento) e valutati con metriche
   robuste per dataset sbilanciati (Recall Macro, Balanced Accuracy). Il tuning degli
   iperparametri è stato condotto con Optuna.

Il classificatore **TabNet**, dopo ottimizzazione, è risultato il modello con il miglior
compromesso tra accuratezza complessiva e capacità di riconoscere le classi minoritarie.

## Dataset

Questo progetto utilizza il dataset **Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer**,
una raccolta pubblica di oltre 170.000 immagini di cellule del midollo osseo, de-identificate
e annotate da esperti, provenienti da 945 pazienti. Le immagini sono state acquisite con
microscopio a campo chiaro (40x, immersione in olio) presso il Munich Leukemia Laboratory
(MLL), con scansione a cura del Fraunhofer IIS e post-processing sviluppato da Helmholtz
Munich. Il dataset copre 21 tipologie cellulari e un ampio spettro di condizioni
ematologiche, incluse leucemie e altre displasie.

Per questo progetto è stato selezionato un sottoinsieme di **5 classi cellulari**: Basofili
(BAS), Eritroblasti (EBO), Eosinofili (EOS), Linfociti (LYT) e Monociti (MON).

I file usati in questo repository (`train_features_bone_marrow.h5`,
`test_features_bone_marrow.h5`) **non contengono le immagini originali**, ma un set di
64 feature numeriche per cellula, già estratte e fornite dal laboratorio/relatore di tesi,
più l'etichetta di classe.

**Fonte e citazione del dataset originale:**
Matek, C., Krappe, S., Münzenmayer, C., Haferlach, T., & Marr, C. (2021).
*An Expert-Annotated Dataset of Bone Marrow Cytology in Hematologic Malignancies* [Data set].
The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.AXH3-T579

Il dataset originale (immagini) è disponibile pubblicamente su TCIA, soggetto alla
[Data Usage Policy di TCIA](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/).
Non è incluso in questo repository per motivi di dimensione e di policy di ridistribuzione;
per riprodurre l'analisi è necessario procurarsi o ricreare feature nello stesso formato
(vedi `data/README.md`).

## Struttura del repository

```
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb                    # Analisi esplorativa e visualizzazione (PCA, t-SNE, UMAP)
│   ├── 02_model_comparison.ipynb       # Training e confronto modelli (RF, XGBoost, LightGBM, TabNet)
│   └── 03_hyperparameter_tuning.ipynb  # Ottimizzazione iperparametri con Optuna
├── src/
│   ├── data_loading.py                 # Caricamento dati e mapping delle classi
│   ├── eda_utils.py                    # Funzioni di visualizzazione riusate nell'EDA
│   └── evaluation.py                   # Calcolo metriche di valutazione dei modelli
└── data/
    └── README.md                       # Formato atteso dei dati di input
```

## Metodologia

- **Preprocessing:** imputazione dei valori mancanti, normalizzazione (StandardScaler),
  gestione dello sbilanciamento tra classi.
- **Modelli confrontati:** Random Forest, XGBoost, LightGBM, TabNet.
- **Metriche di valutazione:** Balanced Accuracy, Recall Macro, F1 Macro — scelte per la
  loro robustezza in presenza di classi minoritarie sottorappresentate.
- **Tuning:** ricerca automatica degli iperparametri con Optuna, con validazione tramite
  k-fold cross validation.

## Come eseguire il progetto

```bash
pip install -r requirements.txt
```

Procurarsi i file di dati nel formato descritto in `data/README.md` e posizionarli nella
cartella `data/`, quindi eseguire i notebook nell'ordine indicato in `notebooks/`.

## Tesi completa

Il testo completo della tesi è disponibile [qui] (docs/Metodologie_ di_ML_analisi_classificazione_cellule_Ematopoietiche.pdf).

## Autore

Tesi di laurea — Francesca Longo
