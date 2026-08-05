# Classificazione automatica di cellule del midollo osseo tramite tecniche di IA

Tesi di laurea — analisi esplorativa e classificazione multiclasse di caratteristiche
cellulari ematologiche estratte da immagini di midollo osseo.


![Visualizzazione UMAP 3D delle classi cellulari](visualizations/dimensionality_reduction/umap_3d.png)

## Descrizione del progetto

Il progetto affronta il problema della classificazione automatica di cellule del sangue
a partire da caratteristiche numeriche estratte da immagini microscopiche, con l'obiettivo
di supportare l'analisi clinica in ambito ematologico. Lo studio si concentra su cinque
tipologie cellulari: **Basofili, Eosinofili, Eritroblasti, Monociti e Linfociti**, alcune
delle quali associate a condizioni patologiche come i tumori del sangue.

Il lavoro si articola in due fasi principali:

1. **Analisi esplorativa dei dati (EDA)** — statistiche descrittive, distribuzione delle
   classi (fortemente sbilanciata), asimmetria e correlazione tra le feature, con tecniche
   di riduzione dimensionale (PCA, t-SNE, UMAP) per visualizzare cluster naturali e
   sovrapposizioni tra classi in 2D e 3D.
2. **Classificazione multiclasse** — confronto tra quattro algoritmi di apprendimento
   supervisionato (Random Forest, XGBoost, LightGBM, TabNet), ciascuno integrato in una
   pipeline di preprocessing (imputazione, normalizzazione, bilanciamento dei pesi per
   classe) e valutato con metriche robuste per dataset sbilanciati (Balanced Accuracy,
   Recall Macro, F1 Macro). Segue una fase di tuning degli iperparametri con Optuna.

Il classificatore **TabNet**, dopo ottimizzazione, è risultato il modello con il miglior
compromesso tra accuratezza complessiva e capacità di riconoscere le classi minoritarie —
un aspetto particolarmente rilevante in ambito clinico.

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
├── main.py                             # Pipeline completa eseguibile da riga di comando
├── notebooks/
│   ├── 01_eda.ipynb                    # Analisi esplorativa e visualizzazione (PCA, t-SNE, UMAP)
│   ├── 02_model_comparison.ipynb       # Training e confronto modelli (RF, XGBoost, LightGBM, TabNet)
│   └── 03_hyperparameter_tuning.ipynb  # Ottimizzazione iperparametri (GridSearchCV, Optuna)
├── src/
│   ├── data_loading.py                 # Caricamento dati e mapping delle classi
│   ├── eda_utils.py                    # Funzioni di visualizzazione ed EDA riusate
│   ├── preprocessing.py                # Pipeline di preprocessing condivisa tra i modelli
│   └── evaluation.py                   # Calcolo metriche e confusion matrix
├── data/
│   └── README.md                       # Formato atteso dei dati di input
├── visualizations/
│   ├── dimensionality_reduction/       # Screenshot PCA/t-SNE/UMAP 2D e 3D
│   └── model_results/                  # Screenshot confusion matrix del modello finale
└── docs/
    └── tesi.pdf                        # Testo completo della tesi (o link alla piattaforma d'ateneo)
```

## Metodologia

- **Selezione delle variabili:** rimozione delle feature con correlazione assoluta > 0.5,
  seguita dalla selezione delle 30 variabili più rilevanti secondo i loadings della PCA
  (da 64 a 30 variabili finali).
- **Preprocessing:** imputazione dei valori mancanti (media), normalizzazione
  (`StandardScaler`), pesi per campione (`compute_sample_weight`) per bilanciare
  l'importanza delle classi minoritarie durante il training.
- **Modelli confrontati:** Random Forest, XGBoost, LightGBM, TabNet.
- **Metriche di valutazione:** Balanced Accuracy, Recall Macro, F1 Macro — scelte per la
  loro robustezza in presenza di classi minoritarie sottorappresentate.
- **Tuning:** ricerca automatica degli iperparametri con Optuna (TabNet) e GridSearchCV
  (LightGBM), con validazione tramite k-fold cross validation stratificata.

## Risultati


![Confusion matrix di TabNet ottimizzato](visualizations/model_results/confusion_matrix_tabnet_tuned.png)


| Modello            | Accuracy | Balanced Accuracy | F1 Macro | Recall Macro |
|---------------------|----------|--------------------|----------|--------------|
| Random Forest       | 0.77     | 0.64               | 0.56     | 0.64         |
| XGBoost             | 0.85     | 0.66               | 0.67     | 0.66         |
| LightGBM            | 0.85     | 0.65               | 0.66     | 0.65         |
| TabNet              | 0.79     | 0.69               | 0.62     | 0.68         |
| **TabNet (tuned)**  | **0.82** | **0.72**           | 0.65     | **0.72**     |

TabNet, dopo tuning degli iperparametri con Optuna, è risultato il modello con il miglior
compromesso tra accuratezza complessiva e capacità di riconoscere correttamente le classi
minoritarie (in particolare le cellule basofile, la classe meno rappresentata) — un aspetto
fondamentale in ambito medico, dove una bassa sensibilità su classi rare può portare a
conseguenze cliniche rilevanti.

## Come eseguire il progetto

```bash
pip install -r requirements.txt
```

Procurarsi i file di dati nel formato descritto in `data/README.md` e posizionarli nella
cartella `data/`. Da qui due opzioni:

- **Esplorazione passo passo:** eseguire i notebook in `notebooks/`, nell'ordine indicato.
- **Pipeline completa in un comando:**
  ```bash
  python main.py
  ```
  Allena e valuta in sequenza i quattro modelli, stampando un riepilogo finale delle
  metriche a confronto.

## Tesi completa

Il testo completo della tesi è disponibile [qui](docs/tesi.pdf)
*(oppure link alla piattaforma di ateneo, se la tesi è depositata lì).*

## Autore

Francesca Longo — Corso di Laurea in Statistica per i Big Data,
Università degli Studi di Salerno (A.A. 2024/2025)
