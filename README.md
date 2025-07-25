# Prédiction du risque de diabète

## 📌 Description du Projet
Ce projet vise à créer des clusters pour les personnes à risque de diabète puis à entrainer un modèle pour predire les clusters :
- **Données médicales**
- **Natures des données** (glucose, blood pressure etc.)
- **Modèle avancé** : Logistic Regression avec optimisation des hyperparamètres

**Approche technique** :
- Clustering en groupe selon les risques de diabète
- Comparaison de plusieurs modèles (Random Forest, SVR, Logistic Regression)
- Optimisation via GridSearch et CV

## 📂 Structure des Fichiers

├── data/

│ └── dataset.csv # Données brutes

├── scripts/

│ ├── analyse_exploratoire.ipynb # Analyse exploratoire (EDA)

│ ├── preprocessing.py # Fonctions permettant le nettoyage des données et la fonction permettant la création des clusters

│ ├── cluster.ipynb # Analyse et création des clusters 

│ ├── Models.ipynb # Entrainement et selection du meilleur modèle

│ └── Test_models.ipynb # Test sur les data pour évaluer le modèle

├── models/

│ └── model.pkl # Modèle sauvegardé (Logistic Regression)

└── README.md

  - Jira: 
 
 ## 🛠️ Technologies Utilisées
- **Langage**: Python
- **Librairies principales**:
  - `pandas`, `numpy` (traitement des données)
  - `scikit-learn` (modèles de ML)
  - `imblearn` (oversampling, pipeline)
  - `matplotlib`, `seaborn` (visualisation)


  📝 Méthodologie
- **Prétraitement** 
  - Nettoyage des valeurs aberrantes
  - Normalisation des données numériques

- **Clustering**
  - Choix du nombre de clusters
  - Determination des clusters

- **Modeles et Optimisation**
  - Validation croisée (5 folds)
  - Recherche d'hyperparamètres à l'aide de GridSearch et CV
  - Choix du meilleur modèle (Logistic Regression)


  ## 🚀 Résultats Clés
### Performance des Modèles sur le jeu test (Comparaison)
| Modèle                 | CV F1 (moyen)| F1 test|
|----------------------  |-------       |------- |
| **Logistic Regression**| 0.99         | 1.0    |
| Random Forest          | 0.95         | 0.94   |
| Gradient Boosting      | 0.96         | 0.97   |
| SVM                    | 0.99         | 1.0    |