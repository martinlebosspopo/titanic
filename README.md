# 1. Github
## 1.1 Branches
### 1.1.1 master
> Branche sur laquelle sont édités les fichiers / fonctions qui peuvent servir à tous 
les modèles.

_Exemples_: 
- `utils/` : Fonctions utiles à tous les modèles
- `main_sk_model.ipynb` : Modèle de Notebook à suivre pour les modèles Sklearn.
- `main_tf_model.ipynb` : Modèle de Notebook à suivre pour les modèles tensorflow.

**PS :** Use `DummyClassifier` as the standard ML model.

### 1.1.2 _nom_model_

> Branche spécifique à **UN** modèle ML.
- `main.ipynb` : Notebook principal
- `main_XX_model.ipynb` : ! NE PAS MODIFIER !
- `utils/` : Peut modifier si doit rajouter / modifier des fonctions spécifiquement 
  pour ce modèle. **Mais à éviter** !! Pour garder mêmes versions de fonctions partout autant que 
  possible.

## 1.2 Workflow

### a) Préparation des données / Modifications pouvant toucher tous les modèles
- Passer sur `master`
- Modifier `utils/...`
- Adapter `main_sk_model.ipynb` + `main_tf_model.ipynb` en conséquence
- Commit 📤
- Passer sur `nom_model`
- Merge master ☯
- Adapter `main.ipynb`
- Commit 📤
- MLflow run 🌊

!! Bien pensé à commit sur `nom_model` avant de lancer `mlflow run` !!
(Sinon `mlflow` montrera précédent commit)

### b) Modifications pour un modèle
- Passer sur `nom_model`
- Faire modifs
- Commit 📤
- MLflow run 🌊
------
# 2. Notebooks

- `main_sk_model.ipynb` : Modèle de Notebook. Contient les étapes qui apparaissent pour la plupart des 
modèles sklearn.
- `main.ipynb` : Notebook vide. A personnaliser pour chaque modèle ML