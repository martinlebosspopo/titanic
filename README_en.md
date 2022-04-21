# 1 Github
## 1.1 Branches
### 1.1.1 master
> Branch on which are edited the files / functions that can are useful for all 
models.

_Examples_: 
- `utils/` : Functions useful for all models
- `main_sk_model.ipynb` : Notebook template for Sklearn models.
- `main_tf_model.ipynb` : Notebook template for tensorflow models.

Here, use `DummyClassifier` as the standard ML model.

### 1.1.2 _model_name_

> Branch specific to **ONE** ML model.
- `main.ipynb` : Main notebook
- `main_XX_model.ipynb` : ! DO NOT MODIFY !
- `utils/` : Can modify if need to modify functions specifically for this model. **But try to 
  avoid** ! In order to keep same versions of functions everywhere as much as possible.

## 1.2 Workflow

### a) Data preparation / Modifications that can affect all models
- Switch to `master`
- Modify `utils/...`
- Adapt `main_sk_model.ipynb` + `main_tf_model.ipynb` accordingly
- Commit 📤
- Switch to `model_name`.
- Merge master ☯
- Adapt `main.ipynb` accordingly
- Commit 📤
- MLflow run 🌊

!! Make sure to commit on `model_name` before running `mlflow run` !!!
(Otherwise `mlflow` will show previous commit on UI)

### b) Modifications for one model
- Switch to `model_name`.
- Do modifs
- Commit 📤
- MLflow run 🌊
------
# 2. Notebooks

- `main_sk_model.ipynb` : Notebook template. Contains the steps that appear for most of the 
sklearn models.
- `main.ipynb` : Empty notebook. To be customized for each ML model
