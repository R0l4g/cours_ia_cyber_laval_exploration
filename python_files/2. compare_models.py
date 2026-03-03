# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (pixi)
#     language: python
#     name: cours_ia_cyber_laval_exploration
# ---

# %% [markdown]
# # Compare machine learning models
#
# In this notebook, we will compare 3 pre-trained models that predict the
# **Census Region** of a respondent based on their survey answers.
#
# The 3 models are:
# - **Logistic Regression**: a simple linear model
# - **Random Forest**: a model based on many decision trees
# - **Gradient Boosting**: a model that builds trees sequentially

# %% [markdown]
# ## Load the dataset

# %%
from skrub.datasets import fetch_midwest_survey

dataset = fetch_midwest_survey()
X = dataset.X
y = dataset.y

# %%
# To simplify evaluation, we will group categories in the target to deal with
# a binary classification problem instead of a multiclass one.
y = y.apply(
    lambda x: "North Central"
    if x in ["East North Central", "West North Central"]
    else "other"
)

# %%
# Train / test split based on a sample of 1000 rows for training
sample_idx = X.sample(n=1000, random_state=1).index
X_train = X.loc[sample_idx].reset_index(drop=True)
y_train = y.loc[sample_idx].reset_index(drop=True)
X_test = X.drop(sample_idx).reset_index(drop=True)
y_test = y.drop(sample_idx).reset_index(drop=True)

# %% [markdown]
# ## Load the 3 models
#
# The models were saved as `.pkl` files. We use `joblib` to load them.

# %%
import joblib
from midwest_survey_models.transformers import NumericalStabilizer  # type: ignore  # noqa: F401

model_lr = joblib.load("../model_logistic_regression.pkl")
model_rf = joblib.load("../model_random_forest.pkl")
model_gb = joblib.load("../model_gradient_boosting.pkl")

# %% [markdown]
# Let's inspect what each model looks like. They are **pipelines**: they
# first transform the data, then make predictions.

# %%
print("Logistic Regression pipeline:")
print(model_lr)

# %%
print("\nRandom Forest pipeline:")
print(model_rf)

# %%
print("\nGradient Boosting pipeline:")
print(model_gb)

# %% [markdown]
# ## Evaluate the models with cross-validation
#
# To fairly evaluate each model, we use **cross-validation**.

# %%
from sklearn.model_selection import cross_val_score

cv_lr = cross_val_score(model_lr, X, y, cv=5)
cv_rf = cross_val_score(model_rf, X, y, cv=5)
cv_gb = cross_val_score(model_gb, X, y, cv=5)

print("\n=== Cross-validation accuracy ===")
print("Logistic Regression CV accuracy:",
      cv_lr.mean(), "+/-", cv_lr.std())
print("Random Forest CV accuracy:",
      cv_rf.mean(), "+/-", cv_rf.std())
print("Gradient Boosting CV accuracy:",
      cv_gb.mean(), "+/-", cv_gb.std())

# %% [markdown]
# ## Question 6: Among the three models, which one has the best recall?
#
# We define the positive class as **\"North Central\"** and use
# `EstimatorReport` to compute precision, recall and f1-score.

# %%
from skore import EstimatorReport  # type: ignore

reports = {}
print("\n=== Metrics (precision / recall / f1 for 'North Central') ===")
for name, model in [
    ("Logistic Regression", model_lr),
    ("Random Forest", model_rf),
    ("Gradient Boosting", model_gb),
]:
    rep = EstimatorReport(
        estimator=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pos_label="North Central",
    )
    reports[name] = rep
    print("\n====", name, "====")
    from sklearn.metrics import precision_score, recall_score, f1_score
    import pandas as pd

    y_pred = model.predict(X_test)
    prec = precision_score(y_test, y_pred, pos_label="North Central")
    rec = recall_score(y_test, y_pred, pos_label="North Central")
    f1 = f1_score(y_test, y_pred, pos_label="North Central")

    df_metrics = pd.DataFrame({
        "precision": [prec],
        "recall": [rec],
        "f1": [f1],
    }, index=[name])
    print(df_metrics)

# %% [markdown]
# From the tables above, we can read which model has the **highest recall**
# on the positive class "North Central".

# %% [markdown]
# ## Question 7: Which model has the best practical application?
#
# We define a custom practical score:
# - True positive:  +5
# - True negative:  +2
# - False positive: -10
# - False negative: -1

# %%
import numpy as np

def practical_score(y_true, y_pred):
    score = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == "North Central" and yp == "North Central":
            score += 5      # true positive
        elif yt != "North Central" and yp != "North Central":
            score += 2      # true negative
        elif yt != "North Central" and yp == "North Central":
            score -= 10     # false positive
        elif yt == "North Central" and yp != "North Central":
            score -= 1      # false negative
    return score

print("\n=== Practical scores ===")
for name, model in [
    ("Logistic Regression", model_lr),
    ("Random Forest", model_rf),
    ("Gradient Boosting", model_gb),
]:
    y_pred = model.predict(X_test)
    score = practical_score(y_test, y_pred)
    print(name, "practical score:", score)

# %% [markdown]
# The model with the **highest practical score** above is the one that makes
# the most useful predictions in practice under this cost setup.

# %% [markdown]
# ## Question 8: Which model generalizes the best?
#
# We compare train and test accuracy for each model, and also look at the
# cross-validation scores computed earlier.

# %%
print("\n=== Train / test accuracy ===")
for name, model in [
    ("Logistic Regression", model_lr),
    ("Random Forest", model_rf),
    ("Gradient Boosting", model_gb),
]:
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"\n{name}")
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")

# %% [markdown]
# Using the train/test accuracies and the cross-validation means printed
# above, we can determine:
# - which model has the **smallest gap** between train and test accuracy
#   (best generalization),
# - which model has the **largest gap** and is likely overfitting.

# %%
# TODO: Based on the results above, which model would you choose
# for a real application? Write your answer as a comment below.

# My choice: ...
# Reason: ...
