# DecodeLabs AI Internship - Project 2

## Iris Flower Classification with K-Nearest Neighbors

This project trains a supervised classifier to predict the species of an Iris
flower from its sepal and petal measurements. It uses scikit-learn's built-in
Iris dataset, so no separate dataset download is needed.

## Dataset

- 150 flower samples
- 4 numeric measurements per sample
- 3 balanced species: setosa, versicolor, and virginica

## Requirements

- Python 3.11 or newer
- The package listed in `requirements.txt`

## Run on Windows

Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python iris_classifier.py
```

If PowerShell blocks virtual-environment activation, skip the activation line
and run these instead:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe iris_classifier.py
```

## Workflow

1. Load the Iris dataset and print its features, classes, counts, and sample rows.
2. Shuffle and split the data into 80% training and 20% held-out test sets,
   keeping the species balanced in each split.
3. Compare K values from 1 to 15 with five-fold cross-validation on the training
   set only.
4. Scale measurements with `StandardScaler` and train a KNN model using the
   selected K. The scaler is inside a pipeline so it never learns from the
   held-out test set.
5. Evaluate once on the test set and report accuracy, macro F1, a confusion
   matrix, and per-species precision/recall/F1.

The random seed is fixed so the split and scores can be reproduced.

## Example result

With the included fixed split, cross-validation selects `K=3`. The held-out
test set in the verification run scored 93.33% accuracy and 0.933 macro F1.
