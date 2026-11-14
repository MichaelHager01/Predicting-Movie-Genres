# Predicting movie genres

Two multi-label models that predict up to five genres from a movie overview and its keywords.

## What it does

- `src/tfidf_model.py` vectorizes the overview plus keyword ids with TF-IDF (10,000 features) and fits one logistic regression per genre.
- `src/neural_network_model.py` embeds up to ten keyword ids and trains a Keras network with a sigmoid output and binary cross-entropy.
- Both scripts score micro and macro F1, Hamming loss, subset accuracy, and ROC AUC on a 20% holdout.
- The Movie dataset files are not in this repository. Place `keywords.csv` and `movies_metadata.csv` in `data/` before running.

## Layout

- `src/tfidf_model.py` — TF-IDF and one-vs-rest logistic regression
- `src/neural_network_model.py` — keyword embedding network
- `data/` — you add `keywords.csv` and `movies_metadata.csv` here

## Run

From the repository root, after the two CSVs are in `data/`:

```bash
pip install -r requirements.txt
python src/tfidf_model.py
python src/neural_network_model.py
```

## Stack

Python, pandas, NumPy, scikit-learn, TensorFlow, Matplotlib, seaborn. Written for the Purdue AI and machine learning program.
