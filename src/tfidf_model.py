import pandas as pd
import numpy as np
import time
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    accuracy_score,
    roc_auc_score,
    log_loss
)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# Load dataset
df_metadata = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "movies_metadata.csv")
df_keywords = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "keywords.csv")

# Merge metadata and keywords on 'id'
df_metadata['id'] = pd.to_numeric(df_metadata['id'], errors='coerce')
df_keywords['id'] = pd.to_numeric(df_keywords['id'], errors='coerce')
df_merged = pd.merge(df_metadata, df_keywords, on='id')

data = {}
for row in df_merged.itertuples():
    try:
        genres_list = ast.literal_eval(row.genres)
        genre_ids = [item['id'] for item in genres_list]

        keywords_list = ast.literal_eval(row.keywords)
        keyword_ids = [item['id'] for item in keywords_list]

        overview = row.overview if isinstance(row.overview, str) else ""

        if overview.strip() and genre_ids:
            data[int(row.id)] = (overview, keyword_ids[:10], genre_ids[:5])
    except:
        continue


df = pd.DataFrame.from_dict(data, orient='index', columns=['overview', 'id_list', 'genre_list'])

# Combine overview and keywords
df['keywords_str'] = df['id_list'].apply(lambda ids: ' '.join(map(str, ids)))
df['text_input'] = df['overview'] + ' ' + df['keywords_str']

# OPTIONAL: Drop rows where 'text_input' is empty
df = df[df['text_input'].str.strip().astype(bool)]

# Define the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

# Vectorize the combined text input
X = vectorizer.fit_transform(df['text_input'])

# Multi-label binarization for genres
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genre_list'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))
model.fit(X_train, y_train)

# Predictions (with probabilities for AUC/LogLoss)
y_pred = model.predict(X_test)
y_pred_probs = model.predict_proba(X_test)

# Evaluation
print("\n--- Overall Evaluation Metrics ---")
print(f"Micro F1 Score     : {f1_score(y_test, y_pred, average='micro'):.4f}")
print(f"Macro F1 Score     : {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Micro Precision    : {precision_score(y_test, y_pred, average='micro', zero_division=0):.4f}")
print(f"Micro Recall       : {recall_score(y_test, y_pred, average='micro', zero_division=0):.4f}")
print(f"Hamming Loss       : {hamming_loss(y_test, y_pred):.4f}")
print(f"Subset Accuracy    : {accuracy_score(y_test, y_pred):.4f}")

# ROC AUC
try:
    roc_auc_micro = roc_auc_score(y_test, np.array(y_pred_probs), average='micro')
    roc_auc_macro = roc_auc_score(y_test, np.array(y_pred_probs), average='macro')
    print(f"ROC AUC (Micro)    : {roc_auc_micro:.4f}")
    print(f"ROC AUC (Macro)    : {roc_auc_macro:.4f}")
except ValueError:
    print("ROC AUC            : Not available (labels or predictions missing positives)")

# Log Loss
try:
    print(f"Log Loss           : {log_loss(y_test, y_pred_probs):.4f}")
except ValueError:
    print("Log Loss           : Not available (invalid predicted probabilities)")

# Speed Benchmark
print("\n--- Model Speed Benchmark ---")
start = time.time()
_ = model.predict(X_test[:1])
print(f"Single prediction time: {time.time() - start:.6f} sec")

start = time.time()
_ = model.predict(X_test)
print(f"Batch prediction time : {time.time() - start:.6f} sec")

samples = min(1000, X_test.shape[0])
start = time.time()
_ = model.predict(X_test[:samples])
print(f"Per-sample time (avg)  : {(time.time() - start)/samples:.6f} sec")

# Run PCA on the full TF-IDF matrix
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())  # X is sparse; convert to dense

# Assign a dominant genre for color coding (pick the first genre in the list for each sample)
dominant_genres = [mlb.classes_[labels[0]] if any(labels) else 'None' for labels in y]

# Plot
plt.figure(figsize=(12, 8))
unique_genres = list(set(dominant_genres))
colors = plt.cm.get_cmap('tab20', len(unique_genres))

for i, genre in enumerate(unique_genres):
    idxs = [j for j, g in enumerate(dominant_genres) if g == genre]
    plt.scatter(
        X_reduced[idxs, 0],
        X_reduced[idxs, 1],
        s=10,
        alpha=0.6,
        label=genre,
        color=colors(i)
    )

plt.title("PCA of TF-IDF Vectors Colored by Dominant Genre")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
