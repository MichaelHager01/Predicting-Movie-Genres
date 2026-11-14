import pandas as pd
import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
    auc,
    roc_curve,
    multilabel_confusion_matrix
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import ast
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
df_keywords = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "keywords.csv")
df_metadata = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "movies_metadata.csv")
data = {}

# Process keywords
for row in df_keywords.itertuples():
    try:
        keywords_list = ast.literal_eval(row.keywords)
        id_list = [item['id'] for item in keywords_list]
        data[int(row.id)] = [id_list[:10]]
    except:
        continue

# Process genres
for row in df_metadata.itertuples():
    try:
        genres_list = ast.literal_eval(row.genres)
        id_list = [item['id'] for item in genres_list]
        data[int(row.id)].append(id_list[:5])
    except:
        continue

# Remove entries with missing keywords or genres
for key in list(data.keys()):
    if len(data[key]) != 2 or len(data[key][0]) == 0 or len(data[key][1]) == 0:
        del data[key]

# Analyze max lengths
max_keyword_len = 0
max_keyword_id = -1

max_genre_len = 0
max_genre_id = -1

for key in data:
    kw_len = len(data[key][0])
    genre_len = len(data[key][1])
    if kw_len > max_keyword_len:
        max_keyword_len = kw_len
        max_keyword_id = key
    if genre_len > max_genre_len:
        max_genre_len = genre_len
        max_genre_id = key
        
print(f"Max keyword length: {max_keyword_len}")
print(f"Max genre length  : {max_genre_len}")
    
# Create a DataFrame from the data dictionary
df = pd.DataFrame.from_dict(data, orient='index', columns=['keyword_list', 'genre_list'])
print(df.size)
print(df.head())

# Pad keyword sequences (integer IDs)
X = pad_sequences(df['keyword_list'], maxlen=max_keyword_len, padding='post')

# Multi-label binarization for genres
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genre_list'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

all_keyword_ids = [id_ for sublist in df['keyword_list'] for id_ in sublist]
vocab_size = max(all_keyword_ids) + 1

maxlen = max_keyword_len

# Model Definition and Training
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(len(mlb.classes_), activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save training history
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Plot loss and accuracy
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Predict on test set
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# Evaluation
print("\n--- Overall Evaluation Metrics ---")
print(f"Micro F1 Score     : {f1_score(y_test, y_pred, average='micro'):.4f}")
print(f"Macro F1 Score     : {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Micro Precision    : {precision_score(y_test, y_pred, average='micro', zero_division=0):.4f}")
print(f"Micro Recall       : {recall_score(y_test, y_pred, average='micro', zero_division=0):.4f}")
print(f"Hamming Loss       : {hamming_loss(y_test, y_pred):.4f}")
print(f"Subset Accuracy    : {accuracy_score(y_test, y_pred):.4f}")

# ROC AUC (only valid if y_test and y_pred_probs are not all 0/1)
try:
    roc_auc_micro = roc_auc_score(y_test, y_pred_probs, average='micro')
    roc_auc_macro = roc_auc_score(y_test, y_pred_probs, average='macro')
    print(f"ROC AUC (Micro)    : {roc_auc_micro:.4f}")
    print(f"ROC AUC (Macro)    : {roc_auc_macro:.4f}")
except ValueError:
    print("ROC AUC            : Not available (y_test or y_pred_probs missing positive labels)")

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

samples = 1000
start = time.time()
_ = model.predict(X_test[:samples])
print(f"Per-sample time (avg)  : {(time.time() - start)/samples:.6f} sec")

genre_labels = mlb.classes_

# Get confusion matrices for each genre
conf_matrices = multilabel_confusion_matrix(y_test, y_pred)

for i, genre in enumerate(genre_labels):
    plt.figure(figsize=(4, 4))
    sns.heatmap(conf_matrices[i], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {genre}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

for i, genre in enumerate(genre_labels[:5]):  # top 5 genres for illustration
    fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{genre} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves for Selected Genres')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()
