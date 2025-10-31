import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import numpy as np
import matplotlib.pyplot as plt

# --------- LOAD DATA ---------
with open('data_landmarks.pickle', 'rb') as f:
    dataset = pickle.load(f)

X = dataset['data']
y = dataset['labels']

print(f"✅ Loaded dataset with {len(X)} samples and {len(set(y))} classes.")

# --------- ENCODE LABELS ---------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --------- STANDARDIZE FEATURES ---------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------- SPLIT TRAIN/TEST ---------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --------- TRAIN MODEL ---------
clf = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    verbose=True
)
clf.fit(X_train, y_train)

# --------- EVALUATE ---------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n📊 Evaluation Metrics:")
print(f"Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --------- CONFUSION MATRIX ---------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
plt.figure(figsize=(10, 8))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()

# --------- SAVE MODEL & SCALER ---------
model_file = 'model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump({'model': clf, 'scaler': scaler, 'label_encoder': le}, f)

print(f"\n✅ Training complete! Model saved as '{model_file}'")
