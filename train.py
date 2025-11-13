# train.py
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np
import os

os.makedirs("models", exist_ok=True)

def main():
    # Load dataset (will download if not present)
    data = fetch_olivetti_faces()
    X = data.images.reshape((len(data.images), -1))  # flatten the 64x64 images
    y = data.target

    # 70% train, 30% test; stratify to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train DecisionTree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Save model and test split
    joblib.dump(clf, "models/savedmodel.pth")
    np.savez("models/test_data.npz", X_test=X_test, y_test=y_test)

    print("Training complete.")
    print("Model saved to models/savedmodel.pth")
    print("Test split saved to models/test_data.npz")

if __name__ == "__main__":
    main()
