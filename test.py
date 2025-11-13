# test.py
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
import sys

def main():
    try:
        clf = joblib.load("models/savedmodel.pth")
    except FileNotFoundError:
        print("Model file not found. Run train.py first.", file=sys.stderr)
        sys.exit(1)

    data = np.load("models/test_data.npz")
    X_test = data["X_test"]
    y_test = data["y_test"]

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
