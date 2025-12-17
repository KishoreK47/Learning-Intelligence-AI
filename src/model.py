import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os


MODEL_PATH = r"C:\Users\kisho\Downloads\learning_intelligence_ai\models\completion_model.joblib"


def train_and_save_model(data_path= r"C:\Users\kisho\Downloads\learning_intelligence_ai\data\sample_input.csv"):
    df = pd.read_csv(data_path)

    # Features and target
    X = df[["time_spent", "score", "chapter_order"]]
    y = df["completed"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluation (basic sanity check)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"✅ Model trained successfully")
    print(f"📊 Validation Accuracy: {acc:.2f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"💾 Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
