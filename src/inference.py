import os
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "completion_model.joblib")


def run_inference(df):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    features = df[["time_spent", "score", "chapter_order"]]
    probs = model.predict_proba(features)[:, 1]

    df["completion_probability"] = probs
    df["completion_prediction"] = (probs >= 0.5).astype(int)
    df["risk_flag"] = df["completion_probability"].apply(
        lambda x: "HIGH" if x < 0.4 else "LOW"
    )

    return df
