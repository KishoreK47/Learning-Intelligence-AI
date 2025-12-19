import os
import joblib
import pandas as pd

# Resolve project root robustly (works locally + Streamlit Cloud)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "completion_model.joblib")


def run_inference(df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Ensure completion_model.joblib is present in the models/ directory."
        )

    model = joblib.load(MODEL_PATH)

    features = df[["time_spent", "score", "chapter_order"]]

    completion_prob = model.predict_proba(features)[:, 1]

    df["completion_probability"] = completion_prob
    df["completion_prediction"] = (completion_prob >= 0.5).astype(int)
    df["risk_flag"] = df["completion_probability"].apply(
        lambda x: "HIGH" if x < 0.4 else "LOW"
    )

    return df
