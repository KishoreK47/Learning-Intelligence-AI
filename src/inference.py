import pandas as pd
import joblib

MODEL_PATH = r"C:\Users\kisho\Downloads\learning_intelligence_ai\models\completion_model.joblib"


def run_inference(df):
    model = joblib.load(MODEL_PATH)

    features = df[["time_spent", "score", "chapter_order"]]

    # Probability of completion
    completion_prob = model.predict_proba(features)[:, 1]

    df["completion_probability"] = completion_prob
    df["completion_prediction"] = (completion_prob >= 0.5).astype(int)

    # Early risk detection
    df["risk_flag"] = df["completion_probability"].apply(
        lambda x: "HIGH" if x < 0.4 else "LOW"
    )

    return df
