import pandas as pd
from src.inference import run_inference
from src.insights import analyze_chapter_difficulty


def test_inference_output_columns():
    df = pd.DataFrame({
        "time_spent": [10, 30],
        "score": [40, 80],
        "chapter_order": [1, 2]
    })

    result = run_inference(df)

    assert "completion_probability" in result.columns
    assert "completion_prediction" in result.columns
    assert "risk_flag" in result.columns


def test_chapter_difficulty_generation():
    df = pd.DataFrame({
        "chapter_order": [1, 1, 2, 2],
        "score": [50, 60, 30, 40],
        "time_spent": [20, 25, 40, 45],
        "completion_prediction": [1, 0, 0, 0]
    })

    chapter_stats = analyze_chapter_difficulty(df)

    assert not chapter_stats.empty
    assert "difficulty_score" in chapter_stats.columns
