import pandas as pd


def analyze_chapter_difficulty(df):
    chapter_stats = df.groupby("chapter_order").agg(
        avg_score=("score", "mean"),
        avg_time_spent=("time_spent", "mean"),
        dropout_rate=("completion_prediction", lambda x: 1 - x.mean())
    ).reset_index()

    # Difficulty score (simple weighted logic)
    chapter_stats["difficulty_score"] = (
        (1 - chapter_stats["avg_score"] / 100) * 0.5 +
        (chapter_stats["dropout_rate"]) * 0.3 +
        (chapter_stats["avg_time_spent"] / chapter_stats["avg_time_spent"].max()) * 0.2
    )

    return chapter_stats.sort_values(by="difficulty_score", ascending=False)


def generate_insights(df, chapter_stats):
    insights = []

    high_risk_count = (df["risk_flag"] == "HIGH").sum()
    insights.append(
        f"{high_risk_count} students are identified as high risk for course dropout."
    )

    top_chapter = chapter_stats.iloc[0]
    insights.append(
        f"Chapter {int(top_chapter['chapter_order'])} appears to be the most difficult "
        f"with a dropout rate of {top_chapter['dropout_rate']:.2f}."
    )

    insights.append(
        "Lower assessment scores and reduced engagement time are strong indicators "
        "of non-completion."
    )

    return insights
