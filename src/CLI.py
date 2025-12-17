import argparse
import json
import os
import pandas as pd

from inference import run_inference
from insights import analyze_chapter_difficulty, generate_insights


def main(input_path):
    print("🚀 Learning Intelligence AI Tool Started")

    # Load data
    df = pd.read_csv(input_path)
    print(f"📥 Loaded data from {input_path}")

    # Run model inference
    df = run_inference(df)
    print("🤖 Model inference completed")

    # Chapter difficulty analysis
    chapter_stats = analyze_chapter_difficulty(df)
    print("📊 Chapter difficulty analysis completed")

    # Generate insights
    insights = generate_insights(df, chapter_stats)

    # Output summary
    print("\n===== AI INSIGHTS =====")
    for insight in insights:
        print(f"- {insight}")

    # Save output
    os.makedirs("output", exist_ok=True)
    output_data = {
        "high_risk_students": df[df["risk_flag"] == "HIGH"]["student_id"].unique().tolist(),
        "chapter_difficulty": chapter_stats.to_dict(orient="records"),
        "insights": insights
    }

    with open("output/results.json", "w") as f:
        json.dump(output_data, f, indent=4)

    print("\n💾 Results saved to output/results.json")
    print("✅ AI Tool execution completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning Intelligence AI Tool")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file"
    )

    args = parser.parse_args()
    main(args.input)
