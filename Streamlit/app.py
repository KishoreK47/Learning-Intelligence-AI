import streamlit as st
import pandas as pd

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

from inference import run_inference
from insights import analyze_chapter_difficulty, generate_insights


st.set_page_config(page_title="Learning Intelligence AI Tool")

st.title("📘 Learning Intelligence AI Tool")

uploaded_file = st.file_uploader("Upload learner CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully")

    df = run_inference(df)
    chapter_stats = analyze_chapter_difficulty(df)
    insights = generate_insights(df, chapter_stats)

    st.subheader("🔍 AI Insights")
    for ins in insights:
        st.write("•", ins)

    st.subheader("⚠️ High Risk Students")
    st.write(df[df["risk_flag"] == "HIGH"]["student_id"].unique())

    st.subheader("📊 Chapter Difficulty")
    st.dataframe(chapter_stats)

    st.download_button(
        "Download Results",
        data=df.to_csv(index=False),
        file_name="results.csv"
    )
