# Learning Intelligence AI Tool

## Overview
This project is an AI-powered Learning Intelligence Tool designed to analyze learner behavior and performance data for an internship or training platform.

The tool predicts course completion, detects early dropout risk, identifies difficult chapters, and generates actionable insights for mentors and administrators.

This is a fully executable AI tool (CLI-based), The AI tool can be accessed via CLI or a Streamlit-based desktop interface.

---

## Features
- Course completion prediction (binary classification)
- Early risk detection for potential dropouts
- Chapter difficulty analysis using engagement and performance metrics
- Human-readable insight generation
- JSON-based output reporting

---

## Technology Stack
- Python 3.x
- Scikit-learn
- Pandas
- Joblib

---

## Project Structure
Learning-Intelligence-AI/
├── data/
│ └── sample_input.csv
├── models/
│ └── completion_model.joblib
├── src/
│ ├── app.py
│ ├── CLI.py
│ ├── model.py
│ ├── inference.py
│ ├── insights.py
│ └── generate_data.py
├── tests/
│ └── test_core.py
├── schema.yaml
├── requirements.txt
├── README.md
└── .gitignore
