import pandas as pd
import numpy as np

np.random.seed(42)

NUM_STUDENTS = 200
NUM_COURSES = 3
CHAPTERS_PER_COURSE = 6

rows = []

for student_id in range(1, NUM_STUDENTS + 1):
    course_id = np.random.randint(1, NUM_COURSES + 1)
    base_ability = np.random.uniform(40, 90)

    for chapter in range(1, CHAPTERS_PER_COURSE + 1):
        time_spent = np.random.uniform(5, 60)
        score = max(0, min(100, base_ability + np.random.normal(0, 10)))

        dropout_prob = 0.4 if score < 50 or time_spent < 10 else 0.1
        completed = 0 if np.random.rand() < dropout_prob else 1

        rows.append([
            student_id,
            course_id,
            chapter,
            round(time_spent, 2),
            round(score, 2),
            completed
        ])

df = pd.DataFrame(rows, columns=[
    "student_id",
    "course_id",
    "chapter_order",
    "time_spent",
    "score",
    "completed"
])

df.to_csv(r"C:\Users\kisho\Downloads\learning_intelligence_ai\data\sample_input.csv", index=False)

print("✅ Synthetic dataset generated: data/sample_input.csv")
print(df.head())
