import pandas as pd
import numpy as np

np.random.seed(99)

NUM_STUDENTS = 120
NUM_COURSES = 2
CHAPTERS_PER_COURSE = 6

rows = []

for student_id in range(1001, 1001 + NUM_STUDENTS):
    course_id = np.random.randint(1, NUM_COURSES + 1)

    # Slightly weaker overall learners (harder test data)
    base_ability = np.random.uniform(30, 75)

    for chapter in range(1, CHAPTERS_PER_COURSE + 1):
        time_spent = np.random.uniform(3, 50)
        score = max(0, min(100, base_ability + np.random.normal(0, 15)))

        # Higher dropout probability (stress test)
        dropout_prob = 0.55 if score < 55 or time_spent < 12 else 0.2
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

df.to_csv(r"C:\Users\kisho\Downloads\learning_intelligence_ai\data\synthetic_test_input.csv", index=False)

print("✅ Synthetic TEST dataset generated: data/synthetic_test_input.csv")
print(df.head())
