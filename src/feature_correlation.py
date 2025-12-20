import pandas as pd
import matplotlib.pyplot as plt

# Load training-style data
df = pd.read_csv(r"C:\Users\kisho\Downloads\learning_intelligence_ai\data\sample_input.csv")

# Select relevant numeric features
features = [
    "time_spent",
    "score",
    "chapter_order",
    "completed"
]

corr = df[features].corr()

print("📊 Feature Correlation Matrix:\n")
print(corr)

# Plot correlation heatmap (simple, no seaborn)
plt.figure(figsize=(6, 4))
plt.imshow(corr, interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(features)), features, rotation=45)
plt.yticks(range(len(features)), features)
plt.title("Feature Correlation with Completion")
plt.tight_layout()
plt.show()
