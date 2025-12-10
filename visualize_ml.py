import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import os

# -------------------
# LOAD TRAINED MODEL
# -------------------
with open("dda_model.pkl", "rb") as f:
    data = pickle.load(f)
    model: RandomForestRegressor = data["model"]
    scaler = data["scaler"]
    feature_columns = data["feature_columns"]

print("Model and scaler loaded successfully!")

# -------------------
# LOAD SYNTHETIC DATA FOR VISUALIZATION
# -------------------
# If you still have synthetic data from training, load it.
# If not, create small fake data for plotting.
try:
    df = pd.read_csv("synthetic_training_data.csv")
except:
    print("Synthetic data not found — generating small sample.")
    df = pd.DataFrame(np.random.rand(300, len(feature_columns)), columns=feature_columns)
    df["actual_rt"] = np.random.uniform(0.2, 1.0, size=len(df))

# Predict using the model
X_scaled = scaler.transform(df[feature_columns])
df["predicted_rt"] = model.predict(X_scaled)

# -------------------
# PLOT 1: FEATURE IMPORTANCE
# -------------------
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

sns.barplot(x=importances[indices][:12],
            y=np.array(feature_columns)[indices][:12],
            palette="viridis")

plt.title("Top 12 Most Important Features in Random Forest Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# -------------------
# PLOT 2: PREDICTED vs ACTUAL REACTION TIME
# -------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["actual_rt"], y=df["predicted_rt"], alpha=0.5)

plt.plot([0, 1.5], [0, 1.5], "r--", label="Perfect Prediction Line")
plt.xlabel("Actual Reaction Time (s)")
plt.ylabel("Predicted Reaction Time (s)")
plt.title("Predicted vs Actual Reaction Times")
plt.legend()
plt.savefig("prediction_vs_actual.png")
plt.tight_layout()
plt.show()

# -------------------
# PLOT 3: DISTRIBUTION OF PREDICTED REACTION TIMES
# -------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["predicted_rt"], kde=True, bins=20, color='blue')

plt.title("Distribution of Predicted Reaction Times")
plt.xlabel("Predicted Reaction Time (s)")
plt.ylabel("Frequency")
plt.savefig("predicted_rt_distribution.png")
plt.tight_layout()
plt.show()

# -------------------
# PLOT 4: IF YOU LOGGED GAMEPLAY, PLOT DIFFICULTY OVER TIME
# -------------------
# Expecting a CSV like: gameplay_log.csv
# With columns: click_index, difficulty, reaction_time, accuracy

if "gameplay_log.csv" in df.columns:
    print("Skipping gameplay plotting — file seems incorrect.")
else:
    if "gameplay_log.csv" in [f for f in os.listdir(".")]:
        log = pd.read_csv("gameplay_log.csv")

        plt.figure(figsize=(10, 5))
        plt.plot(log["click_index"], log["difficulty"], color='purple')
        plt.title("Difficulty Level Over Time")
        plt.xlabel("Click Number")
        plt.ylabel("Difficulty")
        plt.grid(True)
        plt.savefig("difficulty_over_time.png")
        plt.show()
    else:
        print("\nNo gameplay_log.csv found — skipping difficulty plot.")
