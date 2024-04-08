import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

#rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2)  # Increase the number of estimators and tune max_depth
#gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)  # Increase the number of estimators and tune max_depth
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()

# Adding a VotingClassifier classifier with multiple classifiers to boost the score.
model = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb)
    ],
).fit(X, y)
print("In Training  the model")
with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
