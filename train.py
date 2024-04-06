
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

#model = LogisticRegression().fit(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(max_iter=1000).fit(X_scaled, y)

#model = LogisticRegression(max_iter=1000)  # Increase from default (100)
#model.fit(X, y)
#model.fit(X, y)

with open("/app/data/model.pkl", 'wb') as f:
    pickle.dump(model, f)
~                                 
