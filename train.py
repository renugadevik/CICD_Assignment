import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")

# Example of basic feature engineering
df['FeatureSum'] = df.drop(columns=['Disease']).sum(axis=1)

X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the hyperparameters to search
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'bootstrap': [True, False]
}

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1)

# Fit the model
random_search.fit(X, y)

# Get the best model
best_model = random_search.best_estimator_

# Feature selection
feature_importances = best_model.feature_importances_
selected_features = np.argsort(feature_importances)[::-1][:10]  # Select top 10 features

X_selected = X[:, selected_features]

# Refit the model with selected features
best_model.fit(X_selected, y)

with open("model.pkl", 'wb') as f:
    pickle.dump(best_model, f)
