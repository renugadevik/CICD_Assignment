import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import pickle
import numpy as np

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

with open("model.pkl", 'wb') as f:
    pickle.dump(best_model, f)

