import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
le = LabelEncoder()
y = le.fit_transform(y)

# Define the parameter grid
param_grid = {
    'n_estimators': [3000, 5000],
    'max_depth': [30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# Perform GridSearchCV
grid_search.fit(X, y)

# Get the best estimator
best_rf = grid_search.best_estimator_

# Train the model using the best estimator
model = best_rf.fit(X, y)

# Save the model
with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
print("End train")
