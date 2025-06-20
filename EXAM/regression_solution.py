import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Read the dataset
dataframe = pd.read_csv('dataset_01.csv', sep=';')

# Create a pipeline with scaling and ridge regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge())
])

# Define parameter grid for grid search
param_grid = {
    'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='neg_max_error',
    n_jobs=-1
)

# Fit the grid search
grid_search.fit(dataframe[['x1', 'x2', 'x3']], dataframe['y'])

# Get the best model
best_model = grid_search.best_estimator_

# Perform cross-validation with the best model
scores = abs(
    cross_val_score(
        best_model,
        dataframe[['x1', 'x2', 'x3']],
        dataframe['y'],
        cv=5,
        scoring='neg_max_error'
    )
)

# Calculate mean max_error
mean_max_error = np.mean(scores)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Cross-validation scores (max_error): {scores}")
print(f"Mean max_error: {mean_max_error:.4f}")

# Check if the requirement is met
if mean_max_error <= 0.22:
    print("\nSuccess! The mean max_error is less than or equal to 0.22")
else:
    print("\nThe mean max_error exceeds 0.22. Further optimization may be needed.") 