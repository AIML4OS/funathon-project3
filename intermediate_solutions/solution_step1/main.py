# %%
# ============================================
# STEP 1 — Generate synthetic regression data
# ============================================

import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Reproducibility
RANDOM_STATE = 20230516
np.random.seed(RANDOM_STATE)

# Generate numeric regression data
X_num, y = make_regression(
    n_samples=2000,
    n_features=5,
    noise=10.0,
    random_state=RANDOM_STATE
)

# Create a DataFrame
df = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(X_num.shape[1])])

# Add a categorical feature
df["category"] = np.random.choice(["A", "B", "C"], size=len(df))

# Add target
df["target"] = y


# %%
# ============================================
# STEP 2 — Preprocessing
# ============================================

# YOUR CODE HERE


# %%
# ============================================
# STEP 3 — Train / test split, model fitting,
#          and performance evaluation
# ============================================

# YOUR CODE HERE
