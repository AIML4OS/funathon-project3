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
#   - Outlier removal
#   - Scaling
#   - One-hot encoding
# ============================================

# Simple outlier removal using IQR on numeric columns
numeric_cols = [col for col in df.columns if col.startswith("num_")]

Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
         (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

df = df.loc[mask].reset_index(drop=True)

# Split features / target
X = df.drop(columns="target")
y = df["target"]

# Column types
categorical_cols = ["category"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)


# %%
# ============================================
# STEP 3 — Train / test split, model fitting,
#          and performance evaluation
# ============================================

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)

# Full modeling pipeline
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )),
    ]
)

# Fit model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"R² score: {r2:.3f}")
# %%
