import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

# CONFIGURATION
UCI_ID = 390
TARGET_COL = "Annual Return.1"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Pre-optimized trial configuration using descriptive names
TRIALS = [
    {"learning_rate": 0.01, "iterations": 500},
    {"learning_rate": 0.01, "iterations": 1000},
    {"learning_rate": 0.05, "iterations": 1000},
    {"learning_rate": 0.05, "iterations": 2000}
]

LOG_FILE = "part1_custom_gd_trials_log.csv"
#HELPER DATA CLEANING FUNCTION

def parse_percent_series(series_input):
    """Cleans a pandas Series containing numeric, currency, or percentage strings."""
    string_series = pd.Series(series_input).astype(str).str.strip()
    missing_placeholders = {'': np.nan, 'nan': np.nan, 'None': np.nan}
    cleaned_series = string_series.replace(missing_placeholders)
    cleaned_series = cleaned_series.str.replace(',', '', regex=False)
    cleaned_series = cleaned_series.str.replace(r'[\$]', '', regex=True)
    is_percentage_mask = cleaned_series.str.contains('%', na=False)
    numeric_strings = cleaned_series.str.rstrip('%')
    numeric_output = pd.to_numeric(numeric_strings, errors='coerce')
    if is_percentage_mask.any():
        numeric_output.loc[is_percentage_mask] = numeric_output.loc[is_percentage_mask] / 100.0
    return numeric_output

# 1. LOAD DATA VIA UCI API
repo = fetch_ucirepo(id=UCI_ID)
if hasattr(repo.data, "original") and repo.data.original is not None:
    df = repo.data.original.copy()
else:
    df = pd.concat([repo.data.features.reset_index(drop=True), 
                    repo.data.targets.reset_index(drop=True)], axis=1)

# 2. PRE-PROCESSING
df = df.drop_duplicates()

if TARGET_COL in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET_COL]):
    df[TARGET_COL] = parse_percent_series(df[TARGET_COL])

for col in df.columns:
    if col != TARGET_COL and not pd.api.types.is_numeric_dtype(df[col]):
        parsed = parse_percent_series(df[col])
        if parsed.notna().sum() > (0.5 * len(parsed)):
            df[col] = parsed

categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate Features and Target (Imputation happens POST-split to prevent leakage)
X_all = df.drop(columns=[TARGET_COL]).copy()
y_all = pd.Series(df[TARGET_COL]).copy()

# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# 4. PROFESSOR RECOMMENDED MEAN IMPUTATION (TRAIN-DRIVEN)
X_train_nan = X_train.replace(0, np.nan)
feature_means = X_train_nan.mean()

X_train = X_train.replace(0, np.nan).fillna(feature_means)
X_test = X_test.replace(0, np.nan).fillna(feature_means)

y_train_nan = y_train.replace(0, np.nan)
target_mean = y_train_nan.mean()

y_train = y_train.replace(0, np.nan).fillna(target_mean).values.astype(float)
y_test = y_test.replace(0, np.nan).fillna(target_mean).values.astype(float)

# 5. STANDARDIZE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. CUSTOM LINEAR REGRESSION GD CLASS
class LinearRegressionGD:
    def __init__(self, lr=0.05, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d, dtype=float)
        self.b = 0.0
        self.loss_history = []

        for _ in range(self.n_iter):
            y_pred = X @ self.w + self.b
            error = y_pred - y
            self.loss_history.append(float(np.mean(error ** 2)))

            dw = (2.0 / n) * (X.T @ error)
            db = (2.0 / n) * np.sum(error)
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict(self, X):
        return X @ self.w + self.b

# 7. EXECUTE RUNS & TUNING LOGS
trial_logs = []
best_mse = float("inf")
best_model = None

plt.figure(figsize=(8, 5))

for idx, trial in enumerate(TRIALS):
    model = LinearRegressionGD(lr=trial["learning_rate"], n_iter=trial["iterations"])
    model.fit(X_train_scaled, y_train)
    
    p_train = model.predict(X_train_scaled)
    p_test = model.predict(X_test_scaled)
    
    tr_mse = mean_squared_error(y_train, p_train)
    te_mse = mean_squared_error(y_test, p_test)
    
    trial_logs.append({
        "trial": idx+1, 
        "learning_rate": trial["learning_rate"], 
        "iterations": trial["iterations"],
        "train_mse": tr_mse, 
        "test_mse": te_mse
    })
    
    plt.plot(model.loss_history, label=f"Trial {idx+1}: LR={trial['learning_rate']} Iter={trial['iterations']}")
    
    if te_mse < best_mse:
        best_mse = te_mse
        best_model = model

# Save Log File
pd.DataFrame(trial_logs).to_csv(LOG_FILE, index=False)

# Plot 1: Convergence
plt.title("Part 1: Custom GD MSE Loss Curve Evolution")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("part1_loss_comparison.png", dpi=150)
plt.close()

# Plot 2: Feature Visualization
best_feat_idx = np.argmax(np.abs(best_model.w))
best_feat_name = X_all.columns[best_feat_idx]

plt.figure(figsize=(7, 5))
plt.scatter(X_test.iloc[:, best_feat_idx], y_test, color='darkblue', alpha=0.6, label="Actual Targets")
plt.scatter(X_test.iloc[:, best_feat_idx], best_model.predict(X_test_scaled), color='red', alpha=0.4, label="Predicted Targets")
plt.title(f"Target vs. Most Crucial Feature: {best_feat_name}")
plt.xlabel(best_feat_name)
plt.ylabel(TARGET_COL)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("part1_feature_vs_target.png", dpi=150)
plt.close()

# 8. OUTPUT REPORT METRICS
final_pred = best_model.predict(X_test_scaled)
print("\n=== PART 1 EVALUATION STATISTICS ===")
print(f"Optimal Model Config Found: LR={best_model.lr}, Iterations={best_model.n_iter}")
print(f"Final Test Mean Squared Error (MSE): {mean_squared_error(y_test, final_pred):.6f}")
print(f"Final Test Coefficient of Determination (R2): {r2_score(y_test, final_pred):.4f}")
print(f"Final Test Explained Variance Score: {explained_variance_score(y_test, final_pred):.4f}")
print("\n--- Weight Coefficients Vector ---")
for col, coef in zip(X_all.columns, best_model.w):
    print(f" Feature '{col}': {coef:.6f}")
print(f" Intercept Bias (b): {best_model.b:.6f}")