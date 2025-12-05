import pandas as pd
import numpy as np

import pandas as pd

# Read in your files
csv1 = pd.read_csv('../all_mice_GSE64398.csv')
csv2 = pd.read_csv('../same_mouse_human_gene_mapping.csv')
csv3 = pd.read_csv('../all_mice_GSE64398_metadata.csv')  # must have 'Accession', 'Age_weeks', 'AD'

# Step 1: Row filtering & reordering as before
csv1['ID_REF'] = csv1['ID_REF'].astype(str)
csv2['gene_id_mouse'] = csv2['gene_id_mouse'].astype(str)
ordered_genes = csv2['gene_id_mouse'].tolist()
csv1_filtered = csv1[csv1['ID_REF'].isin(ordered_genes)]
csv1_filtered = csv1_filtered.set_index('ID_REF')
csv1_matched_order = csv1_filtered.reindex(ordered_genes).reset_index()

# Step 2: Map age & AD for each sample column
# Make sure Accession is string and matches columns
csv3['Accession'] = csv3['Accession'].astype(str)
age_dict = dict(zip(csv3['Accession'], csv3['Age_weeks']))
ad_dict  = dict(zip(csv3['Accession'], csv3['AD']))

# For all columns that start with GSM (samples), build age and AD rows
sample_cols = [col for col in csv1_matched_order.columns if col.startswith('GSM')]

age_row = ['age'] + [age_dict.get(col, '') for col in sample_cols]
ad_row  = ['AD']  + [ad_dict.get(col, '')  for col in sample_cols]

# Append the rows to the DataFrame
df_plus_rows = pd.concat([
    csv1_matched_order,
    pd.DataFrame([age_row, ad_row], columns=csv1_matched_order.columns)
], ignore_index=True)

# --- Your existing preprocessing up to "data" ---
# df_plus_rows should already be created as in your snippet.

GENE_ROWS = slice(0, 9633)      # rows 0..9632 are genes
AGE_ROW = 9633
LABEL_ROW = 9634

df = df_plus_rows.copy()

# 1. Split components
gene_expr = df.iloc[GENE_ROWS, :].copy()
age_series = df.iloc[AGE_ROW, :].copy()
label_series = df.iloc[LABEL_ROW, :].copy()

# 2. Ensure numeric gene expression
gene_expr = gene_expr.apply(pd.to_numeric, errors='coerce')

# 3. Build sample rows
X_genes = gene_expr.T
X_genes.columns = [f'gene_{i}' for i in range(X_genes.shape[1])]

age = pd.to_numeric(age_series, errors='coerce')
X_genes['age'] = age

y = label_series.str.strip().str.lower().map({'yes': 1, 'no': 0})

data = X_genes.copy()
data['AD'] = y

# Drop samples with missing label or age
data = data.dropna(subset=['AD', 'age'])

# Drop genes with >20% missing values
threshold = 0.2 * data.shape[0]
cols_to_keep = [c for c in data.columns if c.startswith('gene_') and data[c].isna().sum() <= threshold]
cols_to_keep += ['age', 'AD']
data = data[cols_to_keep]

# Split X, y (do not impute here; do it inside the pipeline to avoid leakage)
X = data.drop(columns=['AD'])
y = data['AD']

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

# X, y as in your current setup (no leakage: imputer inside pipeline)

# Base classifier with elastic net
base_clf = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    max_iter=5000,
    class_weight='balanced',
    n_jobs=-1
)

# We wrap with calibration to improve probability estimates (optional)
# Use 'sigmoid' with cv=3 inside the training set to avoid leakage
calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=3)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('var', VarianceThreshold(threshold=0.0)),
    # Univariate filter to cut dimensionality; K will be tuned
    ('kbest', SelectKBest(score_func=mutual_info_classif)),
    ('scaler', StandardScaler(with_mean=True)),
    ('clf', calibrated_clf)
])

# Parameter grid:
# - kbest__k: number of genes retained (plus age); values scale with dataset size
# - clf__base_estimator__C: regularization strength
# - clf__base_estimator__l1_ratio: elastic net mixing parameter
param_grid = {
    'kbest__k': [100, 300, 600, 1000],
    'clf__estimator__C': [0.05, 0.1, 0.2, 0.5, 1.0],
    'clf__estimator__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
}

# Inner CV for tuning
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=inner_cv,
    n_jobs=-1,
    refit=True
)

# Outer CV for unbiased performance estimate
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

outer_aucs = []
best_params_per_fold = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_params_per_fold.append(grid.best_params_)

    y_proba = grid.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    outer_aucs.append(auc)
    print(f"Outer fold {fold_idx}: AUC = {auc:.4f} | best: {grid.best_params_}")

print(f"Nested CV AUC: {np.mean(outer_aucs):.4f} ± {np.std(outer_aucs):.4f}")
print("Best params per fold:")
for i, bp in enumerate(best_params_per_fold, 1):
    print(i, bp)