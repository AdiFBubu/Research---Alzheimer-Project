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

# --- Cross-validation pipeline ---
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Pipeline: impute -> variance filter -> scale -> classifier
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('var', VarianceThreshold(threshold=0.0)),
    ('scaler', StandardScaler(with_mean=True)),
    ('clf', LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=5000,
        class_weight='balanced',
        n_jobs=-1
    ))
])

# Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_aucs = []
fold_reports = []
# For feature aggregation across folds
kept_feature_names_per_fold = []
coef_per_fold = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Fit pipeline on training fold
    pipeline.fit(X_train, y_train)

    # Predict on held-out fold
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    fold_aucs.append(auc)

    report = classification_report(y_test, y_pred, output_dict=True)
    fold_reports.append(report)

    # Recover kept features after variance threshold
    var_selector = pipeline.named_steps['var']
    mask_after_impute = var_selector.get_support()  # mask over columns AFTER imputation step
    # The imputer keeps the original column order, so we can map mask to X_train columns
    feature_names_after_impute = X_train.columns[mask_after_impute]

    # Coefficients from classifier correspond to features after var selector and scaler
    clf = pipeline.named_steps['clf']
    coefs = clf.coef_.ravel()

    kept_feature_names_per_fold.append(feature_names_after_impute)
    coef_per_fold.append(coefs)

    print(f"Fold {fold_idx}: ROC AUC = {auc:.4f}")

# Aggregate AUC
mean_auc = np.mean(fold_aucs)
std_auc = np.std(fold_aucs)
print(f"Mean ROC AUC across {n_splits} folds: {mean_auc:.4f} ± {std_auc:.4f}")

# Aggregate classification report (macro-averaged over folds)
def average_reports(reports):
    # Average precision/recall/f1 for 'macro avg' and 'weighted avg'
    keys = ['precision', 'recall', 'f1-score']
    agg = {'macro avg': {}, 'weighted avg': {}}
    for avg_key in agg.keys():
        for k in keys:
            agg[avg_key][k] = np.mean([r[avg_key][k] for r in reports])
        agg[avg_key]['support'] = np.sum([r[avg_key]['support'] for r in reports])
    # Per-class averaging if present (0 and 1)
    for cls in ['0', '1']:
        if all(cls in r for r in reports):
            agg[cls] = {k: np.mean([r[cls][k] for r in reports]) for k in keys}
            agg[cls]['support'] = np.sum([r[cls]['support'] for r in reports])
    return agg

avg_report = average_reports(fold_reports)
print("Averaged classification report (macro/weighted):")
for avg_key in ['macro avg', 'weighted avg']:
    pr = avg_report[avg_key]['precision']
    rc = avg_report[avg_key]['recall']
    f1 = avg_report[avg_key]['f1-score']
    print(f"{avg_key}: precision={pr:.4f}, recall={rc:.4f}, f1={f1:.4f}, support={avg_report[avg_key]['support']}")

# Aggregate feature weights across folds
# We'll compute:
# - Mean absolute coefficient per feature
# - Selection frequency: in how many folds the feature survived VarianceThreshold
from collections import defaultdict

abs_coef_sums = defaultdict(float)
coef_sums = defaultdict(float)
selection_counts = defaultdict(int)

for fold_feature_names, fold_coefs in zip(kept_feature_names_per_fold, coef_per_fold):
    for fname, w in zip(fold_feature_names, fold_coefs):
        abs_coef_sums[fname] += abs(w)
        coef_sums[fname] += w
        selection_counts[fname] += 1

# Convert to DataFrame
feat_stats = pd.DataFrame({
    'feature': list(abs_coef_sums.keys()),
    'mean_abs_weight': [abs_coef_sums[f] / selection_counts[f] for f in abs_coef_sums.keys()],
    'mean_weight': [coef_sums[f] / selection_counts[f] for f in coef_sums.keys()],
    'selection_freq': [selection_counts[f] / n_splits for f in selection_counts.keys()],
})

# Sort by mean_abs_weight descending, then selection frequency
feat_stats = feat_stats.sort_values(by=['mean_abs_weight', 'selection_freq'], ascending=False)

# Take top N features
N = 100
top_feats = feat_stats.head(N)

print(f"Top {N} features across folds (by mean absolute weight):")
print(top_feats.to_string(index=False))

# If you only want genes (exclude age):
top_gene_feats = top_feats[~top_feats['feature'].eq('age')]
print(f"Top {min(N, len(top_gene_feats))} gene features across folds:")
print(top_gene_feats.to_string(index=False))