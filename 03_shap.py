# %% Imports
from pathlib import Path
from time import perf_counter
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import sklearn
import imblearn
import joblib

from utils import DataLoader, build_pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, ConfusionMatrixDisplay
)

# %% Configuration
RANDOM_STATE = 2021
EXPLANATION_SIZE = 60
BACKGROUND_SIZE = 100
save_dir = Path('results/shap')
save_dir.mkdir(parents=True, exist_ok=True)

# %% Load and Split Data
loader = DataLoader()
data = loader.load_dataset()
X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val = (
    loader.get_train_val_test_split(random_state=RANDOM_STATE)
)
print('Train / Validation / Test:', len(X_train), len(X_val), len(X_test))

# Save reproducible split indices without repeating EDA outputs
for name, subset in [('train', X_train), ('validation', X_val), ('test', X_test)]:
    pd.DataFrame({'Dataset Index': subset.index}).to_csv(save_dir / f'{name}_indices.csv', index=False)
dataset_hash = hashlib.sha256(
    pd.util.hash_pandas_object(data, index=True).values.tobytes()
).hexdigest()

# %% Random Forest Hyperparameter Tuning
pipeline = build_pipeline(
    X_train, RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    random_state=RANDOM_STATE
)
parameters = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 5, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}
search = RandomizedSearchCV(
    pipeline, parameters, n_iter=20, scoring='average_precision',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    random_state=RANDOM_STATE, n_jobs=-1
)
search.fit(X_train, y_train)
pd.DataFrame(search.cv_results_).to_csv(save_dir / 'rf_hyperparameter_tuning.csv', index=False)
print('Best Parameters:', search.best_params_)
print('Best CV Average Precision:', search.best_score_)

# %% Select Validation Threshold
val_prob = search.best_estimator_.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_prob)
f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
    precision[:-1] + recall[:-1], 1e-12
)
best_threshold = float(thresholds[np.argmax(f1_values)])
print('Selected Threshold:', best_threshold)

# %% Refit and Evaluate on Untouched Test Set
model = clone(search.best_estimator_)
model.fit(X_train_val, y_train_val)
rf = model.named_steps['classifier']
preprocessor = model.named_steps['preprocessor']
class_index = list(rf.classes_).index(1)
joblib.dump(model, save_dir / 'random_forest_pipeline.joblib')

test_prob = model.predict_proba(X_test)[:, class_index]
test_pred = (test_prob >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
metrics = {
    'Model': 'RandomForest',
    'Accuracy': accuracy_score(y_test, test_pred),
    'Balanced Accuracy': balanced_accuracy_score(y_test, test_pred),
    'Precision': precision_score(y_test, test_pred, zero_division=0),
    'Recall': recall_score(y_test, test_pred, zero_division=0),
    'F1': f1_score(y_test, test_pred, zero_division=0),
    'ROC AUC': roc_auc_score(y_test, test_prob),
    'Average Precision': average_precision_score(y_test, test_prob),
    'Prevalence': float(y_test.mean()),
    'Positive Support': int(y_test.sum()),
    'Threshold': best_threshold,
    'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
}
pd.DataFrame([metrics]).to_csv(save_dir / 'RandomForest_metrics.csv', index=False)
print('\nTest Performance:\n', pd.DataFrame([metrics]).to_string(index=False))
ConfusionMatrixDisplay.from_predictions(y_test, test_pred, labels=[0, 1])
plt.savefig(save_dir / 'RandomForest_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# %% Prepare Shared SHAP Sample
feature_names = preprocessor.get_feature_names_out().tolist()
def transform_features(X):
    return pd.DataFrame(preprocessor.transform(X), columns=feature_names, index=X.index)

X_train_processed = transform_features(X_train_val)
X_test_processed = transform_features(X_test)
sample_size = min(EXPLANATION_SIZE, len(X_test))
if sample_size == len(X_test):
    positions = np.arange(len(X_test))
else:
    positions, _ = train_test_split(
        np.arange(len(X_test)), train_size=sample_size,
        stratify=y_test, random_state=RANDOM_STATE
    )
    positions = np.sort(positions)
X_explain = X_test_processed.iloc[positions]
y_explain = y_test.iloc[positions]
pd.DataFrame({'Dataset Index': X_explain.index}).to_csv(
    save_dir / 'shared_explanation_indices.csv', index=False
)

# %% Tree SHAP (Positive-Class Probability)
background = X_train_processed.sample(
    n=min(BACKGROUND_SIZE, len(X_train_processed)), random_state=RANDOM_STATE
)
start = perf_counter()
explainer = shap.TreeExplainer(
    rf, data=background, feature_perturbation='interventional',
    model_output='probability'
)
values = explainer.shap_values(X_explain, check_additivity=False)
shap_runtime = perf_counter() - start
stroke_shap = (np.asarray(values[class_index]) if isinstance(values, list)
               else np.asarray(values)[:, :, class_index])
base_value = float(np.asarray(explainer.expected_value)[class_index])
assert stroke_shap.shape == X_explain.shape
assert list(X_explain.columns) == feature_names
assert rf.n_features_in_ == len(feature_names)
np.save(save_dir / 'rf_shap_values.npy', stroke_shap)
np.save(save_dir / 'rf_background.npy', background.to_numpy())
pd.DataFrame({'Dataset Index': background.index}).to_csv(
    save_dir / 'rf_background_indices.csv', index=False
)


# %% Verify SHAP Additivity
SHAP_TOLERANCE = 1e-3
model_prob = rf.predict_proba(X_explain.to_numpy())[:, class_index]
reconstructed = base_value + stroke_shap.sum(axis=1)
residuals = np.abs(reconstructed - model_prob)
verification = pd.DataFrame({
    'Dataset Index': X_explain.index,
    'Base Value': base_value,
    'SHAP Sum': stroke_shap.sum(axis=1),
    'Reconstructed Probability': reconstructed,
    'Model Probability': model_prob,
    'Absolute Residual': residuals
})
verification.to_csv(
    save_dir / 'rf_shap_verification.csv',
    index=False
)
max_residual = float(residuals.max())
print('Maximum SHAP Residual:', max_residual)
print('Additivity Passed:', max_residual <= SHAP_TOLERANCE)

if max_residual > SHAP_TOLERANCE:
    raise RuntimeError(
        f'SHAP additivity residual {max_residual:.6f} '
        f'exceeds tolerance {SHAP_TOLERANCE:.6f}'
    )


# %% Task 3: Local Force Plot
positive_positions = np.flatnonzero(y_explain.to_numpy() == 1)
if len(positive_positions) == 0:
    raise ValueError('No stroke-positive patient in the explanation sample.')
local_index = int(positive_positions[0])
patient_index = int(X_explain.index[local_index])
actual = int(y_explain.iloc[local_index])
probability = float(model_prob[local_index])
predicted = int(probability >= best_threshold)
case_type = ('True Positive' if actual == 1 and predicted == 1 else
             'False Negative' if actual == 1 else
             'False Positive' if predicted == 1 else 'True Negative')
local_values = stroke_shap[local_index]
local_features = X_explain.iloc[local_index]
X_test.loc[[patient_index]].to_csv(save_dir / 'selected_patient_original_values.csv', index=False)

shap.force_plot(base_value, local_values, local_features,
                matplotlib=True, show=False, figsize=(18, 4))
plt.savefig(save_dir / 'rf_local_force_plot.png', dpi=300, bbox_inches='tight')
plt.close()
shap.save_html(str(save_dir / 'rf_local_force_plot.html'),
               shap.force_plot(base_value, local_values, local_features))

# %% Task 3: Local Contributions
local_table = pd.DataFrame({
    'Feature': feature_names,
    'Patient Value': local_features.to_numpy(),
    'SHAP Value': local_values
})
local_table['Absolute SHAP'] = local_table['SHAP Value'].abs()
local_table['Direction'] = np.select(
    [local_table['SHAP Value'] > 0, local_table['SHAP Value'] < 0],
    ['Increases', 'Decreases'], default='Neutral'
)
local_table = local_table.sort_values('Absolute SHAP', ascending=False)
local_table.to_csv(save_dir / 'rf_local_contributions.csv', index=False)
top3 = local_table.head(3)
remaining = float(local_table.iloc[3:]['SHAP Value'].sum())
pd.DataFrame([{
    'Patient Index': patient_index, 'Actual': actual, 'Predicted': predicted,
    'Case Type': case_type, 'Probability': probability,
    'Threshold': best_threshold, 'Base Value': base_value,
    'Top 3 SHAP Sum': float(top3['SHAP Value'].sum()),
    'Remaining SHAP Sum': remaining,
    'Reconstructed Probability': float(reconstructed[local_index]),
    'Absolute Residual': float(residuals[local_index]),
    'Selection Rule': 'First stroke-positive case in seeded held-out sample'
}]).to_csv(save_dir / 'rf_selected_case.csv', index=False)
print('\nLocal Patient:', patient_index, '|', case_type, '| Probability:', probability)
print(top3[['Feature', 'SHAP Value', 'Direction']].to_string(index=False))

# %% Task 4: Global Beeswarm and Importance
shap.summary_plot(stroke_shap, X_explain, max_display=15, show=False)
plt.xlabel('SHAP contribution to stroke=1 probability')
plt.savefig(save_dir / 'rf_global_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()

importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean Absolute SHAP': np.abs(stroke_shap).mean(axis=0)
}).sort_values('Mean Absolute SHAP', ascending=False).reset_index(drop=True)
importance.insert(0, 'Rank', np.arange(1, len(importance) + 1))
importance.to_csv(save_dir / 'rf_feature_importance.csv', index=False)
shap.summary_plot(stroke_shap, X_explain, plot_type='bar', max_display=15, show=False)
plt.savefig(save_dir / 'rf_global_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print('\nTop 5 Global Features:\n', importance.head(5).to_string(index=False))

# %% Task 4: Top 5 Feature Patterns
# Binary and one-hot: 0 vs 1. Continuous: <= median vs > median.
continuous_features = {'age', 'avg_glucose_level', 'bmi'}
binary_features = {'hypertension', 'heart_disease'}
patterns = []
for feature in importance.head(5)['Feature']:
    j = feature_names.index(feature)
    values = X_explain[feature].to_numpy()
    contributions = stroke_shap[:, j]
    if feature in continuous_features:
        feature_type = 'Continuous'
        cutoff = float(np.median(values))
        low_mask, high_mask = values <= cutoff, values > cutoff
        grouping = f'<= {cutoff:.2f} vs > {cutoff:.2f}'
    else:
        feature_type = 'Binary' if feature in binary_features else 'One-hot category'
        low_mask, high_mask = values == 0, values == 1
        grouping = '0 vs 1 (absent vs present for one-hot)'

    low_n, high_n = int(low_mask.sum()), int(high_mask.sum())
    low_mean = float(contributions[low_mask].mean()) if low_n else np.nan
    high_mean = float(contributions[high_mask].mean()) if high_n else np.nan
    if not low_n or not high_n:
        pattern = 'Insufficient observations for comparison'
    elif np.isclose(low_mean, high_mean):
        pattern = 'Similar mean SHAP contributions'
    elif high_mean > low_mean:
        pattern = 'High/1 group has more positive mean SHAP contribution'
    else:
        pattern = 'High/1 group has more negative mean SHAP contribution'

    patterns.append({
        'Feature': feature, 'Feature Type': feature_type,
        'Grouping': grouping, 'Low/0 Count': low_n, 'High/1 Count': high_n,
        'Low/0 Mean SHAP': low_mean, 'High/1 Mean SHAP': high_mean,
        'Mean Absolute SHAP': float(np.abs(contributions).mean()),
        'Min SHAP': float(contributions.min()),
        'Max SHAP': float(contributions.max()),
        'Observed Pattern': pattern
    })
pd.DataFrame(patterns).to_csv(save_dir / 'rf_top5_patterns.csv', index=False)

# %% Task 4: Local vs Global Comparison
comparison = importance.copy()
comparison['Local SHAP'] = comparison['Feature'].map(dict(zip(feature_names, local_values)))
comparison['Local Absolute SHAP'] = comparison['Local SHAP'].abs()
comparison['Local Rank'] = comparison['Local Absolute SHAP'].rank(
    method='min', ascending=False
).astype(int)
comparison.to_csv(save_dir / 'rf_local_global_comparison.csv', index=False)

# For the report: local rank, direction and whether it agrees with global grouping.
local_vs_global = pd.DataFrame(patterns).merge(
    comparison[['Feature', 'Rank', 'Local Rank', 'Local SHAP']],
    on='Feature', how='left'
)
local_vs_global['Local Direction'] = np.select(
    [local_vs_global['Local SHAP'] > 0, local_vs_global['Local SHAP'] < 0],
    ['Increases model output', 'Decreases model output'], default='Neutral'
)
local_vs_global.to_csv(save_dir / 'rf_top5_local_global.csv', index=False)

# %% Save Reproducibility Information
configuration = {
    'Model': 'RandomForestClassifier', 'Explainer': 'TreeExplainer',
    'Random State': RANDOM_STATE, 'CV Folds': 5,
    'CV Scoring': 'average_precision', 'Search Iterations': 20,
    'Best Parameters': search.best_params_,
    'Best CV Average Precision': float(search.best_score_),
    'Validation Threshold': best_threshold,
    'Train Rows': len(X_train), 'Validation Rows': len(X_val),
    'Test Rows': len(X_test), 'Final Training Rows': len(X_train_val),
    'Explanation Rows': len(X_explain),
    'Explanation Positive Cases': int(y_explain.sum()),
    'Explanation Selection': 'Stratified held-out test sample',
    'Background Rows': len(background),
    'Background Source': 'Final training data, before oversampling',
    'Feature Representation': 'Numerical imputation and one-hot encoding',
    'Feature Perturbation': 'interventional',
    'SHAP Output': 'Stroke=1 predicted probability',
    'SHAP Class Index': class_index,
    'SHAP Runtime Seconds': float(shap_runtime),
    'Maximum SHAP Residual': float(residuals.max()),
    'Selected Patient Index': patient_index,
    'Dataset SHA256 Fingerprint': dataset_hash,
    'Python Version': platform.python_version(),
    'SHAP Version': shap.__version__,
    'Scikit-learn Version': sklearn.__version__,
    'Imbalanced-learn Version': imblearn.__version__,
    'Pandas Version': pd.__version__,
    'Numpy Version': np.__version__,
    'Probability Calibration Assessed': False,
    'Operating System': platform.platform(),
    'Dataset Path': 'data/healthcare-dataset-stroke-data.csv',
    'Sampling Seed': RANDOM_STATE,
    'Background Seed': RANDOM_STATE,
    'Explainer Additivity Tolerance': SHAP_TOLERANCE,
    'Selected Patient Selection Rule': 'First stroke-positive case in seeded held-out sample',
    'Feature Count': len(feature_names),
    'Explainer Version': shap.__version__
}
with open(save_dir / 'rf_configuration.json', 'w', encoding='utf-8') as file:
    json.dump(configuration, file, indent=4)
# Save an environment file and feature names for reproducibility.
versions = {
    'python': platform.python_version(), 'numpy': np.__version__,
    'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
    'imbalanced-learn': imblearn.__version__, 'shap': shap.__version__,
    'matplotlib': plt.matplotlib.__version__, 'joblib': joblib.__version__
}
with open(save_dir / 'environment_versions.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(f'{name}=={version}' for name, version in versions.items()) + '\n')
pd.DataFrame({'Feature': feature_names}).to_csv(save_dir / 'rf_feature_names.csv', index=False)
print('\nTask 2-4 completed. Results saved to:', save_dir)
