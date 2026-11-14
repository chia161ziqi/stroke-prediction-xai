# %% 1. Imports and configuration
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import imblearn
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from utils import DataLoader, build_pipeline

RANDOM_STATE = 2021
CV_FOLDS = 5
SEARCH_ITERATIONS = 12
BACKGROUND_SIZE = 100
KERNEL_NSAMPLES = 200  # SHAP coalition samples per explained row; increase for more precise estimates
SHAP_TOLERANCE = 0.05  # Kernel SHAP is approximate; used only for a diagnostic, not a strict pass requirement
DATA_PATH = Path('data/healthcare-dataset-stroke-data.csv')
RF_DIR = Path('results/shap')
SAVE_DIR = Path('results/kernel_shap')
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(df, name):
    df.to_csv(SAVE_DIR / name, index=False)


def positive_shap_values(values, n_rows, n_features):
    """Normalise SHAP versions returning 2-D, 3-D, or class-wise lists."""
    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape != (n_rows, n_features):
        raise ValueError(f'Unexpected SHAP shape: {values.shape}, expected {(n_rows, n_features)}')
    return values


def metrics_row(y_true, probability, threshold, name):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, prediction),
        'Balanced Accuracy': balanced_accuracy_score(y_true, prediction),
        'Precision': precision_score(y_true, prediction, zero_division=0),
        'Recall': recall_score(y_true, prediction, zero_division=0),
        'F1': f1_score(y_true, prediction, zero_division=0),
        'ROC AUC': roc_auc_score(y_true, probability),
        'Average Precision': average_precision_score(y_true, probability),
        'Prevalence': float(np.mean(y_true)),
        'Positive Support': int(np.sum(y_true)),
        'Threshold': threshold,
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
    }


# %% 2. Load exactly the same split as Random Forest
for required in [
    'shared_explanation_indices.csv', 'rf_selected_case.csv',
    'RandomForest_metrics.csv', 'rf_feature_importance.csv',
    'train_indices.csv', 'validation_indices.csv', 'test_indices.csv',
]:
    if not (RF_DIR / required).exists():
        raise FileNotFoundError(f'Missing {RF_DIR / required}. Run 03_shap.py first.')

loader = DataLoader()
df = loader.load_dataset(str(DATA_PATH))
X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val = (
    loader.get_train_val_test_split(random_state=RANDOM_STATE)
)
print('Train / Validation / Test:', len(X_train), len(X_val), len(X_test))

# Check saved RF split membership; these CSVs should contain the original dataframe indices.
def saved_indices(path):
    frame = pd.read_csv(path)
    for candidate in ['Dataset Index', 'Index', 'index']:
        if candidate in frame.columns:
            return frame[candidate].to_numpy()
    if frame.shape[1] == 1:
        return frame.iloc[:, 0].to_numpy()
    raise ValueError(f'Cannot identify index column in {path}; columns: {frame.columns.tolist()}')

for filename, actual in [
    ('train_indices.csv', X_train.index),
    ('validation_indices.csv', X_val.index),
    ('test_indices.csv', X_test.index),
]:
    stored = saved_indices(RF_DIR / filename)
    if set(stored.tolist()) != set(actual.tolist()):
        raise ValueError(f'Split mismatch with Random Forest: {filename}')

shared_indices = saved_indices(RF_DIR / 'shared_explanation_indices.csv')
if len(shared_indices) != len(set(shared_indices.tolist())):
    raise ValueError('Duplicate explanation indices in shared_explanation_indices.csv')
if not set(shared_indices.tolist()).issubset(set(X_test.index.tolist())):
    raise ValueError('Shared explanation indices are not all in the test set')
X_explain_raw = X_test.loc[shared_indices].copy()
y_explain = y_test.loc[shared_indices].copy()

rf_case = pd.read_csv(RF_DIR / 'rf_selected_case.csv')
if 'Patient Index' not in rf_case.columns:
    raise ValueError('rf_selected_case.csv must contain Patient Index')
patient_index = int(rf_case.loc[0, 'Patient Index'])
if patient_index not in X_explain_raw.index:
    raise ValueError('Selected RF patient not found in shared explanation indices')
patient_position = int(X_explain_raw.index.get_loc(patient_index))

# %% 3. Tune SVM using training data only
# RandomOverSampler stays INSIDE the pipeline, so it is applied only to CV training folds.
base_pipeline = build_pipeline(
    X_train,
    SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    random_state=RANDOM_STATE,
)
param_distributions = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 0.001, 0.01, 0.1],
    'classifier__class_weight': [None, 'balanced'],
}
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
search = RandomizedSearchCV(
    estimator=base_pipeline,
    param_distributions=param_distributions,
    n_iter=SEARCH_ITERATIONS,
    scoring='average_precision',
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
    error_score='raise',
)
search.fit(X_train, y_train)
print('Best SVM Parameters:', search.best_params_)
print('Best CV Average Precision:', search.best_score_)
save_csv(pd.DataFrame(search.cv_results_).sort_values('rank_test_score'), 'svm_hyperparameter_tuning.csv')

# %% 4. Select threshold on validation, then refit on train + validation
val_probability = search.best_estimator_.predict_proba(X_val)[:, 1]
threshold_candidates = np.unique(np.concatenate(([0.0, 1.0], val_probability)))
val_f1 = np.array([
    f1_score(y_val, val_probability >= threshold, zero_division=0)
    for threshold in threshold_candidates
])
best_threshold = float(threshold_candidates[int(np.argmax(val_f1))])
print('Selected SVM Threshold:', best_threshold)

final_model = build_pipeline(
    X_train_val,
    SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    random_state=RANDOM_STATE,
)
final_model.set_params(**search.best_params_)
final_model.fit(X_train_val, y_train_val)
joblib.dump(final_model, SAVE_DIR / 'svm_pipeline.joblib')

test_probability = final_model.predict_proba(X_test)[:, 1]
svm_metrics = pd.DataFrame([metrics_row(y_test, test_probability, best_threshold, 'SVM')])
save_csv(svm_metrics, 'SVM_metrics.csv')
print('\nSVM Test Performance:\n', svm_metrics.to_string(index=False))

prediction = (test_probability >= best_threshold).astype(int)
ConfusionMatrixDisplay.from_predictions(y_test, prediction, labels=[0, 1], cmap='Blues', values_format='d')
plt.title('SVM Confusion Matrix (Held-out Test Set)')
plt.tight_layout()
plt.savefig(SAVE_DIR / 'SVM_confusion_matrix.png', dpi=200)
plt.close()

# %% 5. Prepare transformed features and Kernel SHAP background
# Preprocessor is fitted on the final training split, never on test data.
preprocessor = final_model.named_steps['preprocessor']
feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
X_explain = pd.DataFrame(
    preprocessor.transform(X_explain_raw),
    columns=feature_names, index=X_explain_raw.index,
)
background_raw = X_train_val.sample(n=min(BACKGROUND_SIZE, len(X_train_val)), random_state=RANDOM_STATE)
background = pd.DataFrame(
    preprocessor.transform(background_raw),
    columns=feature_names, index=background_raw.index,
)
save_csv(pd.DataFrame({'Dataset Index': background_raw.index}), 'svm_background_indices.csv')
np.save(SAVE_DIR / 'svm_background.npy', background.to_numpy())
save_csv(pd.DataFrame({'Feature': feature_names}), 'svm_feature_names.csv')

# IMPORTANT: KernelExplainer receives transformed arrays, so use only the SVC here.
# Passing the full pipeline would apply preprocessing twice.
svc = final_model.named_steps['classifier']
positive_class_index = int(np.where(svc.classes_ == 1)[0][0])
def predict_positive(transformed_features):
    return svc.predict_proba(np.asarray(transformed_features))[:, positive_class_index]

# %% 6. Kernel SHAP for exactly the same 60 test samples
print(f'\nCalculating Kernel SHAP for {len(X_explain)} samples...')
start = time.perf_counter()
explainer = shap.KernelExplainer(predict_positive, background.to_numpy(), link='identity')
raw_shap = explainer.shap_values(X_explain.to_numpy(), nsamples=KERNEL_NSAMPLES, silent=True)
shap_values = positive_shap_values(raw_shap, len(X_explain), len(feature_names))
shap_runtime = time.perf_counter() - start
np.save(SAVE_DIR / 'svm_shap_values.npy', shap_values)

expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
model_probability = predict_positive(X_explain.to_numpy())
shap_reconstructed = expected_value + shap_values.sum(axis=1)
residuals = np.abs(shap_reconstructed - model_probability)
verification = pd.DataFrame({
    'Dataset Index': X_explain.index,
    'Model Probability': model_probability,
    'SHAP Reconstructed Probability': shap_reconstructed,
    'Absolute Residual': residuals,
})
save_csv(verification, 'svm_shap_verification.csv')
print('Kernel SHAP Runtime (seconds):', round(shap_runtime, 2))
print('Maximum Kernel SHAP Residual:', float(residuals.max()))
print('Approximation Diagnostic (<= 0.05):', bool(np.all(residuals <= SHAP_TOLERANCE)))

# %% 7. Local explanation — exactly the same patient as RF
patient_shap = shap_values[patient_position]
patient_prob = float(model_probability[patient_position])
patient_true = int(y_explain.iloc[patient_position])
patient_prediction = int(patient_prob >= best_threshold)
case_type = {
    (1, 1): 'True Positive', (0, 1): 'False Positive',
    (1, 0): 'False Negative', (0, 0): 'True Negative',
}[(patient_true, patient_prediction)]

local = pd.DataFrame({
    'Feature': feature_names,
    'Feature Value': X_explain.iloc[patient_position].to_numpy(),
    'SHAP Value': patient_shap,
})
local['Direction'] = np.where(local['SHAP Value'] >= 0, 'Increases', 'Decreases')
local = local.reindex(local['SHAP Value'].abs().sort_values(ascending=False).index).reset_index(drop=True)
save_csv(local, 'svm_local_contributions.csv')
save_csv(pd.DataFrame([{
    'Patient Index': patient_index,
    'Actual Stroke': patient_true,
    'Predicted Stroke': patient_prediction,
    'Case Type': case_type,
    'Probability': patient_prob,
    'Threshold': best_threshold,
    'SHAP Expected Value': expected_value,
}]), 'svm_selected_case.csv')
save_csv(X_explain_raw.loc[[patient_index]].reset_index().rename(columns={'index': 'Dataset Index'}),
         'svm_selected_patient_original_values.csv')

force = shap.force_plot(
    expected_value, patient_shap, X_explain.iloc[patient_position],
    feature_names=feature_names.tolist(), matplotlib=False,
)
shap.save_html(str(SAVE_DIR / 'svm_local_force_plot.html'), force)
plt.figure(figsize=(15, 4))
shap.force_plot(
    expected_value, patient_shap, X_explain.iloc[patient_position],
    feature_names=feature_names.tolist(), matplotlib=True, show=False,
)
plt.tight_layout()
plt.savefig(SAVE_DIR / 'svm_local_force_plot.png', dpi=200, bbox_inches='tight')
plt.close('all')
print(f'\nLocal Patient: {patient_index} | {case_type} | Probability: {patient_prob:.6f}')
print(local[['Feature', 'SHAP Value', 'Direction']].head(3).to_string(index=False))

# %% 8. Global Kernel SHAP: beeswarm, importance and feature patterns
explanation = shap.Explanation(
    values=shap_values,
    base_values=np.repeat(expected_value, len(X_explain)),
    data=X_explain.to_numpy(),
    feature_names=feature_names.tolist(),
)
shap.plots.beeswarm(explanation, max_display=min(15, len(feature_names)), show=False)
plt.tight_layout()
plt.savefig(SAVE_DIR / 'svm_global_beeswarm.png', dpi=200, bbox_inches='tight')
plt.close('all')

importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean Absolute SHAP': np.mean(np.abs(shap_values), axis=0),
}).sort_values('Mean Absolute SHAP', ascending=False).reset_index(drop=True)
importance.insert(0, 'Rank', np.arange(1, len(importance) + 1))
save_csv(importance, 'svm_feature_importance.csv')

shap.plots.bar(explanation.abs.mean(0), max_display=min(15, len(feature_names)), show=False)
plt.tight_layout()
plt.savefig(SAVE_DIR / 'svm_global_importance.png', dpi=200, bbox_inches='tight')
plt.close('all')

patterns = []
for feature in importance.head(5)['Feature']:
    j = int(np.where(feature_names == feature)[0][0])
    values = X_explain.iloc[:, j].to_numpy(dtype=float)
    median = float(np.median(values))
    low = values <= median
    high = values > median
    # Binary one-hot features can have only one observed level in a small sample.
    patterns.append({
        'Feature': feature,
        'Median Feature Value': median,
        'Low Group Size': int(low.sum()),
        'High Group Size': int(high.sum()),
        'Low Group Mean SHAP': float(np.mean(shap_values[low, j])) if low.any() else np.nan,
        'High Group Mean SHAP': float(np.mean(shap_values[high, j])) if high.any() else np.nan,
        'Interpretation Note': 'One-hot indicator: 0/1 are category absence/presence' if feature not in ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease'] else 'Check distribution and encoding',
    })
save_csv(pd.DataFrame(patterns), 'svm_top5_patterns.csv')

# %% 9. RF vs SVM: performance, global and local SHAP
rf_metrics = pd.read_csv(RF_DIR / 'RandomForest_metrics.csv')
comparison_metrics = pd.concat([rf_metrics, svm_metrics], ignore_index=True)
save_csv(comparison_metrics, 'rf_vs_svm_metrics.csv')

rf_importance = pd.read_csv(RF_DIR / 'rf_feature_importance.csv')
if not {'Feature', 'Mean Absolute SHAP'}.issubset(rf_importance.columns):
    raise ValueError('RF feature importance CSV has unexpected columns')
global_comparison = rf_importance[['Feature', 'Mean Absolute SHAP']].rename(
    columns={'Mean Absolute SHAP': 'RF Mean Absolute SHAP'}
).merge(
    importance[['Feature', 'Mean Absolute SHAP']].rename(columns={'Mean Absolute SHAP': 'SVM Mean Absolute SHAP'}),
    on='Feature', how='outer',
)
global_comparison['RF Rank'] = global_comparison['RF Mean Absolute SHAP'].rank(ascending=False, method='min')
global_comparison['SVM Rank'] = global_comparison['SVM Mean Absolute SHAP'].rank(ascending=False, method='min')
save_csv(global_comparison.sort_values('RF Rank'), 'rf_vs_svm_global_shap.csv')

rf_local = pd.read_csv(RF_DIR / 'rf_local_contributions.csv')
if not {'Feature', 'SHAP Value'}.issubset(rf_local.columns):
    raise ValueError('RF local contributions CSV has unexpected columns')
local_comparison = rf_local[['Feature', 'SHAP Value']].rename(
    columns={'SHAP Value': 'RF SHAP Value'}
).merge(
    local[['Feature', 'SHAP Value']].rename(columns={'SHAP Value': 'SVM SHAP Value'}),
    on='Feature', how='outer',
)
save_csv(local_comparison, 'rf_vs_svm_local_shap.csv')

rf_patient_probability = float(rf_case.loc[0, 'Probability']) if 'Probability' in rf_case.columns else np.nan
save_csv(pd.DataFrame([{
    'Patient Index': patient_index,
    'True Label': patient_true,
    'RF Probability': rf_patient_probability,
    'SVM Probability': patient_prob,
    'SVM Threshold': best_threshold,
    'SVM Case Type': case_type,
}]), 'rf_vs_svm_patient_comparison.csv')

# %% 10. Reproducibility information
config = {
    'Model': 'SVC', 'Explainer': 'KernelExplainer',
    'Random State': RANDOM_STATE,
    'CV Folds': CV_FOLDS, 'CV Scoring': 'average_precision',
    'Search Iterations': SEARCH_ITERATIONS,
    'Best Parameters': search.best_params_,
    'Best CV Average Precision': float(search.best_score_),
    'Validation Threshold': best_threshold,
    'Train Rows': len(X_train), 'Validation Rows': len(X_val),
    'Test Rows': len(X_test), 'Final Training Rows': len(X_train_val),
    'Explanation Rows': len(X_explain),
    'Explanation Positive Cases': int(y_explain.sum()),
    'Explanation Source': 'Shared RF held-out test indices',
    'Background Rows': len(background),
    'Background Source': 'Final training data, before oversampling',
    'Kernel SHAP nsamples': KERNEL_NSAMPLES,
    'Kernel SHAP Link': 'identity',
    'SHAP Output': 'Stroke=1 predicted probability',
    'SHAP Expected Value': expected_value,
    'SHAP Runtime Seconds': shap_runtime,
    'Maximum SHAP Residual': float(residuals.max()),
    'SHAP Approximation Diagnostic Tolerance': SHAP_TOLERANCE,
    'Selected Patient Index': patient_index,
    'Feature Count': len(feature_names),
    'Python Version': sys.version.split()[0],
    'SHAP Version': shap.__version__,
    'Scikit-learn Version': sklearn.__version__,
    'Imbalanced-learn Version': imblearn.__version__,
    'Pandas Version': pd.__version__,
    'Numpy Version': np.__version__,
    'Operating System': platform.platform(),
    'Dataset Path': str(DATA_PATH),
    'Dataset SHA256 Fingerprint': hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
    'Probability Calibration Assessed': False,
}
with open(SAVE_DIR / 'svm_configuration.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
with open(SAVE_DIR / 'environment_versions.txt', 'w', encoding='utf-8') as f:
    for key in ['Python Version', 'SHAP Version', 'Scikit-learn Version',
                'Imbalanced-learn Version', 'Pandas Version', 'Numpy Version', 'Operating System']:
        f.write(f'{key}: {config[key]}\n')

print('\nTop 5 SVM Global Features:\n', importance.head(5).to_string(index=False))
print('\nTask 6 completed. Results saved to:', SAVE_DIR)
