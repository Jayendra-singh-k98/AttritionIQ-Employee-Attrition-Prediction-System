from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from collections import Counter
import math
import json
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)


class ID3Node:
    
    def __init__(self):
        self.feature       = None    
        self.feature_name  = None    
        self.threshold     = None    
        self.children      = {}      
        self.label         = None    
        self.is_leaf       = False
        self.entropy       = 0.0     
        self.info_gain     = 0.0      
        self.samples       = 0       
        self.class_counts  = {}      
        self.depth         = 0


def entropy(y):
    """Shannon entropy: H(S) = -Σ p_i * log2(p_i)"""
    if len(y) == 0:
        return 0.0
    counts = Counter(y)
    total  = len(y)
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def information_gain(y, y_left, y_right):
    """
    IG(S, A) = H(S) - Σ |S_v|/|S| * H(S_v)
    Measures how much entropy is reduced by splitting on feature A.
    """
    n       = len(y)
    n_left  = len(y_left)
    n_right = len(y_right)
    if n == 0:
        return 0.0
    parent_entropy = entropy(y)
    child_entropy  = (n_left / n) * entropy(y_left) + (n_right / n) * entropy(y_right)
    return parent_entropy - child_entropy


def best_split(X, y, feature_names, min_samples_split=10):
    """
    Find the best (feature, threshold) pair by maximising Information Gain.
    For each numeric feature we try all midpoints between consecutive sorted values.
    Returns: (best_feature_idx, best_threshold, best_gain, best_feature_name)
    """
    best_gain    = -1
    best_feature = None
    best_thresh  = None
    best_name    = None
    n_features   = X.shape[1]

    for feat_idx in range(n_features):
        values = np.sort(np.unique(X[:, feat_idx]))
        if len(values) < 2:
            continue
   
        thresholds = (values[:-1] + values[1:]) / 2.0

        for thresh in thresholds:
            left_mask  = X[:, feat_idx] <= thresh
            right_mask = ~left_mask
            if left_mask.sum() < min_samples_split or right_mask.sum() < min_samples_split:
                continue
            gain = information_gain(y, y[left_mask], y[right_mask])
            if gain > best_gain:
                best_gain    = gain
                best_feature = feat_idx
                best_thresh  = thresh
                best_name    = feature_names[feat_idx]

    return best_feature, best_thresh, best_gain, best_name


def build_id3(X, y, feature_names, max_depth=8, min_samples_split=10,
              min_gain=1e-5, depth=0):
    """
    Recursively build the ID3 tree.
    Stopping conditions:
      - Pure node (entropy == 0)
      - max_depth reached
      - Too few samples to split
      - No information gain available
    """
    node = ID3Node()
    node.depth         = depth
    node.samples       = len(y)
    node.entropy       = entropy(y)
    node.class_counts  = dict(Counter(y))
    majority_class     = max(Counter(y), key=Counter(y).get) if len(y) > 0 else 0
    node.label         = majority_class

    
    if (node.entropy == 0 or depth >= max_depth
            or len(y) < min_samples_split):
        node.is_leaf = True
        return node

    feat_idx, thresh, gain, feat_name = best_split(
        X, y, feature_names, min_samples_split)

    if feat_idx is None or gain < min_gain:
        node.is_leaf = True
        return node

    node.feature      = feat_idx
    node.feature_name = feat_name
    node.threshold    = thresh
    node.info_gain    = gain

    left_mask  = X[:, feat_idx] <= thresh
    right_mask = ~left_mask

    node.children['left']  = build_id3(
        X[left_mask], y[left_mask], feature_names,
        max_depth, min_samples_split, min_gain, depth + 1)
    node.children['right'] = build_id3(
        X[right_mask], y[right_mask], feature_names,
        max_depth, min_samples_split, min_gain, depth + 1)

    return node


def predict_single(node, x):
    """Traverse the tree for a single sample x, return (label, leaf_node)."""
    if node.is_leaf:
        return node.label, node
    if x[node.feature] <= node.threshold:
        return predict_single(node.children['left'], x)
    else:
        return predict_single(node.children['right'], x)


def predict_proba_single(node, x):
    """Return probability of class=1 based on leaf class distribution."""
    _, leaf = predict_single(node, x)
    total = sum(leaf.class_counts.values())
    if total == 0:
        return 0.5
    return leaf.class_counts.get(1, 0) / total


def predict_batch(node, X):
    """Predict labels for all rows in X."""
    return np.array([predict_single(node, X[i])[0] for i in range(len(X))])


def predict_proba_batch(node, X):
    """Predict class-1 probability for all rows in X."""
    return np.array([predict_proba_single(node, X[i]) for i in range(len(X))])


def compute_feature_importance(node, feature_names, importance=None):
    """
    Feature importance = total weighted information gain attributed to each feature
    across all splits in the tree, normalised to sum to 1.
    """
    if importance is None:
        importance = {name: 0.0 for name in feature_names}
    if node.is_leaf:
        return importance
    if node.feature_name:
        importance[node.feature_name] += node.info_gain * node.samples
    for child in node.children.values():
        compute_feature_importance(child, feature_names, importance)
    return importance


def tree_to_dict(node, depth=0):
    """Serialize the ID3 tree to a JSON-serialisable dict (for the API)."""
    d = {
        'depth':        node.depth,
        'samples':      node.samples,
        'entropy':      round(node.entropy, 4),
        'is_leaf':      node.is_leaf,
        'label':        int(node.label) if node.label is not None else None,
        'class_counts': {str(k): int(v) for k, v in node.class_counts.items()},
    }
    if not node.is_leaf:
        d['feature']    = node.feature_name
        d['threshold']  = round(float(node.threshold), 4)
        d['info_gain']  = round(float(node.info_gain), 6)
        d['children']   = {k: tree_to_dict(v) for k, v in node.children.items()}
    return d


# ══════════════════════════════════════════════════════════
#  DATASET GENERATION
# ══════════════════════════════════════════════════════════
np.random.seed(42)
N = 1000

def generate_dataset():
    departments  = ['Engineering', 'Sales', 'HR', 'Marketing', 'Finance', 'Operations']
    salary_bands = ['Low', 'Medium', 'High']
    data = {
        'Age':                    np.random.randint(22, 60, N),
        'YearsAtCompany':         np.random.randint(0, 30, N),
        'MonthlyIncome':          np.random.randint(2000, 20000, N),
        'JobSatisfaction':        np.random.randint(1, 5, N),
        'WorkLifeBalance':        np.random.randint(1, 5, N),
        'OverTime':               np.random.choice(['Yes', 'No'], N, p=[0.3, 0.7]),
        'NumProjectsHandled':     np.random.randint(1, 10, N),
        'TrainingLastYear':       np.random.randint(0, 7, N),
        'DistanceFromHome':       np.random.randint(1, 30, N),
        'PerformanceRating':      np.random.randint(1, 5, N),
        'Department':             np.random.choice(departments, N),
        'SalaryBand':             np.random.choice(salary_bands, N, p=[0.35, 0.45, 0.20]),
        'RelationshipSatisfaction': np.random.randint(1, 5, N),
        'YearsSinceLastPromotion':  np.random.randint(0, 15, N),
    }
    df = pd.DataFrame(data)
    attrition_score = (
        (df['JobSatisfaction'] <= 2).astype(int) * 2 +
        (df['WorkLifeBalance'] <= 2).astype(int) * 2 +
        (df['OverTime'] == 'Yes').astype(int) * 2 +
        (df['SalaryBand'] == 'Low').astype(int) * 2 +
        (df['YearsAtCompany'] <= 2).astype(int) +
        (df['YearsSinceLastPromotion'] >= 5).astype(int) +
        (df['DistanceFromHome'] >= 20).astype(int) +
        (df['NumProjectsHandled'] >= 8).astype(int) +
        np.random.randint(0, 3, N)
    )
    df['Attrition'] = (attrition_score >= 5).astype(int)
    return df

df = generate_dataset()

# ══════════════════════════════════════════════════════════
#  ENCODING & TRAINING
# ══════════════════════════════════════════════════════════
OVERTIME_MAP = {'Yes': 1, 'No': 0}
DEPT_MAP     = {d: i for i, d in enumerate(sorted(df['Department'].unique()))}
SALARY_MAP   = {'Low': 0, 'Medium': 1, 'High': 2}

df_model = df.copy()
df_model['OverTime_enc']    = df_model['OverTime'].map(OVERTIME_MAP)
df_model['Department_enc']  = df_model['Department'].map(DEPT_MAP)
df_model['SalaryBand_enc']  = df_model['SalaryBand'].map(SALARY_MAP)

FEATURE_COLS = [
    'Age', 'YearsAtCompany', 'MonthlyIncome', 'JobSatisfaction',
    'WorkLifeBalance', 'NumProjectsHandled', 'TrainingLastYear',
    'DistanceFromHome', 'PerformanceRating', 'RelationshipSatisfaction',
    'YearsSinceLastPromotion', 'OverTime_enc', 'Department_enc', 'SalaryBand_enc'
]
FEATURE_NAMES = [
    'Age', 'Years at Company', 'Monthly Income', 'Job Satisfaction',
    'Work-Life Balance', 'Projects Handled', 'Training Sessions',
    'Distance From Home', 'Performance Rating', 'Relationship Satisfaction',
    'Years Since Promotion', 'Overtime', 'Department', 'Salary Band'
]

X_all = df_model[FEATURE_COLS].values.astype(float)
y_all = df_model['Attrition'].values

# Stratified 80/20 train-test split
from collections import defaultdict
idx_by_class = defaultdict(list)
for i, label in enumerate(y_all):
    idx_by_class[label].append(i)

train_idx, test_idx = [], []
rng = np.random.default_rng(42)
for label, idxs in idx_by_class.items():
    idxs = np.array(idxs)
    rng.shuffle(idxs)
    split = int(0.8 * len(idxs))
    train_idx.extend(idxs[:split])
    test_idx.extend(idxs[split:])

train_idx = np.array(train_idx)
test_idx  = np.array(test_idx)

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

print("Training ID3 tree…")
id3_root = build_id3(
    X_train, y_train, FEATURE_NAMES,
    max_depth=8, min_samples_split=15, min_gain=1e-5
)
print("ID3 training complete.")

# ── Evaluation ──
y_pred = predict_batch(id3_root, X_test)
y_prob = predict_proba_batch(id3_root, X_test)

def safe_metric(fn, *args, **kwargs):
    try:
        return round(float(fn(*args, **kwargs)) * 100, 2)
    except Exception:
        return 0.0

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

METRICS = {
    'accuracy':        safe_metric(accuracy_score,  y_test, y_pred),
    'precision':       safe_metric(precision_score, y_test, y_pred, zero_division=0),
    'recall':          safe_metric(recall_score,    y_test, y_pred, zero_division=0),
    'f1':              safe_metric(f1_score,         y_test, y_pred, zero_division=0),
    'auc_roc':         safe_metric(roc_auc_score,   y_test, y_prob),
    'total_employees': N,
    'attrition_count': int(df['Attrition'].sum()),
    'retention_count': int((df['Attrition'] == 0).sum()),
    'attrition_rate':  round(df['Attrition'].mean() * 100, 2),
    # ID3-specific metadata
    'algorithm':       'ID3 (Information Gain)',
    'criterion':       'Entropy / Information Gain',
    'max_depth':       8,
    'min_samples_split': 15,
    'root_entropy':    round(entropy(y_train), 6),
    'dataset_entropy': round(entropy(y_all), 6),
}

cm_data   = confusion_matrix(y_test, y_pred).tolist()
CONF_MATRIX = {'matrix': cm_data, 'labels': ['Retained', 'Attrited']}

# Feature importance
raw_importance = compute_feature_importance(id3_root, FEATURE_NAMES)
total_imp = sum(raw_importance.values()) or 1
FEATURE_IMPORTANCE = dict(sorted(
    {k: round(v / total_imp * 100, 2) for k, v in raw_importance.items()}.items(),
    key=lambda x: x[1], reverse=True
))

# Entropy gain per feature (for the ID3-specific "entropy breakdown" endpoint)
ENTROPY_BREAKDOWN = []
for feat_idx, feat_name in enumerate(FEATURE_NAMES):
    col = X_all[:, feat_idx]
    median_thresh = np.median(col)
    left  = y_all[col <= median_thresh]
    right = y_all[col >  median_thresh]
    gain  = information_gain(y_all, left, right)
    ENTROPY_BREAKDOWN.append({
        'feature':   feat_name,
        'gain':      round(gain, 6),
        'gain_pct':  round(gain / (entropy(y_all) or 1) * 100, 2),
        'left_entropy':  round(entropy(left), 4),
        'right_entropy': round(entropy(right), 4),
    })
ENTROPY_BREAKDOWN.sort(key=lambda x: x['gain'], reverse=True)

print(f"ID3 Accuracy: {METRICS['accuracy']}%  |  AUC-ROC: {METRICS['auc_roc']}%")

# ══════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify({'success': True, 'metrics': METRICS})

@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    return jsonify({'success': True, 'features': FEATURE_IMPORTANCE})

@app.route('/api/confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    return jsonify({'success': True, 'data': CONF_MATRIX})

@app.route('/api/entropy-breakdown', methods=['GET'])
def get_entropy_breakdown():
    """ID3-specific: per-feature information gain breakdown."""
    return jsonify({'success': True, 'data': ENTROPY_BREAKDOWN,
                    'dataset_entropy': METRICS['dataset_entropy'],
                    'root_entropy': METRICS['root_entropy']})

@app.route('/api/tree-structure', methods=['GET'])
def get_tree_structure():
    """Returns the top 3 levels of the ID3 tree as JSON for visualisation."""
    def trim(node, max_d=3, cur_d=0):
        d = {
            'feature':     node.feature_name,
            'threshold':   round(float(node.threshold), 2) if node.threshold is not None else None,
            'info_gain':   round(float(node.info_gain), 4) if node.info_gain else 0,
            'entropy':     round(node.entropy, 4),
            'samples':     node.samples,
            'is_leaf':     node.is_leaf,
            'label':       int(node.label) if node.label is not None else None,
            'class_counts': {str(k): int(v) for k, v in node.class_counts.items()},
        }
        if not node.is_leaf and cur_d < max_d:
            d['children'] = {k: trim(v, max_d, cur_d + 1) for k, v in node.children.items()}
        return d
    return jsonify({'success': True, 'tree': trim(id3_root)})

@app.route('/api/department-stats', methods=['GET'])
def get_department_stats():
    stats = df.groupby('Department').agg(
        total=('Attrition', 'count'), attrited=('Attrition', 'sum')
    ).reset_index()
    stats['rate'] = (stats['attrited'] / stats['total'] * 100).round(2)
    return jsonify({'success': True, 'data': stats.to_dict(orient='records')})

@app.route('/api/salary-stats', methods=['GET'])
def get_salary_stats():
    stats = df.groupby('SalaryBand').agg(
        total=('Attrition', 'count'), attrited=('Attrition', 'sum')
    ).reset_index()
    stats['rate'] = (stats['attrited'] / stats['total'] * 100).round(2)
    return jsonify({'success': True, 'data': stats.to_dict(orient='records')})

@app.route('/api/satisfaction-trend', methods=['GET'])
def get_satisfaction_trend():
    result = []
    for score in [1, 2, 3, 4]:
        subset = df[df['JobSatisfaction'] == score]
        rate = round(subset['Attrition'].mean() * 100, 2)
        result.append({'satisfaction': score, 'attrition_rate': rate, 'count': len(subset)})
    return jsonify({'success': True, 'data': result})

@app.route('/api/age-distribution', methods=['GET'])
def get_age_distribution():
    bins   = [20, 30, 35, 40, 45, 50, 60]
    labels = ['20-29', '30-34', '35-39', '40-44', '45-49', '50-60']
    df2 = df.copy()
    df2['AgeGroup'] = pd.cut(df2['Age'], bins=bins, labels=labels, right=False)
    stats = df2.groupby('AgeGroup', observed=True).agg(
        total=('Attrition', 'count'), attrited=('Attrition', 'sum')
    ).reset_index()
    stats['rate'] = (stats['attrited'] / stats['total'] * 100).round(2)
    return jsonify({'success': True, 'data': stats.to_dict(orient='records')})

@app.route('/api/dataset-sample', methods=['GET'])
def get_dataset_sample():
    sample = df.sample(10, random_state=1)[[
        'Age', 'Department', 'SalaryBand', 'JobSatisfaction',
        'WorkLifeBalance', 'OverTime', 'YearsAtCompany', 'MonthlyIncome', 'Attrition'
    ]].copy()
    sample['Attrition'] = sample['Attrition'].map({1: 'Yes', 0: 'No'})
    return jsonify({'success': True, 'data': sample.to_dict(orient='records')})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        p = request.get_json()
        ot   = OVERTIME_MAP.get(p.get('overtime', 'No'), 0)
        dept = DEPT_MAP.get(p.get('department', 'Engineering'), 0)
        sal  = SALARY_MAP.get(p.get('salary_band', 'Medium'), 1)

        x = np.array([[
            float(p.get('age', 30)),
            float(p.get('years_at_company', 3)),
            float(p.get('monthly_income', 5000)),
            float(p.get('job_satisfaction', 3)),
            float(p.get('work_life_balance', 3)),
            float(p.get('num_projects', 4)),
            float(p.get('training_last_year', 2)),
            float(p.get('distance_from_home', 10)),
            float(p.get('performance_rating', 3)),
            float(p.get('relationship_satisfaction', 3)),
            float(p.get('years_since_promotion', 2)),
            float(ot), float(dept), float(sal)
        ]])

        pred, leaf = predict_single(id3_root, x[0])
        prob = predict_proba_single(id3_root, x[0])
        risk = 'Low' if prob < 0.35 else ('Medium' if prob < 0.65 else 'High')

        # Trace the decision path through the tree
        path = []
        node = id3_root
        while not node.is_leaf:
            val = x[0][node.feature]
            direction = 'left' if val <= node.threshold else 'right'
            path.append({
                'feature':   node.feature_name,
                'threshold': round(float(node.threshold), 2),
                'value':     round(float(val), 2),
                'direction': direction,
                'gain':      round(float(node.info_gain), 4),
            })
            node = node.children[direction]

        # Rule-based risk factors
        factors = []
        if p.get('overtime') == 'Yes':               factors.append('Overtime work detected')
        if int(p.get('job_satisfaction', 3)) <= 2:    factors.append('Low job satisfaction')
        if int(p.get('work_life_balance', 3)) <= 2:   factors.append('Poor work-life balance')
        if p.get('salary_band') == 'Low':             factors.append('Below-market salary band')
        if int(p.get('years_since_promotion', 2)) >= 5: factors.append('No promotion in 5+ years')
        if int(p.get('years_at_company', 3)) <= 2:    factors.append('New employee — high churn window')
        if int(p.get('num_projects', 4)) >= 8:        factors.append('Project overload (8+ projects)')
        if int(p.get('distance_from_home', 10)) >= 20: factors.append('Long commute distance')

        return jsonify({
            'success':      True,
            'prediction':   int(pred),
            'probability':  round(float(prob) * 100, 2),
            'risk_level':   risk,
            'risk_factors': factors,
            'decision_path': path,
            'leaf_entropy': round(leaf.entropy, 4),
            'leaf_samples': leaf.samples,
            'leaf_counts':  {str(k): int(v) for k, v in leaf.class_counts.items()},
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
